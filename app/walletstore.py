"""Wallet ledger storage.

SQLite is the default (local development). When ``WALLET_DATABASE_URL`` is
set, the wallet uses a hosted Postgres database instead, so the local app and
the Render deployment can read and write the same balance.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .config import WALLET_DATABASE_URL

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .storage import Store


class PostgresWalletStore:
    """Small Postgres-backed wallet ledger.

    Only the wallet lives here. Signals, orders and trades still use the local
    SQLite store; the shared balance is what makes LIVE/DEMO and local/Render
    agree.
    """

    backend = "postgres"

    def __init__(self, url: str = WALLET_DATABASE_URL) -> None:
        if not url:
            raise ValueError("WALLET_DATABASE_URL is empty")
        import psycopg

        self._psycopg = psycopg
        self._url = url
        self._lock = threading.Lock()
        self._conn = None
        self._connect()

    # ------------------------------------------------------------ connection
    def _connect(self):
        conn = self._psycopg.connect(self._url, autocommit=True, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS niftypulse_wallet_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    ts TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    balance_after DOUBLE PRECISION NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    ref TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_niftypulse_wallet_ledger_mode "
                "ON niftypulse_wallet_ledger(mode, id)"
            )
            # Imported local rows carry their SQLite id so re-running the
            # import cannot double-count a deposit.
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_niftypulse_wallet_ledger_source "
                "ON niftypulse_wallet_ledger(source, source_id) WHERE source <> ''"
            )
        self._conn = conn
        return conn

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _run(self, fn):
        """Run a function with the shared connection, reconnecting once."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    conn = self._conn or self._connect()
                    return fn(conn)
                except self._psycopg.Error:
                    self._close()
                    if attempt == 2:
                        raise
        raise RuntimeError("wallet database unavailable")

    @staticmethod
    def _lock_key(mode: str) -> str:
        return f"niftypulse-wallet:{mode}"

    # ----------------------------------------------------------------- reads
    def wallet_balance(self, mode: str) -> float:
        def read(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(SUM(amount), 0) FROM niftypulse_wallet_ledger WHERE mode = %s",
                    (mode,),
                )
                return float(cur.fetchone()[0] or 0.0)

        return self._run(read)

    def wallet_has_history(self, mode: str) -> bool:
        def read(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM niftypulse_wallet_ledger WHERE mode = %s LIMIT 1", (mode,)
                )
                return cur.fetchone() is not None

        return self._run(read)

    # ---------------------------------------------------------------- writes
    def wallet_append(
        self, mode: str, kind: str, amount: float, note: str, ref: str, ts: str
    ) -> dict:
        amount = round(float(amount), 2)

        def write(conn):
            with conn.transaction():
                with conn.cursor() as cur:
                    # Serialise balance calculations across local + Render.
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                        (self._lock_key(mode),),
                    )
                    cur.execute(
                        "SELECT COALESCE(SUM(amount), 0) FROM niftypulse_wallet_ledger WHERE mode = %s",
                        (mode,),
                    )
                    after = round(float(cur.fetchone()[0] or 0.0) + amount, 2)
                    cur.execute(
                        """
                        INSERT INTO niftypulse_wallet_ledger
                            (ts, mode, kind, amount, balance_after, note, ref)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (ts, mode, kind, amount, after, note, ref),
                    )
                    entry_id = int(cur.fetchone()[0])
            return {
                "id": entry_id,
                "ts": ts,
                "kind": kind,
                "amount": amount,
                "balance_after": after,
                "note": note,
                "ref": ref,
            }

        return self._run(write)

    def wallet_entries(self, mode: str, limit: int = 8) -> list[dict]:
        def read(conn):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, ts, kind, amount, balance_after, note, ref
                    FROM niftypulse_wallet_ledger WHERE mode = %s
                    ORDER BY id DESC LIMIT %s
                    """,
                    (mode, limit),
                )
                rows = cur.fetchall()
            return [
                {
                    "id": int(r[0]),
                    "ts": r[1],
                    "kind": r[2],
                    "amount": round(float(r[3]), 2),
                    "balance_after": round(float(r[4]), 2),
                    "note": r[5],
                    "ref": r[6],
                }
                for r in rows
            ]

        return self._run(read)

    # --------------------------------------------------------------- migration
    def import_from(self, sqlite_store: "Store", mode: str = "live") -> int:
        """Copy an existing SQLite ledger into Postgres exactly once.

        The local app already has the user's deposit. If Render starts first
        and seeds an empty wallet, a later import still copies only the local
        non-opening entries (and de-duplicates by SQLite row id).
        """
        entries = sqlite_store.wallet_all_entries(mode)
        if not entries:
            return 0
        # Keep local and Render SQLite ids from colliding in the shared table.
        source = f"sqlite-live:{getattr(sqlite_store, 'path', 'default')}"

        def write(conn):
            imported = 0
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                        (self._lock_key(mode),),
                    )
                    cur.execute(
                        "SELECT 1 FROM niftypulse_wallet_ledger WHERE mode = %s LIMIT 1",
                        (mode,),
                    )
                    has_history = cur.fetchone() is not None
                    for entry in reversed(entries):  # SQLite returns newest first
                        if entry.get("kind") == "opening" and has_history:
                            continue
                        source_id = str(entry["id"])
                        cur.execute(
                            "SELECT 1 FROM niftypulse_wallet_ledger "
                            "WHERE source = %s AND source_id = %s",
                            (source, source_id),
                        )
                        if cur.fetchone() is not None:
                            continue
                        amount = round(float(entry["amount"]), 2)
                        cur.execute(
                            "SELECT COALESCE(SUM(amount), 0) "
                            "FROM niftypulse_wallet_ledger WHERE mode = %s",
                            (mode,),
                        )
                        after = round(float(cur.fetchone()[0] or 0.0) + amount, 2)
                        cur.execute(
                            """
                            INSERT INTO niftypulse_wallet_ledger
                                (ts, mode, kind, amount, balance_after, note, ref,
                                 source, source_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                entry["ts"],
                                mode,
                                entry["kind"],
                                amount,
                                after,
                                entry["note"],
                                entry["ref"],
                                source,
                                source_id,
                            ),
                        )
                        has_history = True
                        imported += 1
            return imported

        return self._run(write)

    def close(self) -> None:
        with self._lock:
            self._close()


def build_wallet_store(sqlite_store: "Store"):
    """Use Postgres when configured, otherwise keep the local SQLite ledger."""
    if not WALLET_DATABASE_URL:
        return sqlite_store

    try:
        pg = PostgresWalletStore(WALLET_DATABASE_URL)
        try:
            pg.import_from(sqlite_store, "live")
        except Exception as exc:  # never stop the app for a best-effort import
            print(f"wallet: SQLite import skipped: {type(exc).__name__}: {exc}")
        return pg
    except Exception as exc:
        print(
            "wallet: Postgres unavailable, using local SQLite ledger: "
            f"{type(exc).__name__}: {exc}"
        )
        return sqlite_store
