"""SQLite persistence for signals and paper/live trades."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .config import DB_PATH


class Store:
    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    day TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    day TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_day ON trades(day)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    day TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_day ON orders(day)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wallet_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    amount REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    ref TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wallet_mode ON wallet_ledger(mode, id)"
            )

    def save_signal(self, signal: dict) -> None:
        day = signal["ts"][:10]
        payload = json.dumps(signal, default=str)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO signals (id, day, ts, symbol, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload
                """,
                (signal["id"], day, signal["ts"], signal["symbol"], signal["status"], payload),
            )

    def save_trade(self, trade: dict) -> None:
        day = trade["opened_at"][:10]
        payload = json.dumps(trade, default=str)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO trades (id, day, opened_at, symbol, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload
                """,
                (trade["id"], day, trade["opened_at"], trade["symbol"], trade["status"], payload),
            )

    def load_day(self, day: str) -> tuple[list[dict], list[dict]]:
        with self._lock:
            signal_rows = self._conn.execute(
                "SELECT payload FROM signals WHERE day = ? ORDER BY ts ASC", (day,)
            ).fetchall()
            trade_rows = self._conn.execute(
                "SELECT payload FROM trades WHERE day = ? ORDER BY opened_at ASC", (day,)
            ).fetchall()
        signals = [json.loads(row[0]) for row in signal_rows]
        trades = [json.loads(row[0]) for row in trade_rows]
        return signals, trades

    # ------------------------------------------------------------------ orders
    def save_order(self, order: dict) -> None:
        day = (order.get("ts") or "")[:10]
        payload = json.dumps(order, default=str)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO orders (id, day, ts, symbol, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload
                """,
                (order["id"], day, order["ts"], order["symbol"], order["status"], payload),
            )

    def load_orders(self, day: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM orders WHERE day = ? ORDER BY ts ASC", (day,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    # ------------------------------------------------------------------ wallet
    def wallet_balance(self, mode: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM wallet_ledger WHERE mode = ?", (mode,)
            ).fetchone()
        return float(row[0] or 0.0)

    def wallet_has_history(self, mode: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM wallet_ledger WHERE mode = ? LIMIT 1", (mode,)
            ).fetchone()
        return row is not None

    def wallet_append(
        self, mode: str, kind: str, amount: float, note: str, ref: str, ts: str
    ) -> dict:
        """Append a signed ledger entry and return it with the running balance."""
        amount = round(float(amount), 2)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM wallet_ledger WHERE mode = ?", (mode,)
            ).fetchone()
            after = round(float(row[0] or 0.0) + amount, 2)
            cur = self._conn.execute(
                """
                INSERT INTO wallet_ledger (ts, mode, kind, amount, balance_after, note, ref)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, mode, kind, amount, after, note, ref),
            )
            entry_id = cur.lastrowid
        return {
            "id": entry_id,
            "ts": ts,
            "kind": kind,
            "amount": amount,
            "balance_after": after,
            "note": note,
            "ref": ref,
        }

    def wallet_entries(self, mode: str, limit: int = 8) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ts, kind, amount, balance_after, note, ref
                FROM wallet_ledger WHERE mode = ?
                ORDER BY id DESC LIMIT ?
                """,
                (mode, limit),
            ).fetchall()
        return [
            {
                "ts": r[0],
                "kind": r[1],
                "amount": round(float(r[2]), 2),
                "balance_after": round(float(r[3]), 2),
                "note": r[4],
                "ref": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
