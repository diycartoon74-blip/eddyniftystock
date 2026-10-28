"""Cash wallet for the paper and live ledgers.

Both LIVE and DEMO use one ledger, so a deposit made in either mode is visible
in both. Cash is debited when an option is bought and credited when it is
sold; deposits and withdrawals are explicit ledger entries. In live mode the
wallet is a local allocation record - the broker still holds the actual funds.

Storage is SQLite by default. Set ``WALLET_DATABASE_URL`` to a hosted Postgres
database to share the same ledger between the local app and Render.
"""

from __future__ import annotations

import math

from .config import Settings
from .datafeed import now_ist
from .storage import Store
from .walletstore import PostgresWalletStore

KINDS = {"opening", "deposit", "withdraw", "entry", "exit", "adjust"}


class WalletError(ValueError):
    """Rejected wallet operation (bad amount, insufficient cash, ...)."""


class Wallet:
    MAX_TXN = 100_000_000.0  # Rs 10 crore per deposit/withdrawal
    LEDGER_MODE = "live"  # one shared ledger for LIVE and DEMO

    def __init__(
        self, store: Store | PostgresWalletStore, settings: Settings
    ) -> None:
        self.store = store
        self.settings = settings

    @property
    def backend(self) -> str:
        return getattr(self.store, "backend", "sqlite")

    def _key(self, mode: str) -> str:
        """Every display mode reads and writes the same ledger."""
        return self.LEDGER_MODE

    # ------------------------------------------------------------------ setup
    def ensure(self, mode: str) -> bool:
        """Seed an opening balance once. Returns True when seeded now."""
        key = self._key(mode)
        if self.store.wallet_has_history(key):
            return False
        self.store.wallet_append(
            key,
            "opening",
            float(self.settings.capital),
            "Opening balance (shared wallet)",
            "",
            now_ist().isoformat(timespec="seconds"),
        )
        return True

    # ------------------------------------------------------------------ reads
    def balance(self, mode: str) -> float:
        return round(self.store.wallet_balance(self._key(mode)), 2)

    def history(self, mode: str, limit: int = 6) -> list[dict]:
        return self.store.wallet_entries(self._key(mode), limit)

    # ----------------------------------------------------------------- writes
    def _post(self, mode: str, kind: str, amount: float, note: str, ref: str = "") -> dict:
        entry = self.store.wallet_append(
            self._key(mode), kind, amount, note, ref, now_ist().isoformat(timespec="seconds")
        )
        assert kind in KINDS
        return entry

    def _clean_amount(self, amount: float) -> float:
        try:
            value = float(amount)
        except (TypeError, ValueError):
            raise WalletError("amount must be a number") from None
        if not math.isfinite(value):
            raise WalletError("amount must be a finite number")
        value = round(value, 2)
        if value <= 0:
            raise WalletError("amount must be greater than zero")
        if value > self.MAX_TXN:
            raise WalletError(f"amount exceeds the {self.MAX_TXN:,.0f} per-transaction cap")
        return value

    def deposit(self, mode: str, amount: float, note: str = "") -> dict:
        value = self._clean_amount(amount)
        if self.balance(mode) + value > self.MAX_TXN * 10:
            raise WalletError("wallet balance cap reached")
        return self._post(mode, "deposit", value, note or "Money added")

    def withdraw(self, mode: str, amount: float, note: str = "") -> dict:
        value = self._clean_amount(amount)
        cash = self.balance(mode)
        if value > cash + 0.005:
            raise WalletError(f"only Rs {cash:,.2f} is available to withdraw")
        return self._post(mode, "withdraw", -value, note or "Money withdrawn")

    def debit(self, mode: str, amount: float, note: str, ref: str = "") -> dict:
        value = round(float(amount), 2)
        if value <= 0:
            raise WalletError("debit amount must be positive")
        cash = self.balance(mode)
        if value > cash + 0.005:
            raise WalletError(
                f"insufficient wallet balance: need Rs {value:,.2f}, have Rs {cash:,.2f}"
            )
        return self._post(mode, "entry", -value, note, ref)

    def credit(self, mode: str, amount: float, note: str, ref: str = "") -> dict:
        value = round(float(amount), 2)
        if value <= 0:
            raise WalletError("credit amount must be positive")
        return self._post(mode, "exit", value, note, ref)

    def adjust(self, mode: str, amount: float, note: str, ref: str = "") -> dict:
        """Book a non-trade adjustment (used once when migrating old trades)."""
        value = round(float(amount), 2)
        if value == 0:
            raise WalletError("adjustment must be non-zero")
        return self._post(mode, "adjust", value, note, ref)
