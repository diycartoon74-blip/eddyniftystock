"""Configuration and market constants.

Environment variables (optionally loaded from .env in the project root):

  HOST, PORT                 web server bind address (default 127.0.0.1:8000)
  DEMO_MODE=1                start on the simulated feed
  AUTO_TRADE=0               disable automatic execution on start
  ENABLE_LIVE_TRADING=1      allow real broker orders (requires BROKER=kite)
  BROKER=kite                broker adapter to use (paper | kite)
  KITE_API_KEY, KITE_ACCESS_TOKEN
  CAPITAL, RISK_PER_TRADE_PCT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEB_DIR = BASE_DIR / "web"
DB_PATH = DATA_DIR / "niftypulse.db"

IST = timezone(timedelta(hours=5, minutes=30))

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
EOD_SQUARE_OFF = time(15, 12)
NO_NEW_ENTRY_AFTER = time(15, 5)


def _load_dotenv() -> None:
    """Tiny .env loader so the app has no hard dependency on python-dotenv."""
    path = BASE_DIR / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SymbolConfig:
    key: str
    yahoo: str
    step: int
    lot_size: int
    demo_base: float
    demo_vol: float
    label: str
    chain_feed: str = "nse"
    chain_oi: int = 1_150_000
    chain_iv: float = 0.128
    monthly_expiry: bool = False
    expiry_weekday: int = 1  # Monday=0; NSE Tuesday, BSE Thursday


SYMBOLS: dict[str, SymbolConfig] = {
    "NIFTY": SymbolConfig(
        key="NIFTY",
        yahoo="^NSEI",
        step=50,
        lot_size=75,
        demo_base=26050.0,
        demo_vol=0.00062,
        label="NIFTY 50",
        chain_oi=2_400_000,
        chain_iv=0.112,
    ),
    "BANKNIFTY": SymbolConfig(
        key="BANKNIFTY",
        yahoo="^NSEBANK",
        step=100,
        lot_size=35,
        demo_base=59150.0,
        demo_vol=0.00095,
        label="BANK NIFTY",
        monthly_expiry=True,
    ),
    "SENSEX": SymbolConfig(
        key="SENSEX",
        yahoo="^BSESN",
        step=100,
        lot_size=20,
        demo_base=82150.0,
        demo_vol=0.00058,
        label="SENSEX",
        # BSE's derivatives API is fronted by Akamai, so it is fetched with a
        # browser-like client and falls back to the synthetic model on failure.
        chain_feed="bse",
        chain_oi=1_300_000,
        chain_iv=0.125,
        expiry_weekday=3,
    ),
}


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    poll_seconds: int = 15
    candle_cache_seconds: int = 12
    chain_cache_seconds: int = 30

    demo: bool = field(default_factory=lambda: env_bool("DEMO_MODE", False))
    auto_trade: bool = field(default_factory=lambda: env_bool("AUTO_TRADE", True))

    capital: float = field(default_factory=lambda: env_float("CAPITAL", 500000.0))
    risk_per_trade_pct: float = field(default_factory=lambda: env_float("RISK_PER_TRADE_PCT", 1.5))
    max_exposure_pct: float = 35.0
    max_trades_per_day: int = 4
    max_open_positions: int = 2
    daily_loss_limit_pct: float = 2.5

    confidence_threshold: int = 68
    demo_confidence_threshold: int = 58
    cooldown_minutes: int = 20
    demo_cooldown_minutes: int = 4

    stop_loss_pct: float = 20.0
    reward_risk: float = 1.8

    allow_live_trading: bool = field(default_factory=lambda: env_bool("ENABLE_LIVE_TRADING", False))
    broker: str = field(default_factory=lambda: os.getenv("BROKER", "paper").strip().lower())
    kite_api_key: str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    kite_access_token: str = field(default_factory=lambda: os.getenv("KITE_ACCESS_TOKEN", ""))

    @property
    def active_confidence_threshold(self) -> int:
        return self.demo_confidence_threshold if self.demo else self.confidence_threshold

    @property
    def active_cooldown_minutes(self) -> int:
        return self.demo_cooldown_minutes if self.demo else self.cooldown_minutes

    @property
    def live_execution(self) -> bool:
        return (
            self.broker == "kite"
            and self.allow_live_trading
            and bool(self.kite_api_key)
            and bool(self.kite_access_token)
        )
