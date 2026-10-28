"""Plain data models shared by the data, strategy and execution layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candle:
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v}


@dataclass
class OptionRow:
    strike: float
    ce_ltp: float = 0.0
    ce_oi: int = 0
    ce_oi_change: int = 0
    ce_iv: float = 0.0
    pe_ltp: float = 0.0
    pe_oi: int = 0
    pe_oi_change: int = 0
    pe_iv: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "ce_ltp": self.ce_ltp,
            "ce_oi": self.ce_oi,
            "ce_oi_change": self.ce_oi_change,
            "ce_iv": self.ce_iv,
            "pe_ltp": self.pe_ltp,
            "pe_oi": self.pe_oi,
            "pe_oi_change": self.pe_oi_change,
            "pe_iv": self.pe_iv,
        }


@dataclass
class OptionChain:
    symbol: str
    spot: float
    source: str
    expiry: str
    rows: list[OptionRow] = field(default_factory=list)
    updated_at: str = ""
    pcr: float = 1.0
    max_pain: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    atm_iv: float = 0.0

    def row_for(self, strike: float) -> OptionRow | None:
        for row in self.rows:
            if abs(row.strike - strike) < 0.01:
                return row
        return None

    def quote(self, strike: float, side: str) -> float:
        row = self.row_for(strike)
        if row is None:
            return 0.0
        return row.ce_ltp if side == "CE" else row.pe_ltp

    def atm_strike(self) -> float:
        if not self.rows:
            return 0.0
        return min(self.rows, key=lambda r: abs(r.strike - self.spot)).strike

    def to_dict(self, limit: int = 25) -> dict[str, Any]:
        rows = self.rows
        if len(rows) > limit:
            atm_index = min(range(len(rows)), key=lambda i: abs(rows[i].strike - self.spot))
            half = limit // 2
            start = max(0, min(len(rows) - limit, atm_index - half))
            rows = rows[start : start + limit]
        return {
            "symbol": self.symbol,
            "spot": self.spot,
            "source": self.source,
            "expiry": self.expiry,
            "updated_at": self.updated_at,
            "pcr": round(self.pcr, 3),
            "max_pain": self.max_pain,
            "support": self.support,
            "resistance": self.resistance,
            "atm_iv": round(self.atm_iv, 2),
            "atm_strike": self.atm_strike(),
            "rows": [r.to_dict() for r in rows],
        }


@dataclass
class SignalDraft:
    """A setup produced by the strategy before risk checks / execution."""

    symbol: str
    side: str
    strike: float
    expiry: str
    spot: float
    entry: float
    stop: float
    target: float
    confidence: int
    score: int
    bias: str
    reasons: list[dict[str, str]]
    chain_source: str

    @property
    def rr(self) -> float:
        risk = self.entry - self.stop
        if risk <= 0:
            return 0.0
        return round((self.target - self.entry) / risk, 2)
