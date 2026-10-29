"""Multi-timeframe chart candles.

The trading engine always works on 5-minute bars. This service serves the
chart's timeframe selector by resampling a small set of cached base feeds:

  1m / 2m / 3m        <- Groww 1-minute candles (current session)
  5m .. 1h            <- Groww five-session + current-session 5-minute bars
  1d / 1w / 1M / 1y   <- Groww daily candles (one year)

Resampling is done with pandas so 2m, 10m and 1h stay consistent with the
underlying bars. Demo mode builds the same ladder from the simulated feed.
"""

from __future__ import annotations

import random
import threading
import time
import zlib
from datetime import datetime, timedelta

import pandas as pd

from .config import IST, SYMBOLS, Settings
from .datafeed import MarketData, _ts, now_ist
from .indicators import compute_frame
from .models import Candle

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
HIGH_TF = {"1d", "1w", "1M", "1y"}

TIMEFRAMES: dict[str, dict] = {
    "1m": {"label": "1m", "base": "1m", "rule": None, "bars": 750, "period": "5d"},
    "2m": {"label": "2m", "base": "1m", "rule": "2min", "bars": 480, "period": "5d"},
    "3m": {"label": "3m", "base": "1m", "rule": "3min", "bars": 400, "period": "5d"},
    "5m": {"label": "5m", "base": "5m", "rule": None, "bars": 600, "period": "1mo"},
    "10m": {"label": "10m", "base": "5m", "rule": "10min", "bars": 375, "period": "1mo"},
    "15m": {"label": "15m", "base": "5m", "rule": "15min", "bars": 300, "period": "1mo"},
    "30m": {"label": "30m", "base": "5m", "rule": "30min", "bars": 200, "period": "1mo"},
    "1h": {"label": "1h", "base": "5m", "rule": "60min", "bars": 120, "period": "1mo"},
    "1d": {"label": "1d", "base": "1d", "rule": None, "bars": 200, "period": "10y"},
    "1w": {"label": "1w", "base": "1d", "rule": "W", "bars": 200, "period": "10y"},
    "1M": {"label": "1M", "base": "1d", "rule": "M", "bars": 120, "period": "10y"},
    "1y": {"label": "1y", "base": "1d", "rule": None, "bars": 252, "period": "10y"},
}

RAW_TTL = {"1m": 10.0, "5m": 10.0, "1d": 600.0}


def _resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None:
        return df
    if rule in {"W", "M"}:
        # Group Monday-Sunday / calendar month, labelled by the first trading day.
        # Periods are tz-naive by definition; localize first to avoid the warning.
        key = df.index.tz_localize(None).to_period("W-SUN" if rule == "W" else "M")
        out = df.groupby(key).agg(AGG)
        first_ts = df.index.to_series().groupby(key).min().reindex(out.index)
        # DatetimeIndex(Series) keeps the IST tz; .values would drop it.
        out.index = pd.DatetimeIndex(first_ts)
        return out.dropna(subset=["open", "close"]).sort_index()
    out = df.resample(
        rule, label="left", closed="left", origin="start_day", offset="15min"
    ).agg(AGG)
    return out.dropna(subset=["open", "close"])


def _frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "open": float(c.o),
                "high": float(c.h),
                "low": float(c.l),
                "close": float(c.c),
                "volume": float(c.v),
            }
            for c in candles
        ],
        index=pd.DatetimeIndex([pd.Timestamp(c.t) for c in candles]),
    )


def _candles_from_frame(df: pd.DataFrame) -> list[Candle]:
    out: list[Candle] = []
    for ts, row in df.iterrows():
        out.append(
            Candle(
                _ts(ts.to_pydatetime()),
                round(float(row["open"]), 2),
                round(float(row["high"]), 2),
                round(float(row["low"]), 2),
                round(float(row["close"]), 2),
                float(row.get("volume", 0.0)),
            )
        )
    return out


class ChartService:
    def __init__(self, market: MarketData, settings: Settings) -> None:
        self.market = market
        self.settings = settings
        self._lock = threading.RLock()
        self._raw: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._cache: dict[tuple[str, str, bool], tuple[float, dict]] = {}

    # ------------------------------------------------------------------ public
    def get(self, symbol: str, tf: str) -> dict:
        spec = TIMEFRAMES.get(tf)
        if spec is None:
            raise ValueError(f"unknown timeframe: {tf}")
        if symbol not in SYMBOLS:
            raise ValueError(f"unknown symbol: {symbol}")

        with self._lock:
            demo = self.settings.demo
            ttl = 180.0 if tf in HIGH_TF else 10.0
            key = (symbol, tf, demo)
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
            payload = self._build(symbol, tf, spec, demo)
            self._cache[key] = (time.time(), payload)
            return payload

    # ------------------------------------------------------------------ build
    def _build(self, symbol: str, tf: str, spec: dict, demo: bool) -> dict:
        source = "simulated"
        if demo:
            candles = self._demo_candles(symbol, tf, spec)
        else:
            try:
                raw = self._live_raw(symbol, spec["base"], spec["period"])
                candles = _candles_from_frame(_resample(raw, spec["rule"]))
                source = "groww"
            except Exception:
                candles = self._demo_candles(symbol, tf, spec)

        if len(candles) < 10:
            raise RuntimeError("not enough candles for this timeframe")

        frame = compute_frame(candles, reset_vwap=tf not in HIGH_TF)
        if tf in HIGH_TF:
            # VWAP is an intraday anchored measure; daily index bars have no
            # usable volume, so suppress it instead of drawing a 10-year average.
            frame["vwap"] = float("nan")
        frame = frame.iloc[-spec["bars"]:].reset_index(drop=True)
        return self._payload(symbol, tf, spec, frame, source)

    def _live_raw(self, symbol: str, base: str, period: str) -> pd.DataFrame:
        key = (symbol, base)
        now = time.time()
        hit = self._raw.get(key)
        if hit and now - hit[0] < RAW_TTL[base]:
            return hit[1]
        if base == "1m":
            candles = self.market.groww.minute(symbol)
        elif base == "1d":
            candles = self.market.groww.daily(symbol)
        else:
            candles, _ = self.market.groww.get(symbol)
        if len(candles) < 10:
            raise RuntimeError("not enough groww candles")
        df = _frame(candles)
        self._raw[key] = (now, df)
        return df

    # ------------------------------------------------------------------- demo
    def _demo_candles(self, symbol: str, tf: str, spec: dict) -> list[Candle]:
        five, _ = self.market.demo.get(symbol)
        if tf in {"1m", "2m", "3m"}:
            # 225 five-minute bars expand to three sessions of 1-minute data.
            minute = self._expand_minutes(symbol, five[-225:])
            if tf == "1m":
                return minute
            return _candles_from_frame(_resample(_frame(minute), spec["rule"]))
        if tf == "5m":
            return five
        if tf in {"10m", "15m", "30m", "1h"}:
            return _candles_from_frame(_resample(_frame(five), spec["rule"]))
        daily = self._demo_daily(symbol, five[-1].c)
        if spec["rule"]:
            return _candles_from_frame(_resample(_frame(daily), spec["rule"]))
        return daily

    def _expand_minutes(self, symbol: str, five: list[Candle]) -> list[Candle]:
        """Split each 5-minute bar into five 1-minute bars anchored to its OHLC.

        The forming 5-minute bar only expands up to the current minute; showing
        14:12-14:14 while the clock says 14:11 would invent future candles and
        stop the live tape from reaching the last bar.
        """
        out: list[Candle] = []
        cutoff = now_ist().replace(second=0, microsecond=0)
        for c in five:
            rng = random.Random(zlib.crc32(f"{symbol}:{c.t}".encode()) & 0xFFFFFFFF)
            span = max(c.h - c.l, 0.02)
            pts = [c.o]
            for k in range(1, 5):
                base = c.o + (c.c - c.o) * k / 5.0
                pts.append(min(c.h, max(c.l, base + rng.gauss(0, span * 0.18))))
            pts.append(c.c)
            ts = datetime.fromisoformat(c.t)
            for k in range(5):
                minute_ts = ts + timedelta(minutes=k)
                if minute_ts > cutoff:
                    break
                po, pc = pts[k], pts[k + 1]
                hi = min(c.h, max(po, pc) + abs(rng.gauss(0, span * 0.08)))
                lo = max(c.l, min(po, pc) - abs(rng.gauss(0, span * 0.08)))
                out.append(
                    Candle(
                        _ts(minute_ts),
                        round(po, 2),
                        round(max(hi, po, pc), 2),
                        round(min(lo, po, pc), 2),
                        round(pc, 2),
                        max(1.0, c.v / 5.0),
                    )
                )
        return out

    def _demo_daily(self, symbol: str, spot: float) -> list[Candle]:
        """~10 years of simulated daily bars, scaled so the last close is spot."""
        cfg = SYMBOLS[symbol]
        rng = random.Random(zlib.crc32(f"{symbol}:daily".encode()) & 0xFFFFFFFF)
        days: list = []
        day = now_ist().date()
        while len(days) < 2600:
            if day.weekday() < 5:
                days.append(day)
            day -= timedelta(days=1)
        days.reverse()

        price = cfg.demo_base
        rows: list[list[float]] = []
        for _ in days:
            ret = rng.gauss(0.0002, cfg.demo_vol * 8.0)
            close = max(1.0, price * (1.0 + ret))
            high = max(price, close) * (1.0 + abs(rng.gauss(0, cfg.demo_vol * 3.0)))
            low = min(price, close) * (1.0 - abs(rng.gauss(0, cfg.demo_vol * 3.0)))
            rows.append([price, high, low, close])
            price = close

        scale = spot / rows[-1][3] if rows[-1][3] else 1.0
        out: list[Candle] = []
        for day, (o, h, l, c) in zip(days, rows):
            ts = datetime.combine(day, datetime.min.time(), tzinfo=IST).replace(hour=9, minute=15)
            out.append(
                Candle(
                    _ts(ts),
                    round(o * scale, 2),
                    round(h * scale, 2),
                    round(l * scale, 2),
                    round(c * scale, 2),
                    0.0,
                )
            )
        return out

    # ----------------------------------------------------------------- payload
    def _payload(
        self, symbol: str, tf: str, spec: dict, frame: pd.DataFrame, source: str
    ) -> dict:
        ts = frame["ts"]
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(IST)

        candles = []
        for i in range(len(frame)):
            candles.append(
                {
                    "t": ts.iloc[i].isoformat(timespec="minutes"),
                    "o": round(float(frame["open"].iloc[i]), 2),
                    "h": round(float(frame["high"].iloc[i]), 2),
                    "l": round(float(frame["low"].iloc[i]), 2),
                    "c": round(float(frame["close"].iloc[i]), 2),
                    "v": round(float(frame["volume"].iloc[i]), 2),
                }
            )

        def series(column: str) -> list:
            return [None if pd.isna(v) else round(float(v), 2) for v in frame[column]]

        last = frame.iloc[-1]
        close = float(last["close"])
        atr_value = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        vwap = None if pd.isna(last["vwap"]) else round(float(last["vwap"]), 2)
        indicators = {
            "close": round(close, 2),
            "rsi": round(float(last["rsi"]), 1),
            "atr": round(atr_value, 1),
            "atr_pct": round(atr_value / close * 100.0, 2) if close else 0.0,
            "adx": round(float(last["adx"]), 1),
            "ema20": round(float(last["ema20"]), 2),
            "ema50": round(float(last["ema50"]), 2),
            "vwap": vwap,
            "supertrend": round(float(last["supertrend"]), 2),
            "trend": int(last["trend"]),
        }
        return {
            "symbol": symbol,
            "tf": tf,
            "label": spec["label"],
            "source": source,
            "bars": len(candles),
            "candles": candles,
            "series": {
                "ema20": series("ema20"),
                "ema50": series("ema50"),
                "vwap": series("vwap"),
            },
            "indicators": indicators,
            "as_of": now_ist().isoformat(timespec="seconds"),
        }
