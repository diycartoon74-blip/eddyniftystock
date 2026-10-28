"""Market data feeds: Yahoo index candles + NSE/BSE chains, with demo fallback."""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from .config import IST, MARKET_CLOSE, MARKET_OPEN, SYMBOLS
from .models import Candle, OptionChain
from .optionchain import BSEClient, NSEClient, synthetic_chain


def now_ist() -> datetime:
    return datetime.now(IST)


def market_is_open(now: datetime | None = None) -> bool:
    now = now or now_ist()
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def _ts(dt: datetime) -> str:
    return dt.astimezone(IST).isoformat(timespec="minutes")


def _bucket_start(now: datetime) -> datetime | None:
    """Current forming 5-minute bar inside the NSE session, else None."""
    if now.weekday() >= 5:
        return None
    start = datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST)
    if now < start or now.time() > MARKET_CLOSE:
        return None
    minutes = int((now - start).total_seconds() // 300) * 5
    return start + timedelta(minutes=minutes)


class YahooFeed:
    """Intraday index candles from Yahoo Finance (NSE symbols, ~delayed)."""

    def get(self, symbol: str) -> tuple[list[Candle], str]:
        cfg = SYMBOLS[symbol]
        df = yf.download(
            cfg.yahoo,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            raise RuntimeError("yahoo returned no candles")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"unexpected yahoo columns: {list(df.columns)}")

        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            try:
                index = index.tz_convert(IST)
            except Exception:
                pass
        else:
            index = index.tz_localize(IST)

        candles: list[Candle] = []
        for i in range(len(df)):
            row = df.iloc[i]
            if pd.isna(row["close"]):
                continue
            volume = row.get("volume", 0)
            candles.append(
                Candle(
                    t=_ts(index[i].to_pydatetime()),
                    o=round(float(row["open"]), 2),
                    h=round(float(row["high"]), 2),
                    l=round(float(row["low"]), 2),
                    c=round(float(row["close"]), 2),
                    v=0.0 if pd.isna(volume) else float(volume),
                )
            )
        if len(candles) < 30:
            raise RuntimeError("not enough yahoo candles")
        return candles[-750:], "yahoo"


class DemoFeed:
    """Random-walk 5-minute candles aligned to NSE session hours.

    Bars are only created inside 09:15-15:30 on weekdays, so the demo chart
    looks like a real trading day. Outside the session the last bar keeps
    ticking so the dashboard stays alive.
    """

    SESSION_BARS = 75  # 09:15 .. 15:25
    HISTORY_BARS = 750  # ~10 sessions, enough to pan back several days

    def __init__(self) -> None:
        self._candles: dict[str, list[Candle]] = {}
        self._rng: dict[str, random.Random] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _session_slots(now: datetime, count: int) -> list[datetime]:
        slots: list[datetime] = []
        day = now.date()
        while len(slots) < count:
            if day.weekday() < 5:
                start = datetime.combine(day, MARKET_OPEN, tzinfo=IST)
                for i in range(DemoFeed.SESSION_BARS):
                    slot = start + timedelta(minutes=5 * i)
                    if slot <= now:
                        slots.append(slot)
            day -= timedelta(days=1)
        slots.sort()
        return slots[-count:]

    def _genesis(self, symbol: str) -> tuple[list[Candle], random.Random]:
        cfg = SYMBOLS[symbol]
        rng = random.Random(hash(symbol) & 0xFFFF)
        slots = self._session_slots(
            now_ist().replace(second=0, microsecond=0), self.HISTORY_BARS
        )
        price = cfg.demo_base
        drift = 0.0
        session = None
        candles: list[Candle] = []
        for i, ts in enumerate(slots):
            if ts.date() != session:
                session = ts.date()
                drift = cfg.demo_vol * 0.35 * rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            elif i % 26 == 0:
                drift = cfg.demo_vol * 0.35 * rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            shock = rng.gauss(0.0, cfg.demo_vol)
            change = drift + shock
            open_price = price
            close = max(1.0, open_price * (1.0 + change))
            high = max(open_price, close) * (1.0 + abs(rng.gauss(0, cfg.demo_vol * 0.45)))
            low = min(open_price, close) * (1.0 - abs(rng.gauss(0, cfg.demo_vol * 0.45)))
            volume = rng.randint(40_000, 320_000)
            candles.append(
                Candle(_ts(ts), round(open_price, 2), round(high, 2), round(low, 2), round(close, 2), volume)
            )
            price = close
        return candles, rng

    def _ensure(self, symbol: str) -> tuple[list[Candle], random.Random]:
        if symbol not in self._candles:
            candles, rng = self._genesis(symbol)
            self._candles[symbol] = candles
            self._rng[symbol] = rng
        return self._candles[symbol], self._rng[symbol]

    def get(self, symbol: str) -> tuple[list[Candle], float]:
        cfg = SYMBOLS[symbol]
        with self._lock:
            candles, rng = self._ensure(symbol)
            now = now_ist()
            last_ts = datetime.fromisoformat(candles[-1].t).astimezone(IST)

            pending: list[datetime] = []
            if now.weekday() < 5:
                open_dt = datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST)
                for i in range(self.SESSION_BARS):
                    slot = open_dt + timedelta(minutes=5 * i)
                    if slot > last_ts and slot <= now:
                        pending.append(slot)

            if pending:
                # Backfill missed 5-minute bars (capped so a long pause stays cheap).
                for slot in pending[:80]:
                    last = candles[-1]
                    drift = 0.0
                    if rng.random() < 0.12:
                        drift = cfg.demo_vol * rng.choice([-0.6, 0.6, 0.0])
                    close = max(1.0, last.c * (1.0 + drift + rng.gauss(0, cfg.demo_vol)))
                    candles.append(
                        Candle(
                            _ts(slot),
                            last.c,
                            round(max(last.c, close) * (1 + abs(rng.gauss(0, 0.0002))), 2),
                            round(min(last.c, close) * (1 - abs(rng.gauss(0, 0.0002))), 2),
                            round(close, 2),
                            rng.randint(40_000, 320_000),
                        )
                    )
            else:
                last = candles[-1]
                tick = last.c * (1.0 + rng.gauss(0, cfg.demo_vol * 0.18))
                candles[-1] = Candle(
                    last.t,
                    last.o,
                    round(max(last.h, tick), 2),
                    round(min(last.l, tick), 2),
                    round(tick, 2),
                    last.v + rng.randint(100, 900),
                )

            if len(candles) > self.HISTORY_BARS:
                del candles[: -self.HISTORY_BARS]
            # Snapshot under the lock: the live tick loop keeps replacing the
            # forming bar while the engine builds an analysis frame.
            return list(candles), "simulated"

    def tick(self, symbol: str) -> dict:
        """Advance the demo tape by one second and return the forming bar.

        Inside the session a price step updates the current 5-minute candle
        (or opens a fresh one at the boundary). Outside the session the tape is
        frozen, which matches how the chart behaves in live mode.
        """
        cfg = SYMBOLS[symbol]
        with self._lock:
            candles, rng = self._ensure(symbol)
            now = now_ist()
            last = candles[-1]
            bucket = _bucket_start(now)
            if bucket is None:
                return {"price": last.c, "t": now.isoformat(timespec="seconds"), "candle": last.to_dict()}

            last_ts = datetime.fromisoformat(last.t).astimezone(IST)
            if last_ts < bucket:
                price = max(1.0, last.c * (1.0 + rng.gauss(0, cfg.demo_vol * 0.05)))
                candle = Candle(
                    _ts(bucket),
                    round(last.c, 2),
                    round(max(last.c, price), 2),
                    round(min(last.c, price), 2),
                    round(price, 2),
                    rng.randint(100, 700),
                )
                candles.append(candle)
                if len(candles) > self.HISTORY_BARS:
                    del candles[: -self.HISTORY_BARS]
            else:
                # One-second sigma: a 5-minute bar's random walk spread over
                # its ~300 ticks, so intraday volatility stays realistic.
                price = max(1.0, last.c * (1.0 + rng.gauss(0, cfg.demo_vol / 17.32)))
                candle = Candle(
                    last.t,
                    last.o,
                    round(max(last.h, price), 2),
                    round(min(last.l, price), 2),
                    round(price, 2),
                    last.v + rng.randint(20, 160),
                )
                candles[-1] = candle
            return {"price": candle.c, "t": now.isoformat(timespec="seconds"), "candle": candle.to_dict()}


class YahooQuoteFeed:
    """Lightweight last-price poller for the live tape.

    The chart candles still come from the 5-minute downloader; this endpoint
    only supplies the current print so the forming candle can breathe between
    those slower refreshes. NSE prints on Yahoo are delayed, which the UI
    labels separately.
    """

    URL = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        "{target}?interval=1m&range=1d&includePrePost=false"
    )
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
    }
    TTL = 2.0

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._cache: dict[str, tuple[float, dict]] = {}
        self._blocked_until = 0.0
        self._lock = threading.RLock()

    def get(self, symbol: str) -> dict | None:
        now = time.time()
        with self._lock:
            hit = self._cache.get(symbol)
            if hit and now - hit[0] < self.TTL:
                return hit[1]
            if now < self._blocked_until:
                return None

        cfg = SYMBOLS[symbol]
        try:
            res = self._session.get(self.URL.format(target=cfg.yahoo), timeout=4)
            res.raise_for_status()
            results = (res.json().get("chart") or {}).get("result") or []
            meta = (results[0] or {}).get("meta") or {} if results else {}
            price = meta.get("regularMarketPrice")
            if price is None and results:
                quote = ((results[0].get("indicators") or {}).get("quote") or [{}])[0]
                closes = [c for c in (quote.get("close") or []) if c is not None]
                price = closes[-1] if closes else None
            if price is None:
                raise RuntimeError("no quote price")
            stamp = meta.get("regularMarketTime")
            when = datetime.fromtimestamp(float(stamp), IST) if stamp else now_ist()
            out = {
                "price": round(float(price), 2),
                "t": when.isoformat(timespec="seconds"),
            }
            with self._lock:
                self._cache[symbol] = (now, out)
                self._blocked_until = 0.0
            return out
        except Exception:
            with self._lock:
                self._blocked_until = time.time() + 30.0
            return None

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()
            self._blocked_until = 0.0


class MarketData:
    def __init__(self) -> None:
        self.yahoo = YahooFeed()
        self.demo = DemoFeed()
        self.quotes = YahooQuoteFeed()
        self.nse = NSEClient()
        self.bse = BSEClient()
        self._candle_cache: dict[str, tuple[float, list[Candle], str]] = {}
        self._chain_cache: dict[str, tuple[float, OptionChain]] = {}
        self._prev_close_cache: dict[str, float] = {}
        self._feed_sources: dict[str, str] = {}
        self._yahoo_blocked_until = 0.0
        self.last_notes: dict[str, str] = {}

    def _prev_close(self, candles: list[Candle]) -> float:
        if not candles:
            return 0.0
        last_day = candles[-1].t[:10]
        for candle in reversed(candles[:-1]):
            if candle.t[:10] != last_day:
                return candle.c
        return candles[0].o

    def reset(self) -> None:
        """Drop cached candles/chains so a data-mode switch starts clean."""
        self._candle_cache.clear()
        self._chain_cache.clear()
        self._prev_close_cache.clear()
        self._feed_sources.clear()
        self.quotes.reset()
        self._yahoo_blocked_until = 0.0
        self.last_notes.clear()

    def get(self, symbol: str, demo: bool) -> dict:
        ttl = 12
        cached = self._candle_cache.get(symbol)
        if demo:
            candles, feed_source = self.demo.get(symbol)
        elif cached and time.time() - cached[0] < ttl:
            candles, feed_source = cached[1], cached[2]
        else:
            candles, feed_source = [], "simulated"
            if time.time() >= self._yahoo_blocked_until:
                try:
                    candles, feed_source = self.yahoo.get(symbol)
                    self._yahoo_blocked_until = 0.0
                except Exception as exc:  # network / parsing / rate limits
                    self._yahoo_blocked_until = time.time() + 60
                    self.last_notes[symbol] = f"Yahoo unavailable ({type(exc).__name__}); using simulated feed"
            if not candles:
                candles, feed_source = self.demo.get(symbol)
                self.last_notes.setdefault(
                    symbol, "Yahoo unavailable; using simulated feed so the dashboard keeps running"
                )
            self._candle_cache[symbol] = (time.time(), candles, feed_source)

        prev_close = self._prev_close(candles)
        self._prev_close_cache[symbol] = prev_close
        self._feed_sources[symbol] = feed_source
        spot = candles[-1].c
        cfg = SYMBOLS[symbol]
        if demo or feed_source == "simulated" or cfg.chain_feed not in ("nse", "bse"):
            # Rebuild every pass so the ATM strike always tracks the live spot.
            chain = synthetic_chain(symbol, spot)
        else:
            chain_cached = self._chain_cache.get(symbol)
            ttl = 60.0 if cfg.chain_feed == "bse" else 30.0
            if chain_cached and time.time() - chain_cached[0] < ttl:
                chain = chain_cached[1]
            else:
                client = self.bse if cfg.chain_feed == "bse" else self.nse
                label = "BSE" if cfg.chain_feed == "bse" else "NSE"
                try:
                    chain = client.fetch(symbol, spot)
                    self.last_notes.pop(symbol, None)
                except Exception as exc:
                    if chain_cached and time.time() - chain_cached[0] < 600:
                        # Keep the last real chain through a transient public
                        # endpoint gap instead of flashing synthetic data.
                        chain = chain_cached[1]
                    else:
                        self.last_notes[symbol] = (
                            f"{label} chain unavailable ({type(exc).__name__}); synthetic chain in use"
                        )
                        chain = synthetic_chain(symbol, spot)
                self._chain_cache[symbol] = (time.time(), chain)

        return {
            "symbol": symbol,
            "candles": candles,
            "feed_source": feed_source,
            "prev_close": prev_close,
            "chain": chain,
            "market_open": True if demo else market_is_open(),
            "note": self.last_notes.get(symbol, ""),
        }

    # ------------------------------------------------------------------ ticks
    def ticks(self, demo: bool) -> dict[str, dict]:
        """Latest prints for the chart's forming candle.

        Demo mode moves the shared simulated tape so the engine and the chart
        agree. Live mode only reads a quote; Yahoo candles stay untouched until
        the next analysis refresh.
        """
        out: dict[str, dict] = {}
        if demo:
            for symbol in SYMBOLS:
                if symbol not in self._prev_close_cache:
                    candles, _ = self.demo.get(symbol)
                    self._prev_close_cache[symbol] = self._prev_close(candles)
                tick = self.demo.tick(symbol)
                prev = self._prev_close_cache.get(symbol, 0.0)
                change = tick["price"] - prev if prev else 0.0
                out[symbol] = {
                    **tick,
                    "change": round(change, 2),
                    "change_pct": round(change / prev * 100.0, 2) if prev else 0.0,
                }
            return out

        for symbol in SYMBOLS:
            # Never mix a real quote into simulated fallback candles: the price
            # levels are unrelated and the chart would jump between them.
            if self._feed_sources.get(symbol) != "yahoo":
                continue
            quote = self.quotes.get(symbol)
            if not quote:
                continue
            prev = self._prev_close_cache.get(symbol, 0.0)
            change = quote["price"] - prev if prev else 0.0
            out[symbol] = {
                **quote,
                "change": round(change, 2),
                "change_pct": round(change / prev * 100.0, 2) if prev else 0.0,
            }
        return out
