"""Technical indicators implemented on pandas, no external TA dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def session_vwap(df: pd.DataFrame, reset: bool = True) -> pd.Series:
    """VWAP that resets every trading day, or anchors to the window start.

    Index feeds carry zero volume, so when no bar has measurable volume we fall
    back to an equal-weighted typical price (TWAP) instead of a flat first-bar
    line.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].astype(float).replace(0.0, np.nan)
    if not volume.notna().any():
        volume = pd.Series(1.0, index=df.index)
    pv = (typical * volume).fillna(0.0)
    if reset:
        day = pd.to_datetime(df["ts"]).dt.date
        cum_pv = pv.groupby(day).cumsum()
        cum_vol = volume.fillna(0.0).groupby(day).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = volume.fillna(0.0).cumsum()
    vwap = cum_pv / cum_vol.replace(0.0, np.nan)
    return vwap.ffill().fillna(typical)


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Returns (trend direction 1/-1, supertrend line)."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    _atr = atr(df, period).to_numpy(dtype=float)

    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * _atr
    lower = hl2 - multiplier * _atr

    final_upper = np.full(len(df), np.nan)
    final_lower = np.full(len(df), np.nan)
    trend = np.ones(len(df), dtype=int)

    start = None
    for i in range(len(df)):
        if not np.isnan(_atr[i]):
            start = i
            break
    if start is None:
        start = min(1, len(df) - 1)

    final_upper[start] = upper[start]
    final_lower[start] = lower[start]
    trend[start] = 1

    for i in range(start + 1, len(df)):
        final_upper[i] = (
            upper[i]
            if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i]
            if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )
        if trend[i - 1] == 1:
            trend[i] = 1 if close[i] >= final_lower[i] else -1
        else:
            trend[i] = -1 if close[i] <= final_upper[i] else 1

    line = np.where(trend == 1, final_lower, final_upper)
    return pd.Series(trend, index=df.index), pd.Series(line, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def compute_frame(candles: list, reset_vwap: bool = True) -> pd.DataFrame:
    """Build an indicator dataframe from Candle objects."""
    if not candles:
        raise ValueError("no candles supplied")
    df = pd.DataFrame(
        [
            {
                "ts": c.t,
                "open": float(c.o),
                "high": float(c.h),
                "low": float(c.l),
                "close": float(c.c),
                "volume": float(c.v),
            }
            for c in candles
        ]
    )
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("candles could not be parsed")
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["vwap"] = session_vwap(df, reset=reset_vwap)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["trend"], df["supertrend"] = supertrend(df, 10, 3.0)
    df["adx"] = adx(df, 14)
    return df
