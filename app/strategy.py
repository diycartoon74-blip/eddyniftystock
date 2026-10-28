"""Multi-factor intraday scoring for index options.

Factors: session VWAP, EMA20/EMA50 structure, Supertrend, RSI, short-term
momentum, option-chain PCR, max pain, and OI support/resistance proximity.
Only decisive scores become tradable signals; everything is explained in the
`reasons` list so the dashboard never shows a black box.
"""

from __future__ import annotations

import math

from .config import SYMBOLS
from .indicators import compute_frame
from .models import Candle, OptionChain, SignalDraft


def _f(value, digits: int = 2) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return round(number, digits)


def _series(frame, column: str, size: int = 150) -> list[float | None]:
    values = frame[column].tail(size).tolist()
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 2) for v in values]


def analyze(
    symbol: str,
    candles: list[Candle],
    chain: OptionChain,
    prev_close: float,
    threshold: int,
    feed_source: str = "yahoo",
    note: str = "",
) -> dict:
    cfg = SYMBOLS[symbol]
    frame = compute_frame(candles)
    last = frame.iloc[-1]
    close = float(last["close"])
    prev_close = prev_close or float(frame["open"].iloc[0])
    change = close - prev_close
    change_pct = (change / prev_close * 100.0) if prev_close else 0.0

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    vwap = float(last["vwap"])
    rsi_value = float(last["rsi"])
    atr_value = float(last["atr"])
    trend = int(last["trend"])
    atr_pct = (atr_value / close * 100.0) if close else 0.0

    score = 50.0
    reasons: list[dict[str, str]] = []

    def factor(delta: float, text: str) -> None:
        nonlocal score
        score += delta
        impact = "bull" if delta > 0 else "bear" if delta < 0 else "neutral"
        reasons.append({"text": text, "impact": impact})

    if close > vwap:
        factor(10, f"Spot {close:,.1f} is above session VWAP {vwap:,.1f}")
    else:
        factor(-10, f"Spot {close:,.1f} is below session VWAP {vwap:,.1f}")

    if ema20 > ema50:
        factor(10, "EMA20 above EMA50 - bullish structure")
    else:
        factor(-10, "EMA20 below EMA50 - bearish structure")

    if trend > 0:
        factor(8, "Supertrend is bullish (price above trailing stop)")
    else:
        factor(-8, "Supertrend is bearish (price below trailing stop)")

    if 55 <= rsi_value <= 72:
        factor(7, f"RSI {rsi_value:.0f} in the bullish momentum zone")
    elif 30 <= rsi_value <= 45:
        factor(-7, f"RSI {rsi_value:.0f} in the bearish momentum zone")
    elif rsi_value > 76:
        factor(-4, f"RSI {rsi_value:.0f} overbought - chasing risk")
    elif rsi_value < 24:
        factor(4, f"RSI {rsi_value:.0f} oversold - bounce risk")

    lookback = min(15, len(frame) - 1)
    if lookback > 0:
        past = float(frame["close"].iloc[-1 - lookback])
        momentum = (close / past - 1.0) * 100.0 if past else 0.0
        if momentum > 0.15:
            factor(6, f"Momentum +{momentum:.2f}% over the last {lookback} bars")
        elif momentum < -0.15:
            factor(-6, f"Momentum {momentum:.2f}% over the last {lookback} bars")

    if chain.pcr >= 1.15:
        factor(6, f"PCR {chain.pcr:.2f} - put writers dominant (bullish)")
    elif chain.pcr <= 0.85:
        factor(-6, f"PCR {chain.pcr:.2f} - call writers dominant (bearish)")
    else:
        reasons.append({"text": f"PCR {chain.pcr:.2f} balanced", "impact": "neutral"})

    if chain.max_pain:
        if chain.max_pain > close:
            factor(3, f"Max pain {chain.max_pain:,.0f} sits above spot")
        elif chain.max_pain < close:
            factor(-3, f"Max pain {chain.max_pain:,.0f} sits below spot")

    if chain.resistance:
        distance = (chain.resistance - close) / close * 100.0
        if 0 <= distance <= 0.35:
            factor(-4, f"Call-OI wall at {chain.resistance:,.0f} just overhead")
    if chain.support:
        distance = (close - chain.support) / close * 100.0
        if 0 <= distance <= 0.35:
            factor(4, f"Put-OI support at {chain.support:,.0f} just below")

    score = max(0.0, min(100.0, score))
    confidence = int(min(97, round(abs(score - 50.0) * 2.3)))
    bias = "BULLISH" if score >= 60 else "BEARISH" if score <= 40 else "NEUTRAL"

    indicators = {
        "ema20": _f(ema20),
        "ema50": _f(ema50),
        "vwap": _f(vwap),
        "rsi": _f(rsi_value, 1),
        "atr": _f(atr_value),
        "atr_pct": _f(atr_pct, 2),
        "supertrend": _f(last["supertrend"]),
        "trend": trend,
        "adx": _f(last["adx"], 1),
        "pcr": _f(chain.pcr, 2),
        "max_pain": _f(chain.max_pain, 0),
        "support": _f(chain.support, 0),
        "resistance": _f(chain.resistance, 0),
        "atm_iv": _f(chain.atm_iv, 1),
        "expiry": chain.expiry,
    }

    draft: SignalDraft | None = None
    if bias != "NEUTRAL" and confidence >= threshold and chain.rows:
        side = "CE" if bias == "BULLISH" else "PE"
        strike = chain.atm_strike()
        entry = chain.quote(strike, side)
        if entry > 1.0:
            stop = round(entry * 0.80, 2)
            target = round(entry + 1.8 * (entry - stop), 2)
            draft = SignalDraft(
                symbol=symbol,
                side=side,
                strike=strike,
                expiry=chain.expiry,
                spot=close,
                entry=round(entry, 2),
                stop=stop,
                target=target,
                confidence=confidence,
                score=int(round(score)),
                bias=bias,
                reasons=reasons,
                chain_source=chain.source,
            )

    size = 150
    series_source = frame.tail(size)
    candles_out = [
        {
            "t": row["ts"].isoformat(),
            "o": _f(row["open"]),
            "h": _f(row["high"]),
            "l": _f(row["low"]),
            "c": _f(row["close"]),
            "v": int(row["volume"]),
        }
        for _, row in frame.tail(size).iterrows()
    ]

    return {
        "symbol": symbol,
        "label": cfg.label,
        "spot": _f(close),
        "change": _f(change),
        "change_pct": _f(change_pct, 2),
        "bias": bias,
        "score": int(round(score)),
        "confidence": confidence,
        "threshold": threshold,
        "reasons": reasons,
        "indicators": indicators,
        "series": {
            "ema20": _series(series_source, "ema20", size),
            "ema50": _series(series_source, "ema50", size),
            "vwap": _series(series_source, "vwap", size),
        },
        "candles": candles_out,
        "chain": chain.to_dict(25),
        "chain_source": chain.source,
        "feed_source": feed_source,
        "note": note,
        "lot_size": cfg.lot_size,
        "draft": draft,
    }
