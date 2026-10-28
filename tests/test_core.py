"""Core tests: indicators, chain math, sizing, signals, broker and engine."""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta

import pandas as pd
import pytest

from app.broker import BrokerError, PaperBroker
from app.chartdata import ChartService, TIMEFRAMES
from app.config import IST, Settings
from app.datafeed import DemoFeed, MarketData, _ts, now_ist
from app.engine import TradingEngine, position_size
from app.indicators import compute_frame, rsi, supertrend
from app.models import Candle
from app.optionchain import synthetic_chain
from app.storage import Store
from app.strategy import analyze
from app.wallet import Wallet, WalletError


def _candles(direction: float = 1.0, count: int = 220, base: float = 26000.0) -> list[Candle]:
    start = datetime(2026, 10, 28, 9, 15, tzinfo=IST)
    price = base
    out = []
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        open_p = price
        close = open_p * (1.0 + direction * 0.00045)
        high = max(open_p, close) * 1.00035
        low = min(open_p, close) * 0.99965
        out.append(Candle(_ts(ts), round(open_p, 2), round(high, 2), round(low, 2), round(close, 2), 120000))
        price = close
    return out


def test_rsi_is_bounded():
    candles = _candles(1.0, 120)
    frame = compute_frame(candles)
    assert frame["rsi"].between(0, 100).all()
    assert frame["rsi"].iloc[-1] > 55


def test_supertrend_tracks_uptrend():
    candles = _candles(1.0, 160)
    frame = compute_frame(candles)
    assert int(frame["trend"].iloc[-1]) == 1
    assert frame["supertrend"].iloc[-1] < frame["close"].iloc[-1]


def test_synthetic_chain_analytics():
    chain = synthetic_chain("NIFTY", 26000.0)
    assert len(chain.rows) == 25
    assert chain.pcr > 0
    assert min(r.strike for r in chain.rows) < chain.max_pain < max(r.strike for r in chain.rows)
    assert chain.support <= 26000.0
    assert chain.resistance >= 26000.0
    assert chain.atm_iv > 0
    assert chain.quote(chain.atm_strike(), "CE") > 0


def test_position_size_respects_risk_budget():
    lots, qty = position_size(300.0, 240.0, 75, 500000.0, 1.0, 35.0)
    assert lots == 1 and qty == 75
    # One lot risk = 60 * 75 = 4,500 which fits the 5,000 budget.
    assert 60 * 75 <= 5000


def test_position_size_blocks_oversized_lot():
    lots, qty = position_size(900.0, 700.0, 75, 100000.0, 1.0, 35.0)
    assert lots == 0 and qty == 0


def test_strategy_generates_bullish_ce_signal():
    candles = _candles(1.0, 220)
    spot = candles[-1].c
    chain = synthetic_chain("NIFTY", spot)
    result = analyze("NIFTY", candles, chain, candles[0].o, threshold=50)
    assert result["bias"] == "BULLISH"
    assert result["confidence"] >= 50
    draft = result["draft"]
    assert draft is not None
    assert draft.side == "CE"
    assert draft.stop < draft.entry < draft.target
    assert result["series"]["ema20"][-1] is not None
    assert len(result["candles"]) == 150


def test_paper_broker_slippage():
    broker = PaperBroker()
    buy = broker.buy("NIFTY", "CE", 26000, "03-Nov-2026", 75, 200.0)
    sell = broker.sell("NIFTY", "CE", 26000, "03-Nov-2026", 75, 200.0)
    assert buy.price == pytest.approx(200.7)
    assert sell.price == pytest.approx(199.3)
    assert buy.order_id.startswith("PAPER-")


def test_paper_broker_order_types():
    broker = PaperBroker()
    limit = broker.buy("NIFTY", "CE", 26000, "03-Nov-2026", 75, 188.0, order_type="limit", limit_price=190.0)
    assert limit.price == pytest.approx(188.0)  # no adverse slippage inside the limit
    with pytest.raises(BrokerError):
        broker.buy("NIFTY", "CE", 26000, "03-Nov-2026", 75, 200.0, order_type="limit", limit_price=190.0)
    stop = broker.buy("NIFTY", "CE", 26000, "03-Nov-2026", 75, 200.0, order_type="stop")
    assert stop.price == pytest.approx(200.7)  # a triggered stop pays the market


def test_engine_demo_cycle(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "test.db")
    engine = TradingEngine(settings, store=store)
    engine.refresh()
    snapshot = engine.snapshot()
    assert snapshot["mode"] == "demo"
    assert set(snapshot["symbols"]) == {"NIFTY", "BANKNIFTY", "SENSEX"}
    for analysis in snapshot["symbols"].values():
        assert analysis["spot"] > 0
        assert len(analysis["candles"]) > 50
        assert analysis["chain"]["rows"]
        assert -15 < analysis["change_pct"] < 15
    assert snapshot["execution"]["label"] == "PAPER"
    assert snapshot["stats"]["total"] == 0.0
    store.close()


def test_demo_feed_session_aligned():
    feed = DemoFeed()
    candles, source = feed.get("NIFTY")
    assert source == "simulated"
    assert len(candles) == DemoFeed.HISTORY_BARS
    assert [c.t for c in candles] == sorted(c.t for c in candles)
    for candle in candles:
        ts = datetime.fromisoformat(candle.t)
        assert ts.weekday() < 5
        assert time(9, 15) <= ts.time() <= time(15, 30)


def test_demo_ticks_move_forming_bar_and_roll_boundary(monkeypatch):
    import app.datafeed as datafeed

    clock = {"now": datetime(2026, 10, 28, 10, 2, 10, tzinfo=IST)}
    monkeypatch.setattr(datafeed, "now_ist", lambda: clock["now"])
    feed = DemoFeed()

    first = feed.tick("NIFTY")
    assert first["candle"]["t"] == _ts(datetime(2026, 10, 28, 10, 0, tzinfo=IST))
    assert first["price"] == first["candle"]["c"]

    clock["now"] = datetime(2026, 10, 28, 10, 2, 40, tzinfo=IST)
    second = feed.tick("NIFTY")
    assert second["candle"]["t"] == first["candle"]["t"]
    assert second["candle"]["v"] > first["candle"]["v"]
    assert second["candle"]["h"] >= max(first["candle"]["h"], second["price"])
    assert second["candle"]["l"] <= min(first["candle"]["l"], second["price"])

    clock["now"] = datetime(2026, 10, 28, 10, 5, 5, tzinfo=IST)
    third = feed.tick("NIFTY")
    assert third["candle"]["t"] == _ts(datetime(2026, 10, 28, 10, 5, tzinfo=IST))
    assert third["candle"]["o"] == second["candle"]["c"]
    assert third["candle"]["l"] <= third["price"] <= third["candle"]["h"]


def test_demo_ticks_freeze_outside_session(monkeypatch):
    import app.datafeed as datafeed

    clock = {"now": datetime(2026, 10, 28, 16, 0, 0, tzinfo=IST)}
    monkeypatch.setattr(datafeed, "now_ist", lambda: clock["now"])
    feed = DemoFeed()
    first = feed.tick("BANKNIFTY")
    clock["now"] = datetime(2026, 10, 28, 16, 0, 2, tzinfo=IST)
    second = feed.tick("BANKNIFTY")
    assert second["candle"] == first["candle"]
    assert second["price"] == first["price"]
    assert first["candle"]["t"].startswith("2026-10-28T15:25")


def test_market_ticks_include_change_and_bar(monkeypatch):
    import app.datafeed as datafeed

    clock = {"now": datetime(2026, 10, 28, 10, 30, 0, tzinfo=IST)}
    monkeypatch.setattr(datafeed, "now_ist", lambda: clock["now"])
    market = MarketData()
    state = market.get("NIFTY", demo=True)
    assert state["prev_close"] > 0

    ticks = market.ticks(True)
    assert set(ticks) == {"NIFTY", "BANKNIFTY", "SENSEX"}
    tick = ticks["NIFTY"]
    assert tick["price"] > 0
    assert tick["candle"]["t"].startswith("2026-10-28T10:30")
    assert math.isfinite(tick["change_pct"])
    assert tick["change"] == round(tick["price"] - state["prev_close"], 2)


def test_live_ticks_never_mix_with_simulated_candles():
    market = MarketData()
    # No Yahoo candle refresh has succeeded, so live quotes must stay off the
    # chart rather than overlaying unrelated price levels.
    assert market.ticks(False) == {}


def test_mode_isolates_accounting(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    store = Store(tmp_path / "mode.db")
    engine = TradingEngine(settings, store=store)

    engine.signals.append({"id": "SIG-DEMO", "symbol": "NIFTY", "status": "executed", "data_mode": "demo"})
    engine.positions["POS-DEMO"] = {"id": "POS-DEMO", "symbol": "NIFTY", "pnl": 150.0, "data_mode": "demo"}
    engine.trades.append({"id": "POS-DEMO", "symbol": "NIFTY", "pnl": -40.0, "data_mode": "demo"})

    demo = engine.snapshot()
    assert demo["signal"]["id"] == "SIG-DEMO"
    assert demo["stats"]["closed_trades"] == 1
    assert demo["stats"]["total"] == 110.0
    assert demo["risk"]["trades_used"] == 2

    engine.set_mode("live")
    live = engine.snapshot()
    assert live["mode"] == "live"
    assert live["signal"] is None
    assert live["positions"] == []
    assert live["trades"] == []
    assert live["stats"]["closed_trades"] == 0
    assert live["stats"]["total"] == 0.0
    assert live["risk"]["trades_used"] == 0
    store.close()


def test_minute_chart_does_not_invent_future_bars(monkeypatch):
    import app.chartdata as chartdata
    import app.datafeed as datafeed

    fixed = datetime(2026, 10, 28, 10, 32, 20, tzinfo=IST)
    monkeypatch.setattr(datafeed, "now_ist", lambda: fixed)
    monkeypatch.setattr(chartdata, "now_ist", lambda: fixed)
    settings = Settings()
    settings.demo = True
    charts = ChartService(MarketData(), settings)

    payload = charts.get("NIFTY", "1m")
    last = datetime.fromisoformat(payload["candles"][-1]["t"])
    assert last == fixed.replace(second=0, microsecond=0)
    assert all(datetime.fromisoformat(c["t"]) <= last for c in payload["candles"])
    assert payload["candles"][-2]["t"] == _ts(fixed.replace(second=0, microsecond=0) - timedelta(minutes=1))


def test_chart_timeframes_demo():
    settings = Settings()
    settings.demo = True
    charts = ChartService(MarketData(), settings)
    payload = charts.get("NIFTY", "1m")
    assert payload["tf"] == "1m" and payload["label"] == "1m"
    assert payload["bars"] >= 100
    assert len(payload["candles"]) == payload["bars"]
    assert len(payload["series"]["ema20"]) == payload["bars"]
    # History spans several sessions, so only same-day pairs are contiguous.
    minute_steps = [
        datetime.fromisoformat(b["t"]) - datetime.fromisoformat(a["t"])
        for a, b in zip(payload["candles"], payload["candles"][1:])
        if a["t"][:10] == b["t"][:10]
    ]
    assert minute_steps and all(s == timedelta(minutes=1) for s in minute_steps)

    h15 = charts.get("NIFTY", "15m")
    quarter_steps = [
        datetime.fromisoformat(b["t"]) - datetime.fromisoformat(a["t"])
        for a, b in zip(h15["candles"], h15["candles"][1:])
        if a["t"][:10] == b["t"][:10]
    ]
    assert quarter_steps and all(s == timedelta(minutes=15) for s in quarter_steps)

    daily = charts.get("NIFTY", "1d")
    assert daily["indicators"]["vwap"] is None
    assert all(v is None for v in daily["series"]["vwap"])
    d_step = datetime.fromisoformat(daily["candles"][1]["t"]) - datetime.fromisoformat(daily["candles"][0]["t"])
    assert d_step >= timedelta(days=1)

    weekly = charts.get("NIFTY", "1w")
    assert weekly["bars"] >= 100
    # Labels keep IST offsets and land on the first trading day of the week.
    first = datetime.fromisoformat(weekly["candles"][-1]["t"])
    assert first.tzinfo is not None
    assert first.weekday() == 0

    yearly = charts.get("BANKNIFTY", "1y")
    assert yearly["bars"] == 252
    assert yearly["indicators"]["vwap"] is None
    assert set(TIMEFRAMES) == {"1m", "2m", "3m", "5m", "10m", "15m", "30m", "1h", "1d", "1w", "1M", "1y"}


def test_chart_intraday_history_spans_multiple_sessions():
    """Horizontal panning must reach well past one trading session.

    The old 150-bar window ended at the previous session's 15:05 bar, so the
    chart could not scroll any further back in time.
    """
    settings = Settings()
    settings.demo = True
    charts = ChartService(MarketData(), settings)

    five = charts.get("NIFTY", "5m")
    assert five["bars"] == 600
    days = {c["t"][:10] for c in five["candles"]}
    assert len(days) >= 5, f"5m history only spans {sorted(days)}"

    minute = charts.get("NIFTY", "1m")
    assert minute["bars"] == 750
    assert len({c["t"][:10] for c in minute["candles"]}) >= 2

    quarter = charts.get("NIFTY", "15m")
    assert quarter["bars"] >= 240
    assert len({c["t"][:10] for c in quarter["candles"]}) >= 3


def _pending_signal(entry: float = 200.0, stop: float = 170.0, age: int = 0, sid: str = "SIG-WALLET") -> dict:
    stamp = now_ist() - timedelta(seconds=age)
    return {
        "id": sid, "ts": stamp.isoformat(timespec="seconds"),
        "symbol": "NIFTY", "label": "NIFTY", "side": "CE", "strike": 26000,
        "expiry": "03-Nov-2026", "spot": 26050.0, "entry": entry, "stop": stop,
        "target": entry + (entry - stop) * 2, "rr": 2.0, "confidence": 72, "score": 72,
        "bias": "BULLISH", "reasons": [], "status": "signal", "qty": 0, "lots": 0,
        "lot_size": 75, "chain_source": "synthetic", "feed_source": "simulated",
        "note": "Auto execution is off", "order_id": None, "data_mode": "demo",
    }


def test_wallet_seed_deposit_withdraw_and_caps(tmp_path):
    settings = Settings()
    settings.capital = 500000.0
    wallet = Wallet(Store(tmp_path / "wallet.db"), settings)

    assert wallet.ensure("demo") is True
    assert wallet.ensure("demo") is False  # seeded exactly once
    assert wallet.ensure("live") is False  # LIVE and DEMO share one ledger
    assert wallet.balance("demo") == 500000.0
    assert wallet.balance("live") == 500000.0

    wallet.deposit("demo", 25000, "Top up")
    wallet.withdraw("demo", 5000)
    assert wallet.balance("demo") == 520000.0
    assert wallet.balance("live") == 520000.0

    with pytest.raises(WalletError):
        wallet.withdraw("demo", 600000)
    with pytest.raises(WalletError):
        wallet.deposit("demo", -100)
    with pytest.raises(WalletError):
        wallet.deposit("demo", Wallet.MAX_TXN + 1)
    with pytest.raises(WalletError):
        wallet.debit("demo", 520001, "too big")

    kinds = [row["kind"] for row in wallet.history("demo", 10)]
    assert kinds == ["withdraw", "deposit", "opening"]
    assert wallet.history("live", 10) == wallet.history("demo", 10)


def test_engine_manual_execution_debits_and_exit_credits(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "engine_wallet.db")
    engine = TradingEngine(settings, store=store)

    engine.signals.append(_pending_signal())
    result = engine.execute_manual("SIG-WALLET")
    assert result["ok"] is True, result
    qty = result["qty"]
    assert qty == result["lots"] * 75 and qty > 0

    position = next(iter(engine.positions.values()))
    entry_cost = position["cost"]
    assert entry_cost == pytest.approx(200.7 * qty, abs=0.01)
    assert engine.wallet.balance("demo") == pytest.approx(500000.0 - entry_cost, abs=0.01)
    debit = engine.wallet.history("demo", 1)[0]
    assert debit["kind"] == "entry"
    assert debit["ref"] == position["id"]

    assert engine.close_position(position["id"]) is True
    assert not engine.positions
    closed = engine.trades[-1]
    assert closed["status"] == "CLOSED"
    credit = engine.wallet.history("demo", 1)[0]
    assert credit["kind"] == "exit"
    assert credit["ref"] == position["id"]
    assert engine.wallet.balance("demo") == pytest.approx(
        500000.0 - entry_cost + closed["proceeds"], abs=0.01
    )

    snap = engine.snapshot()
    assert snap["wallet"]["cash"] == pytest.approx(engine.wallet.balance("demo"), abs=0.01)
    assert snap["wallet"]["invested_value"] == 0.0
    assert snap["wallet"]["entries"][0]["kind"] == "exit"
    assert snap["risk"]["cash"] == snap["wallet"]["cash"]
    store.close()


def test_manual_execution_rejections(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "engine_reject.db")
    engine = TradingEngine(settings, store=store)

    engine.signals.append(_pending_signal(entry=40000.0, stop=39000.0))
    poor = engine.execute_manual("SIG-WALLET")
    assert poor["ok"] is False and "Wallet balance too low" in poor["reason"]
    assert engine.wallet.balance("demo") == 500000.0

    engine.signals.append(_pending_signal(age=20 * 60, sid="SIG-STALE"))
    old = engine.execute_manual("SIG-STALE")
    assert old["ok"] is False and "stale" in old["reason"]

    swapped = _pending_signal()
    swapped["id"] = "SIG-LIVE-ONLY"
    swapped["data_mode"] = "live"
    engine.signals.append(swapped)
    assert engine.execute_manual("SIG-LIVE-ONLY")["ok"] is False
    assert engine.execute_manual("SIG-MISSING")["ok"] is False
    store.close()


def test_loss_limit_and_sizing_follow_wallet(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "engine_equity.db")
    engine = TradingEngine(settings, store=store)

    assert engine.snapshot()["risk"]["loss_limit"] == pytest.approx(-12500.0, abs=0.01)
    engine.deposit(100000, "More capital")
    assert engine.snapshot()["risk"]["equity"] == pytest.approx(600000.0, abs=0.01)
    assert engine.snapshot()["risk"]["loss_limit"] == pytest.approx(-15000.0, abs=0.01)

    # Wallet cash, not just the risk budget, caps the lot count.
    lots_cash, _ = position_size(200.0, 170.0, 75, 600000.0, 1.0, 35.0, cash=15000.0)
    assert lots_cash == 0
    lots_ok, qty = position_size(200.0, 170.0, 75, 600000.0, 1.0, 35.0, cash=15150.0)
    assert lots_ok == 1 and qty == 75
    store.close()


def test_manual_execution_live_mode_uses_live_wallet(tmp_path, monkeypatch):
    import app.engine as engine_mod

    fixed = datetime(2026, 10, 28, 10, 30, tzinfo=IST)
    monkeypatch.setattr(engine_mod, "now_ist", lambda: fixed)

    settings = Settings()
    settings.demo = False
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "engine_live.db")
    engine = TradingEngine(settings, store=store)
    assert engine.wallet.ensure("live") is False
    assert engine.wallet.balance("live") == 500000.0

    sig = _pending_signal(sid="SIG-LIVE-WALLET")
    sig.update(ts=fixed.isoformat(timespec="seconds"), data_mode="live", feed_source="yahoo", chain_source="nse")
    engine.signals.append(sig)

    result = engine.execute_manual("SIG-LIVE-WALLET")
    assert result["ok"] is True, result
    cost = next(iter(engine.positions.values()))["cost"]
    assert engine.wallet.balance("live") == pytest.approx(500000.0 - cost, abs=0.01)
    assert engine.wallet.balance("demo") == pytest.approx(500000.0 - cost, abs=0.01)

    engine.deposit(25000, "Live top-up")
    assert engine.snapshot()["wallet"]["cash"] == pytest.approx(525000.0 - cost, abs=0.01)
    engine.set_mode("demo")
    assert engine.snapshot()["wallet"]["cash"] == pytest.approx(525000.0 - cost, abs=0.01)
    store.close()


def _set_premium(engine, premium: float, symbol: str = "NIFTY", strike: float = 26000) -> None:
    """Publish an option premium the engine can use as the live quote."""
    engine.analyses[symbol] = {
        "chain": {"rows": [{"strike": strike, "ce_ltp": premium, "pe_ltp": premium}]},
        "lot_size": 75,
    }


def test_limit_order_pends_then_fills(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "limit_order.db")
    engine = TradingEngine(settings, store=store)
    engine.signals.append(_pending_signal())
    _set_premium(engine, 200.0)

    placed = engine.place_order("SIG-WALLET", "limit", 190.0)
    assert placed["ok"] is True and placed["status"] == "pending", placed
    assert engine.wallet.balance("demo") == 500000.0  # nothing debited yet
    order = engine.orders[placed["order_id"]]
    assert order["status"] == "pending" and order["price"] == 190.0
    assert engine.signals[0]["status"] == "ordered"

    _set_premium(engine, 195.0)
    engine._manage_orders()
    assert order["status"] == "pending"

    _set_premium(engine, 188.0)
    engine._manage_orders()
    assert order["status"] == "filled"
    assert order["fill_price"] == pytest.approx(188.0)
    assert order["position_id"] in engine.positions
    position = engine.positions[order["position_id"]]
    assert position["entry"] == pytest.approx(188.0)
    assert position["order_type"] == "limit"
    assert engine.wallet.balance("demo") == pytest.approx(500000.0 - position["cost"], abs=0.01)
    assert position["cost"] == pytest.approx(188.0 * position["qty"], abs=0.01)
    assert engine.signals[0]["status"] == "executed"

    snap = engine.snapshot()
    assert snap["orders"][0]["status"] == "filled"
    assert snap["risk"]["pending_orders"] == 0
    store.close()


def test_stop_order_requires_a_trigger_above_market_and_fills_on_breakout(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "stop_order.db")
    engine = TradingEngine(settings, store=store)
    engine.signals.append(_pending_signal())
    _set_premium(engine, 200.0)

    low = engine.place_order("SIG-WALLET", "stop", 195.0)
    assert low["ok"] is False and "above the current premium" in low["reason"]

    placed = engine.place_order("SIG-WALLET", "stoploss", 210.0)
    assert placed["ok"] is True and placed["status"] == "pending", placed
    order = engine.orders[placed["order_id"]]
    assert order["order_type"] == "stop"

    _set_premium(engine, 205.0)
    engine._manage_orders()
    assert order["status"] == "pending"

    _set_premium(engine, 212.0)
    engine._manage_orders()
    assert order["status"] == "filled"
    position = engine.positions[order["position_id"]]
    assert position["entry"] == pytest.approx(212.0 * 1.0035, abs=0.01)
    assert engine.wallet.balance("demo") == pytest.approx(
        500000.0 - position["cost"], abs=0.01
    )
    store.close()


def test_marketable_limit_and_market_orders_fill_immediately(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "marketable.db")
    engine = TradingEngine(settings, store=store)
    engine.signals.append(_pending_signal())
    _set_premium(engine, 200.0)

    limit = engine.place_order("SIG-WALLET", "limit", 205.0)
    assert limit["ok"] is True and limit["status"] == "filled", limit
    assert limit["price"] == pytest.approx(200.0)  # capped by the limit, no slippage
    assert limit["order_id"] and limit["order_id"].startswith("PAPER-")
    assert not engine.orders  # no resting order was created

    engine.signals.append(_pending_signal(sid="SIG-MARKET"))
    _set_premium(engine, 200.0)
    # Free the symbol slot so a second entry is allowed.
    engine.positions.clear()
    market = engine.place_order("SIG-MARKET", "market")
    assert market["ok"] is True and market["status"] == "filled", market
    assert market["price"] == pytest.approx(200.7)  # market pays slippage
    store.close()


def test_cancel_pending_order_and_duplicate_guard(tmp_path):
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "cancel_order.db")
    engine = TradingEngine(settings, store=store)
    engine.signals.append(_pending_signal())
    _set_premium(engine, 200.0)

    placed = engine.place_order("SIG-WALLET", "limit", 190.0)
    order_id = placed["order_id"]
    duplicate = engine.place_order("SIG-WALLET", "limit", 185.0)
    assert duplicate["ok"] is False and "pending order" in duplicate["reason"]
    assert engine.execute_manual("SIG-WALLET")["ok"] is False

    cancelled = engine.cancel_order(order_id)
    assert cancelled["ok"] is True
    assert engine.orders[order_id]["status"] == "cancelled"
    assert engine.signals[0]["status"] == "signal"
    assert engine.wallet.balance("demo") == 500000.0
    assert engine.cancel_order("ORD-MISSING")["ok"] is False

    again = engine.place_order("SIG-WALLET", "limit", 190.0)
    assert again["ok"] is True and again["status"] == "pending"
    store.close()


def test_live_mode_rejects_limit_and_stop_orders(tmp_path, monkeypatch):
    import app.engine as engine_mod

    fixed = datetime(2026, 10, 28, 10, 30, tzinfo=IST)
    monkeypatch.setattr(engine_mod, "now_ist", lambda: fixed)

    settings = Settings()
    settings.demo = False
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(tmp_path / "live_orders.db")
    engine = TradingEngine(settings, store=store)
    sig = _pending_signal(sid="SIG-LIVE-ORD")
    sig.update(ts=fixed.isoformat(timespec="seconds"), data_mode="live", feed_source="yahoo", chain_source="nse")
    engine.signals.append(sig)

    limit = engine.place_order("SIG-LIVE-ORD", "limit", 190.0)
    assert limit["ok"] is False and "MARKET orders only" in limit["reason"]
    stop = engine.place_order("SIG-LIVE-ORD", "stop", 210.0)
    assert stop["ok"] is False and "MARKET orders only" in stop["reason"]
    market = engine.place_order("SIG-LIVE-ORD", "market")
    assert market["ok"] is True and market["status"] == "filled", market
    assert engine.wallet.balance("demo") == engine.wallet.balance("live")
    store.close()


def test_pending_orders_persist_across_restart(tmp_path):
    path = tmp_path / "orders_restart.db"
    settings = Settings()
    settings.demo = True
    settings.auto_trade = False
    settings.capital = 500000.0
    store = Store(path)
    engine = TradingEngine(settings, store=store)
    engine.signals.append(_pending_signal())
    _set_premium(engine, 200.0)
    placed = engine.place_order("SIG-WALLET", "limit", 190.0)
    store.close()

    store2 = Store(path)
    engine2 = TradingEngine(settings, store=store2)
    restored = engine2.orders.get(placed["order_id"])
    assert restored is not None and restored["status"] == "pending"
    assert engine2.signals[0]["status"] == "ordered"
    assert engine2.snapshot()["risk"]["pending_orders"] == 1
    store2.close()
