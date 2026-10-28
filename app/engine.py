"""Trading engine: refresh -> analyze -> signal -> risk checks -> execute.

The engine is intentionally conservative:
  * one shared cash wallet funds every entry and receives every exit
  * whole-lot sizing so a single lot never breaches the per-trade risk budget
  * daily trade cap, max open positions, and a daily loss circuit breaker
  * no live execution on synthetic data
  * automatic stop / target / 15:12 IST square-off management
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections import deque
from datetime import datetime

from .broker import Broker, BrokerError, build_broker
from .config import (
    EOD_SQUARE_OFF,
    IST,
    NO_NEW_ENTRY_AFTER,
    SYMBOLS,
    Settings,
)
from .datafeed import MarketData, market_is_open, now_ist
from .storage import Store
from .strategy import analyze
from .wallet import Wallet, WalletError
from .walletstore import build_wallet_store


def position_size(
    entry: float,
    stop: float,
    lot_size: int,
    capital: float,
    risk_pct: float,
    max_exposure_pct: float,
    cash: float | None = None,
) -> tuple[int, int]:
    """Whole lots that fit the risk budget, exposure cap and available cash."""
    risk_per_unit = max(entry - stop, 0.05)
    risk_budget = capital * risk_pct / 100.0
    lots_by_risk = int(risk_budget // (risk_per_unit * lot_size))
    exposure_budget = capital * max_exposure_pct / 100.0
    lots_by_exposure = int(exposure_budget // (entry * lot_size))
    # Keep a 1% buffer for slippage so the cost can never overdraw the wallet.
    affordable = max(float(cash if cash is not None else capital), 0.0)
    lots_by_cash = int(affordable // (entry * lot_size * 1.01)) if entry > 0 else 0
    lots = max(0, min(lots_by_risk, lots_by_exposure, lots_by_cash))
    return lots, lots * lot_size


class TradingEngine:
    def __init__(self, settings: Settings, store: Store | None = None) -> None:
        self.settings = settings
        self.data = MarketData()
        self.store = store or Store()
        self.broker: Broker
        self.broker, self.broker_note = build_broker(settings)

        self._lock = threading.RLock()
        self._refreshing = False
        self.wallet = Wallet(build_wallet_store(self.store), settings)
        self.day = now_ist().date().isoformat()
        self.analyses: dict[str, dict] = {}
        self.signals: list[dict] = []
        self.signal: dict | None = None
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.orders: dict[str, dict] = {}
        self.logs: deque[dict] = deque(maxlen=150)
        self.last_signal_at: dict[str, datetime] = {}
        self._halts: dict[str, str] = {"demo": "", "live": ""}
        self.last_refresh: str | None = None
        self.last_error: str | None = None
        self._notes_seen: dict[str, str] = {}

        self._load_today()
        if self.wallet.ensure("live"):
            self._backfill_wallet(None)
        self.log(
            "warn" if self.broker.is_live else "info",
            f"Execution: {self.broker.label} - {self.broker_note}",
        )
        if self.settings.demo:
            self.log("info", "DEMO feed active - synthetic candles and chain, paper-safe")
        self.log(
            "info",
            f"Risk: wallet Rs {self.wallet.balance(self._mode()):,.0f}, "
            f"{self.settings.risk_per_trade_pct}% per trade, "
            f"max {self.settings.max_trades_per_day} trades/day, loss limit {self.settings.daily_loss_limit_pct}%",
        )

    # ---------------------------------------------------------------- logging
    def log(self, level: str, message: str) -> None:
        self.logs.append({"ts": now_ist().strftime("%H:%M:%S"), "level": level, "msg": message})

    def _mode(self) -> str:
        return "demo" if self.settings.demo else "live"

    def _positions_view(self) -> list[dict]:
        mode = self._mode()
        return [p for p in self.positions.values() if p.get("data_mode", mode) == mode]

    def _trades_view(self) -> list[dict]:
        mode = self._mode()
        return [t for t in self.trades if t.get("data_mode", mode) == mode]

    def _signals_view(self) -> list[dict]:
        mode = self._mode()
        return [s for s in self.signals if s.get("data_mode", mode) == mode]

    def _orders_view(self) -> list[dict]:
        mode = self._mode()
        return [o for o in self.orders.values() if o.get("data_mode", mode) == mode]

    def _pending_orders(self) -> list[dict]:
        return [o for o in self._orders_view() if o.get("status") == "pending"]

    @property
    def halted(self) -> bool:
        return bool(self._halts[self._mode()])

    @property
    def halt_reason(self) -> str:
        return self._halts[self._mode()]

    # ------------------------------------------------------------ persistence
    def _load_today(self) -> None:
        try:
            signals, trades = self.store.load_day(self.day)
            orders = self.store.load_orders(self.day)
        except Exception as exc:
            self.last_error = f"store load failed: {exc}"
            return
        self.signals = signals[-24:]
        self.signal = signals[-1] if signals else None
        mode = self._mode()
        for sig in self.signals:
            sig.setdefault("data_mode", mode)
        for trade in trades:
            trade.setdefault("data_mode", mode)
            if trade.get("status") == "OPEN":
                self.positions[trade["id"]] = trade
            else:
                self.trades.append(trade)
        for order in orders:
            order.setdefault("data_mode", mode)
            self.orders[order["id"]] = order
        if self.positions or self.trades:
            self.log("info", f"Restored {len(self.trades)} closed and {len(self.positions)} open trade(s) for today")
        pending = self._pending_orders()
        if pending:
            self.log("info", f"Restored {len(pending)} pending order(s) for today")

    def _backfill_wallet(self, mode: str | None = None) -> None:
        """Reconcile today's pre-wallet trades the first time a ledger is created."""
        adjusted = 0
        for trade in self.trades:
            if mode is not None and trade.get("data_mode", mode) != mode:
                continue
            pnl = round(float(trade.get("pnl") or 0.0), 2)
            if pnl:
                self.wallet.adjust(mode or "live", pnl, f"Historic P&L {trade.get('id', '')}", trade.get("id", ""))
                adjusted += 1
        for pos in self.positions.values():
            if mode is not None and pos.get("data_mode", mode) != mode:
                continue
            cost = round(float(pos.get("entry") or 0.0) * float(pos.get("qty") or 0), 2)
            if cost:
                self.wallet.adjust(mode or "live", -cost, f"Existing position {pos.get('id', '')}", pos.get("id", ""))
                adjusted += 1
        if adjusted:
            self.log("info", "Wallet opening balance reconciled with today's restored trades")

    # ------------------------------------------------------------------ wallet
    def _wallet_view(self, mode: str | None = None) -> dict:
        mode = mode or self._mode()
        positions = [p for p in self.positions.values() if p.get("data_mode", mode) == mode]
        cash = self.wallet.balance(mode)
        invested = 0.0
        value = 0.0
        for pos in positions:
            qty = float(pos.get("qty") or 0.0)
            entry = float(pos.get("entry") or 0.0)
            ltp = float(pos.get("ltp") or entry)
            invested += entry * qty
            value += ltp * qty
        equity = cash + value
        return {
            "mode": mode,
            "currency": "INR",
            "backend": self.wallet.backend,
            "shared": self.wallet.backend == "postgres",
            "cash": round(cash, 2),
            "invested": round(invested, 2),
            "invested_value": round(value, 2),
            "unrealized": round(value - invested, 2),
            "equity": round(equity, 2),
            "available": round(cash, 2),
            "pnl_today": round(self._daily_pnl(), 2),
            "live": self.broker.is_live,
            "entries": self.wallet.history(mode, 6),
        }

    def deposit(self, amount: float, note: str = "") -> dict:
        with self._lock:
            entry = self.wallet.deposit(self._mode(), amount, note)
            self.log(
                "info",
                f"Wallet deposit Rs {entry['amount']:,.0f} - balance Rs {entry['balance_after']:,.0f}",
            )
            return entry

    def withdraw(self, amount: float, note: str = "") -> dict:
        with self._lock:
            entry = self.wallet.withdraw(self._mode(), amount, note)
            self.log(
                "warn",
                f"Wallet withdrawal Rs {abs(entry['amount']):,.0f} - balance Rs {entry['balance_after']:,.0f}",
            )
            return entry

    # ------------------------------------------------------------------ loop
    def refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            demo = self.settings.demo
            fresh: dict[str, dict] = {}
            threshold = self.settings.active_confidence_threshold
            for symbol in SYMBOLS:
                try:
                    data = self.data.get(symbol, demo)
                    fresh[symbol] = analyze(
                        symbol=symbol,
                        candles=data["candles"],
                        chain=data["chain"],
                        prev_close=data["prev_close"],
                        threshold=threshold,
                        feed_source=data["feed_source"],
                        note=data["note"],
                    )
                except Exception as exc:
                    self.last_error = f"{symbol}: {type(exc).__name__}: {exc}"
                    self.log("error", f"{symbol} refresh failed: {type(exc).__name__}: {exc}")

            with self._lock:
                if self.settings.demo != demo:
                    # A mode switch landed mid-fetch; drop the stale result.
                    return
                self._roll_day()
                self.analyses.update(fresh)
                for symbol, analysis in fresh.items():
                    note = analysis.get("note") or ""
                    if note and self._notes_seen.get(symbol) != note:
                        self._notes_seen[symbol] = note
                        self.log("warn", f"{symbol}: {note}")

                trading = self.settings.demo or market_is_open()
                if trading:
                    self._manage_positions()
                    self._manage_orders()
                    self._check_loss_limit()
                    self._scan_signals()
                self.last_refresh = now_ist().isoformat(timespec="seconds")
                self.last_error = None
        finally:
            self._refreshing = False

    def _roll_day(self) -> None:
        today = now_ist().date().isoformat()
        if today != self.day:
            for order in list(self.orders.values()):
                if order.get("status") == "pending":
                    self._finish_order(order, "cancelled", "Session closed")
            self.day = today
            self.trades = []
            self.signals = []
            self.signal = None
            self.orders = {}
            self.last_signal_at.clear()
            self._halts = {"demo": "", "live": ""}
            self.log("info", "New session - daily counters reset")

    # -------------------------------------------------------------- positions
    def _option_ltp(self, symbol: str, strike: float, side: str) -> float:
        analysis = self.analyses.get(symbol)
        if not analysis:
            return 0.0
        for row in analysis["chain"]["rows"]:
            if abs(float(row["strike"]) - strike) < 0.01:
                key = "ce_ltp" if side == "CE" else "pe_ltp"
                return float(row.get(key) or 0.0)
        return 0.0

    def _manage_positions(self) -> None:
        now = now_ist()
        current_mode = "demo" if self.settings.demo else "live"
        for pos in list(self.positions.values()):
            if pos.get("data_mode", current_mode) != current_mode:
                continue
            ltp = self._option_ltp(pos["symbol"], pos["strike"], pos["side"])
            if ltp <= 0:
                continue
            pos["ltp"] = round(ltp, 2)
            pos["pnl"] = round((ltp - pos["entry"]) * pos["qty"], 2)
            pos["pnl_pct"] = round((ltp / pos["entry"] - 1) * 100, 2) if pos["entry"] else 0.0

            reason = None
            if ltp <= pos["stop"]:
                reason = "stop"
            elif ltp >= pos["target"]:
                reason = "target"
            elif not self.settings.demo and now.time() >= EOD_SQUARE_OFF:
                reason = "eod"
            if reason:
                self._exit_position(pos, ltp, reason)

    def _exit_position(self, pos: dict, price_hint: float, reason: str) -> None:
        try:
            fill = self.broker.sell(
                pos["symbol"], pos["side"], pos["strike"], pos["expiry"], pos["qty"], price_hint
            )
        except BrokerError as exc:
            self.log("error", f"Exit rejected for {pos['id']}: {exc}")
            return

        mode = pos.get("data_mode", self._mode())
        proceeds = round(fill.price * pos["qty"], 2)
        try:
            self.wallet.credit(
                mode,
                proceeds,
                f"SELL {pos['symbol']} {int(pos['strike'])} {pos['side']} x{pos['qty']}",
                pos["id"],
            )
        except WalletError as exc:
            self.log("error", f"Wallet credit failed for {pos['id']}: {exc}")
        pos["proceeds"] = proceeds

        pos["status"] = "CLOSED"
        pos["closed_at"] = now_ist().isoformat(timespec="seconds")
        pos["exit_price"] = round(fill.price, 2)
        pos["exit_reason"] = reason
        pos["ltp"] = round(fill.price, 2)
        pos["pnl"] = round((fill.price - pos["entry"]) * pos["qty"], 2)
        pos["pnl_pct"] = round((fill.price / pos["entry"] - 1) * 100, 2) if pos["entry"] else 0.0
        pos["exit_order_id"] = fill.order_id

        self.positions.pop(pos["id"], None)
        self.trades.append(pos)
        self.store.save_trade(pos)
        self._update_signal(pos["signal_id"], "closed", f"{reason.upper()} exit at {fill.price:,.2f}")

        level = "trade" if pos["pnl"] >= 0 else "warn"
        self.log(
            level,
            f"{reason.upper()} {pos['symbol']} {int(pos['strike'])} {pos['side']} @ {fill.price:,.2f} "
            f"PnL Rs {pos['pnl']:,.0f}",
        )

    def close_position(self, position_id: str) -> bool:
        with self._lock:
            pos = self.positions.get(position_id)
            if not pos:
                return False
            current_mode = "demo" if self.settings.demo else "live"
            if pos.get("data_mode", current_mode) == current_mode:
                ltp = self._option_ltp(pos["symbol"], pos["strike"], pos["side"])
            else:
                ltp = 0.0
            ltp = ltp or pos.get("ltp") or pos["entry"]
            self._exit_position(pos, ltp, "manual")
            return True

    def square_off_all(self) -> int:
        closed = 0
        with self._lock:
            current_mode = self._mode()
            for pos in list(self._positions_view()):
                if pos.get("data_mode", current_mode) == current_mode:
                    ltp = self._option_ltp(pos["symbol"], pos["strike"], pos["side"])
                else:
                    ltp = 0.0
                ltp = ltp or pos.get("ltp") or pos["entry"]
                before = len(self.positions)
                self._exit_position(pos, ltp, "manual")
                if len(self.positions) < before:
                    closed += 1
        return closed

    # ---------------------------------------------------------------- signals
    def _scan_signals(self) -> None:
        now = now_ist()
        analysis_symbols = set(self.analyses)
        if not analysis_symbols:
            return
        if not self.settings.demo and now.time() > NO_NEW_ENTRY_AFTER:
            return

        for symbol in SYMBOLS:
            analysis = self.analyses.get(symbol)
            if not analysis:
                continue
            draft = analysis.get("draft")
            if draft is None:
                continue
            if any(p["symbol"] == symbol for p in self._positions_view()):
                continue
            if any(o["symbol"] == symbol for o in self._pending_orders()):
                continue
            last = self.last_signal_at.get(symbol)
            cooldown = self.settings.active_cooldown_minutes * 60
            if last and (now - last).total_seconds() < cooldown:
                continue

            lots, qty = position_size(
                draft.entry,
                draft.stop,
                analysis["lot_size"],
                self._wallet_view()["equity"],
                self.settings.risk_per_trade_pct,
                self.settings.max_exposure_pct,
                cash=self.wallet.balance(self._mode()),
            )
            block = self._auto_block_reason(analysis, draft, qty, now)
            status = "signal" if block == "Auto execution is off" else ("blocked" if block else "signal")
            signal = {
                "id": f"SIG-{uuid.uuid4().hex[:8].upper()}",
                "ts": now.isoformat(timespec="seconds"),
                "symbol": symbol,
                "label": analysis["label"],
                "side": draft.side,
                "strike": draft.strike,
                "expiry": draft.expiry,
                "spot": draft.spot,
                "entry": draft.entry,
                "stop": draft.stop,
                "target": draft.target,
                "rr": draft.rr,
                "confidence": draft.confidence,
                "score": draft.score,
                "bias": draft.bias,
                "reasons": draft.reasons,
                "status": status,
                "qty": qty,
                "lots": lots,
                "lot_size": analysis["lot_size"],
                "chain_source": draft.chain_source,
                "feed_source": analysis["feed_source"],
                "note": block,
                "order_id": None,
                "data_mode": self._mode(),
            }
            self._add_signal(signal)
            self.last_signal_at[symbol] = now

            if not block and self.settings.auto_trade:
                self._execute_signal(signal, lots, qty)

    def _auto_block_reason(self, analysis: dict, draft, qty: int, now: datetime) -> str:
        if not self.settings.auto_trade:
            return "Auto execution is off"
        if self.halted:
            return f"Trading halted: {self.halt_reason}"
        if self.settings.live_execution and draft.chain_source != "nse":
            return "Live execution is only wired for NSE option chains"
        if not self.settings.demo and analysis["feed_source"] != "yahoo":
            return "Execution blocked - index feed is simulated"
        trades_used = len(self._trades_view()) + len(self._positions_view()) + len(self._pending_orders())
        if trades_used >= self.settings.max_trades_per_day:
            return f"Daily trade cap reached ({self.settings.max_trades_per_day})"
        if len(self._positions_view()) + len(self._pending_orders()) >= self.settings.max_open_positions:
            return f"Max open positions reached ({self.settings.max_open_positions})"
        if self.wallet.balance(self._mode()) < draft.entry * analysis["lot_size"]:
            return "Wallet balance too low for one lot"
        if qty <= 0:
            return "Risk budget too small for one lot"
        if not self.settings.demo and now.time() > NO_NEW_ENTRY_AFTER:
            return "No new entries after 15:05 IST"
        return ""

    def _add_signal(self, signal: dict) -> None:
        self.signals.append(signal)
        self.signals = self.signals[-24:]
        self.signal = signal
        self.store.save_signal(signal)
        self.log(
            "signal",
            f"{signal['status'].upper()} {signal['symbol']} {int(signal['strike'])} {signal['side']} "
            f"conf {signal['confidence']}% entry {signal['entry']:,.2f}"
            + (f" - {signal['note']}" if signal["note"] else ""),
        )

    def _execute_signal(
        self,
        signal: dict,
        lots: int,
        qty: int,
        order_type: str = "market",
        price_hint: float | None = None,
        limit_price: float | None = None,
    ) -> dict:
        """Send one entry and return a result the API and order book can use."""
        mode = self._mode()

        def reject(reason: str) -> dict:
            signal["status"] = "rejected"
            signal["note"] = reason
            self.store.save_signal(signal)
            self.log("error", f"Entry rejected: {reason}")
            return {
                "ok": False,
                "signal_id": signal["id"],
                "reason": reason,
                "qty": qty,
                "lots": lots,
                "cash": self.wallet.balance(mode),
                "position": None,
                "fill_price": None,
            }

        if lots <= 0 or qty <= 0:
            return reject("Risk budget too small for one lot")
        hint = round(float(price_hint or signal["entry"]), 2)
        # A limit order can never cost more than its limit, so size the cash
        # check against that cap instead of the quoted premium.
        worst = float(limit_price) if (order_type == "limit" and limit_price) else hint
        estimate = round(worst * qty * 1.01, 2)
        cash = self.wallet.balance(mode)
        if estimate > cash + 0.005:
            return reject(
                f"Insufficient wallet balance: need Rs {estimate:,.0f}, have Rs {cash:,.0f}"
            )
        try:
            fill = self.broker.buy(
                signal["symbol"],
                signal["side"],
                signal["strike"],
                signal["expiry"],
                qty,
                hint,
                order_type=order_type,
                limit_price=limit_price,
            )
        except BrokerError as exc:
            return reject(str(exc))

        position_id = f"POS-{uuid.uuid4().hex[:8].upper()}"
        cost = round(fill.price * qty, 2)
        try:
            self.wallet.debit(
                mode,
                cost,
                f"BUY {signal['symbol']} {int(signal['strike'])} {signal['side']} x{qty}",
                position_id,
            )
        except WalletError as exc:
            # A live market fill can land above the estimate; keep the books honest.
            self.wallet.adjust(
                mode,
                -cost,
                f"BUY {signal['symbol']} {int(signal['strike'])} {signal['side']} x{qty} (overdraft)",
                position_id,
            )
            self.log("error", f"Wallet overdrawn by Rs {cost:,.2f}: {exc}")

        position = {
            "id": position_id,
            "signal_id": signal["id"],
            "symbol": signal["symbol"],
            "label": signal["label"],
            "side": signal["side"],
            "strike": signal["strike"],
            "expiry": signal["expiry"],
            "qty": qty,
            "lots": lots,
            "lot_size": signal["lot_size"],
            "entry": round(fill.price, 2),
            "cost": cost,
            "ltp": round(fill.price, 2),
            "stop": signal["stop"],
            "target": signal["target"],
            "opened_at": now_ist().isoformat(timespec="seconds"),
            "status": "OPEN",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "broker": self.broker.name,
            "order_type": order_type,
            "order_id": fill.order_id,
            "exit_price": None,
            "closed_at": None,
            "exit_reason": None,
            "chain_source": signal["chain_source"],
            "data_mode": "demo" if self.settings.demo else "live",
        }
        self.positions[position["id"]] = position
        self.store.save_trade(position)

        signal["status"] = "executed"
        signal["qty"] = qty
        signal["lots"] = lots
        signal["order_id"] = fill.order_id
        signal["note"] = (
            f"Filled at {fill.price:,.2f} ({lots} lot{'s' if lots > 1 else ''})"
            if order_type == "market"
            else f"{order_type.upper()} filled at {fill.price:,.2f} ({lots} lot{'s' if lots > 1 else ''})"
        )
        self.store.save_signal(signal)
        self.log(
            "trade",
            f"BUY {signal['symbol']} {int(signal['strike'])} {signal['side']} x{qty} @ {fill.price:,.2f} "
            f"[{order_type.upper()} {self.broker.label}]",
        )
        return {
            "ok": True,
            "signal_id": signal["id"],
            "reason": "",
            "qty": qty,
            "lots": lots,
            "cash": self.wallet.balance(mode),
            "position": position,
            "fill_price": round(fill.price, 2),
            "order_id": fill.order_id,
        }

    def _update_signal(self, signal_id: str, status: str, note: str) -> None:
        for sig in self.signals:
            if sig["id"] == signal_id:
                sig["status"] = status
                sig["note"] = note
                self.store.save_signal(sig)
                break
        if self.signal and self.signal["id"] == signal_id:
            self.signal["status"] = status
            self.signal["note"] = note

    # ------------------------------------------------------------------- risk
    def _daily_pnl(self) -> float:
        realized = sum(float(t.get("pnl") or 0.0) for t in self._trades_view())
        unrealized = sum(float(p.get("pnl") or 0.0) for p in self._positions_view())
        return realized + unrealized

    def _loss_limit(self) -> float:
        """Daily loss budget, anchored to equity before today's P&L."""
        equity = self._wallet_view()["equity"]
        base = max(equity - self._daily_pnl(), 1.0)
        return round(-base * self.settings.daily_loss_limit_pct / 100.0, 2)

    def _check_loss_limit(self) -> None:
        if self.halted:
            return
        limit = self._loss_limit()
        pnl = self._daily_pnl()
        if pnl <= limit:
            self._halts[self._mode()] = f"Daily loss limit hit (Rs {pnl:,.0f})"
            self.log("error", f"RISK HALT: {self.halt_reason}. New entries disabled.")
            if self._positions_view():
                self.square_off_all()
                self.log("warn", "Open positions squared off after loss-limit breach")

    def resume(self) -> None:
        with self._lock:
            self._halts[self._mode()] = ""
            self.log("info", "Risk halt manually cleared by user")

    def _entry_guard(self, sig: dict | None, now: datetime | None = None) -> str:
        """Shared entry checks for market, limit and stop orders."""
        mode = self._mode()
        if sig is None or sig.get("data_mode", mode) != mode:
            return "Signal not found in this mode"
        if sig.get("status") == "executed":
            return "Signal already executed"
        if sig.get("status") == "ordered":
            return "Signal already has a pending order"
        if sig.get("status") in {"rejected", "closed"}:
            return f"Signal is {sig['status']}"
        if any(p["symbol"] == sig["symbol"] for p in self._positions_view()):
            return "A position is already open on this symbol"
        pending = self._pending_orders()
        if any(o["symbol"] == sig["symbol"] for o in pending):
            return "A pending order already exists for this symbol"
        if self.halted:
            return f"Trading halted: {self.halt_reason}"
        now = now or now_ist()
        if not self.settings.demo and now.time() > NO_NEW_ENTRY_AFTER:
            return "No new entries after 15:05 IST"
        if self.settings.live_execution and sig.get("chain_source") != "nse":
            return "Live execution is only wired for NSE option chains"
        if not self.settings.demo and sig.get("feed_source") != "yahoo":
            return "Execution blocked - index feed is simulated"
        trades_used = len(self._trades_view()) + len(self._positions_view()) + len(pending)
        if trades_used >= self.settings.max_trades_per_day:
            return f"Daily trade cap reached ({self.settings.max_trades_per_day})"
        if len(self._positions_view()) + len(pending) >= self.settings.max_open_positions:
            return f"Max open positions reached ({self.settings.max_open_positions})"
        try:
            age = (now - datetime.fromisoformat(sig["ts"])).total_seconds()
        except (KeyError, TypeError, ValueError):
            age = 0.0
        max_age = 600.0 if self.settings.demo else 900.0
        if age > max_age:
            return f"Signal is stale ({int(age // 60)} min old, limit {int(max_age // 60)} min)"
        return ""

    def _sizing(self, sig: dict) -> tuple[int, int, int, str]:
        """(lots, qty, lot_size, reason) sized against the active wallet."""
        mode = self._mode()
        analysis = self.analyses.get(sig["symbol"])
        lot_size = int(sig.get("lot_size") or (analysis["lot_size"] if analysis else 0) or 0)
        if lot_size <= 0:
            return 0, 0, 0, "Lot size unavailable"
        cash = self.wallet.balance(mode)
        lots, qty = position_size(
            float(sig["entry"]),
            float(sig["stop"]),
            lot_size,
            self._wallet_view()["equity"],
            self.settings.risk_per_trade_pct,
            self.settings.max_exposure_pct,
            cash=cash,
        )
        if qty <= 0:
            reason = (
                "Wallet balance too low for one lot"
                if cash < float(sig["entry"]) * lot_size
                else "Risk budget too small for one lot"
            )
            return lots, qty, lot_size, reason
        return lots, qty, lot_size, ""

    def execute_manual(self, signal_id: str) -> dict:
        """Execute a pending signal at market, funded from the active wallet."""
        with self._lock:
            sig = next((s for s in self.signals if s.get("id") == signal_id), None)
            reason = self._entry_guard(sig)
            if reason:
                return {"ok": False, "signal_id": signal_id, "reason": reason}
            lots, qty, _, reason = self._sizing(sig)
            if reason:
                return {"ok": False, "signal_id": signal_id, "reason": reason}
            result = self._execute_signal(sig, lots, qty, "market")
            return {**result, "signal_id": signal_id, "order_type": "market", "status": "filled" if result["ok"] else "rejected"}

    def place_order(
        self, signal_id: str, order_type: str = "market", price: float | None = None
    ) -> dict:
        """Place a MARKET, LIMIT or STOP entry for a signal.

        Paper mode simulates pending LIMIT/STOP orders and fills them when the
        premium crosses the trigger. Live Kite execution stays market-only.
        """
        with self._lock:
            mode = self._mode()
            otype = str(order_type or "market").strip().lower().replace("-", "_")
            otype = {"sl": "stop", "stop_loss": "stop", "stoploss": "stop"}.get(otype, otype)
            if otype not in {"market", "limit", "stop"}:
                return {
                    "ok": False,
                    "signal_id": signal_id,
                    "order_type": order_type,
                    "reason": "Order type must be market, limit or stop",
                }
            sig = next((s for s in self.signals if s.get("id") == signal_id), None)
            reason = self._entry_guard(sig)
            if reason:
                return {"ok": False, "signal_id": signal_id, "order_type": otype, "reason": reason}
            ref = self._option_ltp(sig["symbol"], float(sig["strike"]), sig["side"]) or float(sig["entry"])
            ref = round(float(ref), 2)
            if not self.settings.demo and otype != "market":
                return {
                    "ok": False,
                    "signal_id": signal_id,
                    "order_type": otype,
                    "reason": "Live execution uses MARKET orders only",
                }
            limit = 0.0
            if otype != "market":
                try:
                    limit = round(float(price), 2)
                except (TypeError, ValueError):
                    limit = 0.0
                if limit <= 0:
                    return {
                        "ok": False,
                        "signal_id": signal_id,
                        "order_type": otype,
                        "reason": "Enter a valid limit/stop price",
                    }
                if otype == "stop" and limit <= ref:
                    return {
                        "ok": False,
                        "signal_id": signal_id,
                        "order_type": otype,
                        "reason": f"Stop trigger must be above the current premium (Rs {ref:,.2f})",
                    }
            lots, qty, _, reason = self._sizing(sig)
            if reason:
                return {"ok": False, "signal_id": signal_id, "order_type": otype, "reason": reason}

            if otype == "market":
                result = self._execute_signal(sig, lots, qty, "market", ref)
                return {
                    **result,
                    "signal_id": signal_id,
                    "order_type": "market",
                    "status": "filled" if result["ok"] else "rejected",
                    "price": result.get("fill_price") or ref,
                    "order_id": None if not result["ok"] else result.get("order_id"),
                }
            if otype == "limit" and limit >= ref:
                # Marketable limit: fill now, but never above the limit.
                result = self._execute_signal(sig, lots, qty, "limit", ref, limit_price=limit)
                return {
                    **result,
                    "signal_id": signal_id,
                    "order_type": "limit",
                    "status": "filled" if result["ok"] else "rejected",
                    "price": result.get("fill_price") or ref,
                    "order_id": None if not result["ok"] else result.get("order_id"),
                }

            order = {
                "id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
                "ts": now_ist().isoformat(timespec="seconds"),
                "signal_id": sig["id"],
                "symbol": sig["symbol"],
                "label": sig.get("label", sig["symbol"]),
                "side": sig["side"],
                "strike": sig["strike"],
                "expiry": sig["expiry"],
                "order_type": otype,
                "price": limit,
                "ltp": ref,
                "qty": qty,
                "lots": lots,
                "lot_size": sig.get("lot_size"),
                "status": "pending",
                "note": (
                    f"Waiting for {sig['side']} premium "
                    f"{'<=' if otype == 'limit' else '>='} {limit:,.2f}"
                ),
                "position_id": None,
                "filled_at": None,
                "fill_price": None,
                "broker_order_id": None,
                "data_mode": mode,
            }
            self.orders[order["id"]] = order
            self.store.save_order(order)
            sig["status"] = "ordered"
            sig["note"] = f"{otype.upper()} order pending at {limit:,.2f}"
            self.store.save_signal(sig)
            self.log(
                "info",
                f"{otype.upper()} order {order['id']} {sig['symbol']} {int(sig['strike'])} {sig['side']} "
                f"x{qty} @ {limit:,.2f} (LTP {ref:,.2f})",
            )
            return {
                "ok": True,
                "signal_id": signal_id,
                "order_id": order["id"],
                "order_type": otype,
                "status": "pending",
                "reason": "",
                "qty": qty,
                "lots": lots,
                "price": limit,
                "cash": self.wallet.balance(mode),
            }

    def cancel_order(self, order_id: str) -> dict:
        with self._lock:
            order = self.orders.get(order_id)
            mode = self._mode()
            if order is None or order.get("data_mode", mode) != mode:
                return {"ok": False, "order_id": order_id, "reason": "Order not found in this mode"}
            if order.get("status") != "pending":
                return {
                    "ok": False,
                    "order_id": order_id,
                    "reason": f"Order is already {order.get('status', 'closed')}",
                }
            self._finish_order(order, "cancelled", "Cancelled by user")
            return {"ok": True, "order_id": order_id, "status": "cancelled", "reason": ""}

    def _finish_order(self, order: dict, status: str, note: str) -> None:
        order["status"] = status
        order["note"] = note
        order["closed_at"] = now_ist().isoformat(timespec="seconds")
        self.store.save_order(order)
        sig = next((s for s in self.signals if s.get("id") == order.get("signal_id")), None)
        if sig and sig.get("status") == "ordered":
            sig["status"] = "signal"
            sig["note"] = f"{str(order.get('order_type', '')).upper()} {status}: {note}"
            self.store.save_signal(sig)
        self.log(
            "info" if status == "cancelled" else "error",
            f"{str(order.get('order_type', '')).upper()} order {order.get('id', '')} {status}: {note}",
        )

    def _manage_orders(self) -> None:
        """Trigger pending LIMIT/STOP entries from the latest option premium."""
        now = now_ist()
        mode = self._mode()
        for order in list(self.orders.values()):
            if order.get("status") != "pending" or order.get("data_mode", mode) != mode:
                continue
            sig = next((s for s in self.signals if s.get("id") == order.get("signal_id")), None)
            if sig is None:
                self._finish_order(order, "cancelled", "Signal is no longer available")
                continue
            if self.halted:
                self._finish_order(order, "cancelled", f"Trading halted: {self.halt_reason}")
                continue
            if not self.settings.demo and now.time() > NO_NEW_ENTRY_AFTER:
                self._finish_order(order, "cancelled", "No new entries after 15:05 IST")
                continue
            ref = self._option_ltp(order["symbol"], float(order["strike"]), order["side"])
            if ref <= 0:
                continue
            ref = round(float(ref), 2)
            if order.get("ltp") != ref:
                order["ltp"] = ref
                self.store.save_order(order)
            triggered = (
                order["order_type"] == "limit" and ref <= float(order["price"])
            ) or (
                order["order_type"] == "stop" and ref >= float(order["price"])
            )
            if not triggered:
                continue
            if any(p["symbol"] == order["symbol"] for p in self._positions_view()):
                self._finish_order(order, "rejected", "A position is already open on this symbol")
                continue
            result = self._execute_signal(
                sig,
                int(order.get("lots") or 0),
                int(order.get("qty") or 0),
                order["order_type"],
                ref,
                limit_price=float(order["price"]) if order["order_type"] == "limit" else None,
            )
            if not result["ok"]:
                self._finish_order(order, "rejected", result.get("reason") or "Entry rejected")
                continue
            order["status"] = "filled"
            order["note"] = f"Filled at {result['fill_price']:,.2f}"
            order["filled_at"] = now.isoformat(timespec="seconds")
            order["fill_price"] = result["fill_price"]
            order["position_id"] = result["position"]["id"] if result.get("position") else None
            order["broker_order_id"] = result.get("order_id")
            self.store.save_order(order)
            self.log(
                "trade",
                f"{order['order_type'].upper()} order {order['id']} filled "
                f"{order['symbol']} {int(order['strike'])} {order['side']} x{order['qty']} "
                f"@ {result['fill_price']:,.2f}",
            )

    # --------------------------------------------------------------- settings
    def set_auto(self, enabled: bool) -> None:
        with self._lock:
            self.settings.auto_trade = enabled
            self.log("info", f"Auto execution {'ENABLED' if enabled else 'DISABLED'}")

    def set_mode(self, mode: str) -> None:
        if mode not in {"live", "demo"}:
            raise ValueError("mode must be live or demo")
        with self._lock:
            demo = mode == "demo"
            if demo != self.settings.demo:
                self.settings.demo = demo
                self.last_signal_at.clear()
                self.analyses.clear()
                self._notes_seen.clear()
                self.data.reset()
                self.log("info", f"Switched to {'DEMO' if demo else 'LIVE'} data mode")

    # --------------------------------------------------------------- snapshot
    def _stats(self) -> dict:
        trades = self._trades_view()
        positions = self._positions_view()
        realized = sum(float(t.get("pnl") or 0.0) for t in trades)
        unrealized = sum(float(p.get("pnl") or 0.0) for p in positions)
        wins = sum(1 for t in trades if float(t.get("pnl") or 0) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl") or 0) < 0)
        closed = len(trades)
        return {
            "realized": round(realized, 2),
            "unrealized": round(unrealized, 2),
            "total": round(realized + unrealized, 2),
            "wins": wins,
            "losses": losses,
            "closed_trades": closed,
            "win_rate": round(wins / closed * 100, 1) if closed else 0.0,
            "open_positions": len(positions),
        }

    def ticks(self) -> dict:
        """Lightweight one-second frame for the chart's forming candle."""
        demo = self.settings.demo
        return {
            "type": "tick",
            "mode": "demo" if demo else "live",
            "source": "simulated" if demo else "yahoo",
            "open": True if demo else market_is_open(),
            "server_time": now_ist().isoformat(timespec="seconds"),
            "prices": self.data.ticks(demo),
        }

    def snapshot(self) -> dict:
        with self._lock:
            now = now_ist()
            open_now = self.settings.demo or market_is_open(now)
            analyses = {}
            for key, analysis in self.analyses.items():
                item = {k: v for k, v in analysis.items() if k != "draft"}
                analyses[key] = item
            stats = self._stats()
            wallet = self._wallet_view()
            limit = self._loss_limit()
            signals = self._signals_view()
            positions = self._positions_view()
            trades = self._trades_view()
            orders = self._orders_view()
            trades_used = len(trades) + len(positions)
            pending = sum(1 for o in orders if o.get("status") == "pending")
            payload = {
                "server_time": now.isoformat(timespec="seconds"),
                "day": self.day,
                "mode": "demo" if self.settings.demo else "live",
                "market": {
                    "open": open_now,
                    "label": "DEMO" if self.settings.demo else ("OPEN" if open_now else "CLOSED"),
                    "clock": now.strftime("%H:%M:%S"),
                    "date": now.strftime("%a, %d %b %Y"),
                    "session": "09:15 - 15:30 IST",
                },
                "execution": {
                    "broker": self.broker.name,
                    "label": self.broker.label,
                    "live": self.broker.is_live,
                    "note": self.broker_note,
                },
                "settings": {
                    "auto_trade": self.settings.auto_trade,
                    "confidence_threshold": self.settings.active_confidence_threshold,
                    "capital": self.settings.capital,
                    "risk_per_trade_pct": self.settings.risk_per_trade_pct,
                    "max_trades_per_day": self.settings.max_trades_per_day,
                    "max_open_positions": self.settings.max_open_positions,
                    "daily_loss_limit_pct": self.settings.daily_loss_limit_pct,
                    "reward_risk": self.settings.reward_risk,
                },
                "risk": {
                    "halted": self.halted,
                    "reason": self.halt_reason,
                    "trades_used": trades_used,
                    "pending_orders": pending,
                    "max_trades": self.settings.max_trades_per_day,
                    "daily_pnl": round(self._daily_pnl(), 2),
                    "loss_limit": round(limit, 2),
                    "equity": wallet["equity"],
                    "cash": wallet["cash"],
                },
                "wallet": wallet,
                "symbols": analyses,
                "signal": copy.deepcopy(signals[-1]) if signals else None,
                "signals": copy.deepcopy(list(reversed(signals[-12:]))),
                "positions": copy.deepcopy(positions),
                "orders": copy.deepcopy(list(reversed(orders[-20:]))),
                "trades": copy.deepcopy(list(reversed(trades[-20:]))),
                "stats": stats,
                "logs": list(self.logs)[-40:],
                "last_refresh": self.last_refresh,
                "last_error": self.last_error,
            }
        return payload
