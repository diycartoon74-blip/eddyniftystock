"""Broker adapters.

PaperBroker is the default and simulates fills with slippage. KiteBroker is a
thin Kite Connect adapter that only activates when live trading is explicitly
enabled and credentials are present. No orders are ever sent in paper mode.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from .config import Settings, SYMBOLS


class BrokerError(RuntimeError):
    pass


@dataclass
class Fill:
    order_id: str
    price: float
    qty: int


class Broker:
    name = "paper"
    label = "PAPER"
    is_live = False

    def buy(
        self,
        symbol: str,
        side: str,
        strike: float,
        expiry: str,
        qty: int,
        price_hint: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Fill:
        raise NotImplementedError

    def sell(
        self,
        symbol: str,
        side: str,
        strike: float,
        expiry: str,
        qty: int,
        price_hint: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Fill:
        raise NotImplementedError


class PaperBroker(Broker):
    """Deterministic paper fills.

    MARKET and STOP fills cross the spread with 0.35% adverse slippage. LIMIT
    fills never pay worse than the limit price, so they fill at the market
    price with no extra slippage.
    """

    name = "paper"
    label = "PAPER"
    is_live = False
    SLIPPAGE = 0.0035
    SUPPORTED = {"market", "limit", "stop"}

    def _fill(
        self,
        qty: int,
        price: float,
        buying: bool,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Fill:
        if qty <= 0:
            raise BrokerError("quantity must be positive")
        if price <= 0:
            raise BrokerError("invalid price")
        if order_type not in self.SUPPORTED:
            raise BrokerError(f"unsupported order type: {order_type}")
        if order_type == "limit":
            if limit_price is None or limit_price <= 0:
                raise BrokerError("limit price required for a limit order")
            if buying and price > limit_price:
                raise BrokerError(
                    f"limit {limit_price:.2f} is below the market {price:.2f}"
                )
            fill_price = round(min(price, limit_price), 2) if buying else round(max(price, limit_price), 2)
        else:
            # MARKET and a triggered STOP pay the market plus adverse slippage.
            slip = self.SLIPPAGE if buying else -self.SLIPPAGE
            fill_price = round(price * (1.0 + slip), 2)
        return Fill(order_id=f"PAPER-{uuid.uuid4().hex[:10].upper()}", price=fill_price, qty=qty)

    def buy(
        self,
        symbol,
        side,
        strike,
        expiry,
        qty,
        price_hint,
        order_type="market",
        limit_price=None,
    ) -> Fill:
        return self._fill(qty, price_hint, buying=True, order_type=order_type, limit_price=limit_price)

    def sell(
        self,
        symbol,
        side,
        strike,
        expiry,
        qty,
        price_hint,
        order_type="market",
        limit_price=None,
    ) -> Fill:
        return self._fill(qty, price_hint, buying=False, order_type=order_type, limit_price=limit_price)


class KiteBroker(Broker):
    """Kite Connect market orders (MIS) on NFO. Requires explicit opt-in."""

    name = "kite"
    label = "KITE LIVE"
    is_live = True
    SLIPPAGE = 0.0025

    def __init__(self, settings: Settings) -> None:
        if not (settings.kite_api_key and settings.kite_access_token):
            raise BrokerError("KITE_API_KEY / KITE_ACCESS_TOKEN missing")
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:
            raise BrokerError("kiteconnect package is not installed") from exc

        self.settings = settings
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        self.kite.set_access_token(settings.kite_access_token)
        self._instruments = None
        self._instrument_at = 0.0
        try:
            profile = self.kite.profile()
            self.user = profile.get("user_id", "connected")
        except Exception as exc:
            raise BrokerError(f"Kite authentication failed: {exc}") from exc

    def _load_instruments(self):
        import time

        if self._instruments is not None and time.time() - self._instrument_at < 6 * 3600:
            return self._instruments
        try:
            self._instruments = self.kite.instruments("NFO")
            self._instrument_at = time.time()
        except Exception as exc:
            raise BrokerError(f"Could not load NFO instruments: {exc}") from exc
        return self._instruments

    def _symbol_for(self, symbol: str, side: str, strike: float, expiry: str) -> str:
        target_expiry = datetime.strptime(expiry, "%d-%b-%Y").date()
        best = None
        for item in self._load_instruments():
            if (
                item.get("name") == symbol
                and item.get("instrument_type") == side
                and abs(float(item.get("strike") or 0) - strike) < 0.5
                and item.get("expiry") == target_expiry
            ):
                best = item.get("tradingsymbol")
                break
        if not best:
            raise BrokerError(f"No Kite instrument for {symbol} {strike} {side} {expiry}")
        return best

    def _order(self, symbol, side, strike, expiry, qty, transaction: str) -> str:
        tradingsymbol = self._symbol_for(symbol, side, strike, expiry)
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction,
                quantity=qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag="niftypulse",
            )
        except Exception as exc:
            raise BrokerError(f"Kite order rejected: {exc}") from exc
        return str(order_id)

    def buy(
        self,
        symbol,
        side,
        strike,
        expiry,
        qty,
        price_hint,
        order_type="market",
        limit_price=None,
    ) -> Fill:
        if order_type != "market":
            raise BrokerError("Kite adapter supports MARKET orders only")
        order_id = self._order(symbol, side, strike, expiry, qty, self.kite.TRANSACTION_TYPE_BUY)
        return Fill(order_id=order_id, price=round(price_hint * (1 + self.SLIPPAGE), 2), qty=qty)

    def sell(
        self,
        symbol,
        side,
        strike,
        expiry,
        qty,
        price_hint,
        order_type="market",
        limit_price=None,
    ) -> Fill:
        if order_type != "market":
            raise BrokerError("Kite adapter supports MARKET orders only")
        order_id = self._order(symbol, side, strike, expiry, qty, self.kite.TRANSACTION_TYPE_SELL)
        return Fill(order_id=order_id, price=round(price_hint * (1 - self.SLIPPAGE), 2), qty=qty)


def build_broker(settings: Settings) -> tuple[Broker, str]:
    """Returns (broker, note). Falls back to paper unless fully configured."""
    if settings.broker == "kite":
        if not settings.allow_live_trading:
            return PaperBroker(), "Live trading disabled (ENABLE_LIVE_TRADING=0); using paper fills"
        try:
            broker = KiteBroker(settings)
            return broker, f"Kite connected as {getattr(broker, 'user', 'user')}"
        except BrokerError as exc:
            return PaperBroker(), f"Kite unavailable: {exc}. Using paper fills"
    return PaperBroker(), "Paper execution with simulated fills"
