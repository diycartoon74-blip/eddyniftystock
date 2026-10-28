"""FastAPI app: serves the dashboard, REST state and WebSocket stream."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .chartdata import TIMEFRAMES, ChartService
from .config import SYMBOLS, WEB_DIR, Settings
from .engine import TradingEngine
from .wallet import WalletError

settings = Settings()
engine = TradingEngine(settings)
charts = ChartService(engine.data, settings)
INDEX = Path(WEB_DIR) / "index.html"

REFRESH_SECONDS = max(5, settings.poll_seconds)
PUSH_SECONDS = 3
TICK_SECONDS = 1.0
_wake = asyncio.Event()
_latest_tick: dict | None = None


async def _refresh_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(engine.refresh)
        except Exception as exc:  # defensive: the loop must survive
            engine.log("error", f"refresh loop: {type(exc).__name__}: {exc}")
        try:
            await asyncio.wait_for(_wake.wait(), timeout=REFRESH_SECONDS)
        except asyncio.TimeoutError:
            pass
        _wake.clear()


async def _tick_loop() -> None:
    """One-second tape for the chart's forming candle.

    Running this once in the app (instead of per browser tab) keeps the demo
    tape and all clients on the same print, and lets the REST fallback reuse it.
    """
    global _latest_tick
    while True:
        started = time.monotonic()
        try:
            _latest_tick = await asyncio.to_thread(engine.ticks)
        except Exception as exc:
            engine.log("error", f"tick loop: {type(exc).__name__}: {exc}")
        await asyncio.sleep(max(0.2, TICK_SECONDS - (time.monotonic() - started)))


@asynccontextmanager
async def lifespan(_: FastAPI):
    tasks = [asyncio.create_task(_refresh_loop()), asyncio.create_task(_tick_loop())]
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="NiftyPulse", version="1.0.0", lifespan=lifespan)


class SettingsUpdate(BaseModel):
    auto_trade: Optional[bool] = None
    mode: Optional[str] = None


class WalletTxn(BaseModel):
    amount: float
    note: Optional[str] = None


class OrderRequest(BaseModel):
    order_type: str = "market"
    price: Optional[float] = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX, headers={"Cache-Control": "no-store"})


@app.get("/api/state")
async def state() -> dict:
    return await asyncio.to_thread(engine.snapshot)


@app.get("/api/ticks")
async def ticks() -> dict:
    """Last one-second tape frame (also used when WebSockets are unavailable)."""
    if _latest_tick is None:
        return await asyncio.to_thread(engine.ticks)
    return _latest_tick


@app.get("/api/health")
async def health() -> dict:
    snapshot = await asyncio.to_thread(engine.snapshot)
    return {
        "status": "ok",
        "mode": snapshot["mode"],
        "broker": snapshot["execution"]["broker"],
        "last_refresh": snapshot["last_refresh"],
        "last_error": snapshot["last_error"],
        "symbols_ready": len(snapshot["symbols"]),
        "wallet_backend": snapshot["wallet"]["backend"],
        "auto_trade": snapshot["settings"]["auto_trade"],
        "halted": snapshot["risk"]["halted"],
    }


@app.get("/api/chart/{symbol}")
async def chart(symbol: str, tf: str = "5m") -> dict:
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"unknown symbol: {symbol}")
    if tf not in TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"unknown timeframe: {tf}")
    try:
        return await asyncio.to_thread(charts.get, symbol, tf)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"chart unavailable: {exc}")


@app.post("/api/settings")
async def update_settings(update: SettingsUpdate) -> dict:
    switched = False
    if update.auto_trade is not None:
        engine.set_auto(update.auto_trade)
    if update.mode is not None:
        if update.mode not in {"live", "demo"}:
            raise HTTPException(status_code=400, detail="mode must be live or demo")
        switched = (update.mode == "demo") != engine.settings.demo
        engine.set_mode(update.mode)
    if switched:
        _wake.set()  # fetch the new mode's candles immediately
    return await asyncio.to_thread(engine.snapshot)


@app.post("/api/resume")
async def resume() -> dict:
    engine.resume()
    return await asyncio.to_thread(engine.snapshot)


@app.post("/api/wallet/deposit")
async def wallet_deposit(txn: WalletTxn) -> dict:
    try:
        entry = await asyncio.to_thread(engine.deposit, txn.amount, txn.note or "")
    except (WalletError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"wallet": "deposit", "entry": entry}
    return snapshot


@app.post("/api/wallet/withdraw")
async def wallet_withdraw(txn: WalletTxn) -> dict:
    try:
        entry = await asyncio.to_thread(engine.withdraw, txn.amount, txn.note or "")
    except (WalletError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"wallet": "withdraw", "entry": entry}
    return snapshot


@app.post("/api/signals/{signal_id}/execute")
async def execute_signal(signal_id: str) -> dict:
    result = await asyncio.to_thread(engine.execute_manual, signal_id)
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"trade": result}
    return snapshot


@app.post("/api/signals/{signal_id}/order")
async def place_order(signal_id: str, req: OrderRequest) -> dict:
    result = await asyncio.to_thread(engine.place_order, signal_id, req.order_type, req.price)
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"order": result}
    return snapshot


@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str) -> dict:
    result = await asyncio.to_thread(engine.cancel_order, order_id)
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"order": result}
    return snapshot


@app.post("/api/refresh")
async def refresh_now() -> dict:
    await asyncio.to_thread(engine.refresh)
    return await asyncio.to_thread(engine.snapshot)


@app.post("/api/positions/{position_id}/close")
async def close_position(position_id: str) -> dict:
    closed = await asyncio.to_thread(engine.close_position, position_id)
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"closed": closed, "position_id": position_id}
    return snapshot


@app.post("/api/square-off")
async def square_off() -> dict:
    closed = await asyncio.to_thread(engine.square_off_all)
    snapshot = await asyncio.to_thread(engine.snapshot)
    snapshot["action"] = {"closed": closed}
    return snapshot


@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    sent_tick = None
    last_state = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last_state >= PUSH_SECONDS:
                payload = await asyncio.to_thread(engine.snapshot)
                payload["type"] = "state"
                await ws.send_json(payload)
                last_state = now
            elif _latest_tick is not None and _latest_tick is not sent_tick:
                await ws.send_json(_latest_tick)
                sent_tick = _latest_tick
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        return
    except Exception:
        with suppress(Exception):
            await ws.close()
