"""Option chain acquisition and analytics.

NSE chains come from NSE's public option-chain endpoints and SENSEX chains from
BSE's derivatives endpoint. Both public endpoints aggressively rate limit /
block plain scripted traffic, so a deterministic synthetic chain
(Black-Scholes priced, marked ``source="synthetic"``) keeps the analysis layer
alive. The dashboard always shows which source is in use, and auto entries on
synthetic data are restricted to paper mode.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, time as dtime, timedelta

import requests

try:  # BSE's edge rejects ordinary TLS clients, but permits browser-like ones.
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - optional dependency
    curl_requests = None

from .config import IST, SYMBOLS
from .models import OptionChain, OptionRow

SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def black_scholes(
    spot: float,
    strike: float,
    years: float,
    iv: float,
    rate: float = 0.065,
    is_call: bool = True,
) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    years = max(years, 1.0 / 365.0)
    iv = max(iv, 0.01)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    if is_call:
        price = spot * norm_cdf(d1) - strike * math.exp(-rate * years) * norm_cdf(d2)
    else:
        price = strike * math.exp(-rate * years) * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return max(price, 0.05)


def next_expiry(now: datetime, monthly: bool = False, weekday: int = 1) -> tuple[str, float]:
    """Approximate NSE expiry (Tuesday) in dd-MMM-yyyy + years to expiry."""
    if monthly:
        year, month = now.year, now.month
        # Last Tuesday of the month, rolling forward when already past.
        for _ in range(2):
            if month == 12:
                first_next = datetime(year + 1, 1, 1, tzinfo=IST)
            else:
                first_next = datetime(year, month + 1, 1, tzinfo=IST)
            last_day = first_next - timedelta(days=1)
            offset = (last_day.weekday() - weekday) % 7
            candidate = (last_day - timedelta(days=offset)).replace(
                hour=15, minute=30, tzinfo=IST
            )
            if candidate > now:
                break
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    else:
        days_ahead = (weekday - now.weekday()) % 7
        candidate = (now + timedelta(days=days_ahead)).replace(
            hour=15, minute=30, second=0, microsecond=0, tzinfo=IST
        )
        if candidate <= now:
            candidate += timedelta(days=7)
    years = max((candidate - now).total_seconds() / (365.0 * 86400.0), 1.0 / 365.0)
    return candidate.strftime("%d-%b-%Y"), years


def _analytics(chain: OptionChain) -> OptionChain:
    rows = chain.rows
    if not rows:
        return chain

    call_oi = sum(r.ce_oi for r in rows)
    put_oi = sum(r.pe_oi for r in rows)
    chain.pcr = (put_oi / call_oi) if call_oi else 0.0

    # Max pain: expiry level that minimises total option-writer payout.
    strikes = [r.strike for r in rows]
    best_strike, best_pain = chain.spot, None
    for candidate in strikes:
        pain = 0.0
        for row in rows:
            pain += row.ce_oi * max(0.0, candidate - row.strike)
            pain += row.pe_oi * max(0.0, row.strike - candidate)
        if best_pain is None or pain < best_pain:
            best_pain, best_strike = pain, candidate
    chain.max_pain = float(best_strike)

    below = [r for r in rows if r.strike <= chain.spot]
    above = [r for r in rows if r.strike >= chain.spot]
    if below:
        chain.support = float(max(below, key=lambda r: r.pe_oi).strike)
    if above:
        chain.resistance = float(max(above, key=lambda r: r.ce_oi).strike)

    atm = chain.atm_strike()
    atm_row = chain.row_for(atm)
    if atm_row:
        ivs = [iv for iv in (atm_row.ce_iv, atm_row.pe_iv) if iv and not math.isnan(iv)]
        chain.atm_iv = sum(ivs) / len(ivs) if ivs else 0.0

    chain.rows.sort(key=lambda r: r.strike)
    return chain


class NSEClient:
    BASE = "https://www.nseindia.com"
    UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.UA,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/option-chain",
                "Connection": "keep-alive",
            }
        )
        self._warmed_at = 0.0
        self._blocked_until = 0.0
        self._expiry_cache: dict[str, tuple[float, list[str]]] = {}

    def _warm(self) -> None:
        if time.time() - self._warmed_at < 240:
            return
        try:
            self.session.get(self.BASE, timeout=5)
            self.session.get(f"{self.BASE}/option-chain", timeout=5)
            self._warmed_at = time.time()
        except requests.RequestException:
            pass

    def fetch(self, symbol: str, spot: float) -> OptionChain:
        if time.time() < self._blocked_until:
            raise RuntimeError("NSE endpoint cooling down")

        self._warm()
        now = datetime.now(IST)
        today = now.date()
        try:
            cached = self._expiry_cache.get(symbol)
            if cached and time.time() - cached[0] < 600:
                expiries = cached[1]
            else:
                info = self.session.get(
                    f"{self.BASE}/api/option-chain-contract-info",
                    params={"symbol": symbol},
                    timeout=7,
                )
                info.raise_for_status()
                expiries = info.json().get("expiryDates") or []
                self._expiry_cache[symbol] = (time.time(), expiries)

            parsed_expiries = []
            for expiry in expiries:
                try:
                    parsed_expiries.append(
                        (datetime.strptime(expiry, "%d-%b-%Y").date(), expiry)
                    )
                except ValueError:
                    continue
            upcoming = sorted([e for e in parsed_expiries if e[0] >= today])
            expiry_label = upcoming[0][1] if upcoming else (expiries[0] if expiries else "")
            if not expiry_label:
                raise RuntimeError("NSE returned no option expiries")

            resp = self.session.get(
                f"{self.BASE}/api/option-chain-v3",
                params={"type": "Indices", "symbol": symbol, "expiry": expiry_label},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            self._blocked_until = time.time() + 120
            raise RuntimeError(f"NSE option chain unavailable: {exc}") from exc

        records = payload.get("records") or {}
        data = records.get("data") or []
        if not data:
            self._blocked_until = time.time() + 120
            raise RuntimeError("NSE returned an empty option chain")

        by_strike: dict[float, OptionRow] = {}
        for item in data:
            if expiry_label and item.get("expiryDates") not in (None, expiry_label):
                continue
            strike = float(item.get("strikePrice") or 0)
            if strike <= 0:
                continue
            row = by_strike.setdefault(strike, OptionRow(strike=strike))
            ce = item.get("CE") or {}
            pe = item.get("PE") or {}
            if ce:
                row.ce_oi = int(ce.get("openInterest") or 0)
                row.ce_oi_change = int(ce.get("changeinOpenInterest") or 0)
                row.ce_iv = float(ce.get("impliedVolatility") or 0.0)
                row.ce_ltp = float(ce.get("lastPrice") or 0.0)
            if pe:
                row.pe_oi = int(pe.get("openInterest") or 0)
                row.pe_oi_change = int(pe.get("changeinOpenInterest") or 0)
                row.pe_iv = float(pe.get("impliedVolatility") or 0.0)
                row.pe_ltp = float(pe.get("lastPrice") or 0.0)

        rows = [r for r in by_strike.values() if r.ce_oi or r.pe_oi]
        if not rows:
            raise RuntimeError("NSE option chain had no usable strikes")

        chain = OptionChain(
            symbol=symbol,
            spot=spot,
            source="nse",
            expiry=expiry_label,
            rows=rows,
            updated_at=now.isoformat(timespec="seconds"),
        )
        return _analytics(chain)


class BSEClient:
    """SENSEX option chain from BSE's derivatives endpoint.

    BSE fronts the API with Akamai, which rejects ordinary ``requests`` TLS
    fingerprints. ``curl_cffi`` impersonates a browser and is used when it is
    installed; otherwise the caller falls back to the synthetic chain.
    """

    BASE = "https://api.bseindia.com/BseIndiaAPI/api"
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }

    def __init__(self) -> None:
        self.session = curl_requests.Session() if curl_requests is not None else None
        self._expiry_cache: tuple[float, list[str]] | None = None
        self._blocked_until = 0.0

    @staticmethod
    def _number(value) -> float:
        try:
            return float(str(value).replace(",", "").strip() or 0)
        except (TypeError, ValueError):
            return 0.0

    def _get(self, path: str, params: dict, timeout: float) -> dict:
        if self.session is None:
            raise RuntimeError("curl_cffi is not installed")
        resp = self.session.get(
            f"{self.BASE}/{path}",
            params=params,
            timeout=timeout,
            impersonate="chrome",
            headers=self.HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    def _expiries(self) -> list[str]:
        if self._expiry_cache and time.time() - self._expiry_cache[0] < 600:
            return self._expiry_cache[1]
        payload = self._get(
            "ddlExpiry_IV/w", {"ProductType": "IO", "scrip_cd": 1}, 8
        )
        today = datetime.now(IST).date()
        upcoming: list[tuple[datetime, str]] = []
        for item in payload.get("Table1") or []:
            label = str(item.get("ExpiryDate") or "")
            try:
                expiry = datetime.strptime(label, "%d %b %Y")
            except ValueError:
                continue
            if expiry.date() >= today:
                upcoming.append((expiry, label))
        upcoming.sort()
        labels = [label for _, label in upcoming]
        self._expiry_cache = (time.time(), labels)
        return labels

    def fetch(self, symbol: str, spot: float) -> OptionChain:
        if symbol != "SENSEX":
            raise RuntimeError(f"BSE client does not support {symbol}")
        if time.time() < self._blocked_until:
            raise RuntimeError("BSE endpoint cooling down")

        now = datetime.now(IST)
        try:
            expiries = self._expiries()
        except Exception as exc:
            self._blocked_until = time.time() + 300
            raise RuntimeError(f"BSE expiries unavailable: {exc}") from exc

        for label in expiries[:2]:
            expiry_date = datetime.strptime(label, "%d %b %Y").date()
            try:
                payload = self._get(
                    "DerivOptionChain_IV/w",
                    {"Expiry": label, "scrip_cd": 1, "strprice": 0},
                    10,
                )
            except Exception as exc:
                self._blocked_until = time.time() + 300
                raise RuntimeError(f"BSE option chain unavailable: {exc}") from exc

            table = payload.get("Table")
            if not isinstance(table, list) or not table:
                # BSE occasionally returns an empty table on the first call.
                time.sleep(0.4)
                payload = self._get(
                    "DerivOptionChain_IV/w",
                    {"Expiry": label, "scrip_cd": 1, "strprice": 0},
                    10,
                )
                table = payload.get("Table")
            if not isinstance(table, list) or not table:
                # An expired-today contract can be empty; any other empty
                # response is treated as a transient BSE gap, not as a cue to
                # silently show the next expiry.
                if expiry_date == now.date():
                    continue
                break

            rows: list[OptionRow] = []
            for item in table:
                strike = self._number(item.get("Strike_Price"))
                if strike <= 0:
                    continue
                ce_oi = int(self._number(item.get("C_Open_Interest")))
                pe_oi = int(self._number(item.get("Open_Interest")))
                ce_ltp = self._number(item.get("C_Last_Trd_Price"))
                pe_ltp = self._number(item.get("Last_Trd_Price"))
                if not (ce_oi or pe_oi or ce_ltp or pe_ltp):
                    continue
                rows.append(
                    OptionRow(
                        strike=strike,
                        ce_ltp=ce_ltp,
                        ce_oi=ce_oi,
                        ce_oi_change=int(self._number(item.get("C_Absolute_Change_OI"))),
                        ce_iv=self._number(item.get("C_IV")),
                        pe_ltp=pe_ltp,
                        pe_oi=pe_oi,
                        pe_oi_change=int(self._number(item.get("Absolute_Change_OI"))),
                        pe_iv=self._number(item.get("IV")),
                    )
                )

            if not rows:
                continue

            chain = OptionChain(
                symbol=symbol,
                spot=spot if spot > 0 else self._number(table[0].get("UlaValue")),
                source="bse",
                expiry=datetime.strptime(label, "%d %b %Y").strftime("%d-%b-%Y"),
                rows=rows,
                updated_at=now.isoformat(timespec="seconds"),
            )
            self._blocked_until = 0.0
            return _analytics(chain)

        raise RuntimeError("BSE returned an empty option chain")


def synthetic_chain(symbol: str, spot: float, now: datetime | None = None) -> OptionChain:
    """Deterministic BS-priced chain used when NSE is unreachable or in demo."""
    now = now or datetime.now(IST)
    cfg = SYMBOLS[symbol]
    step = cfg.step
    atm = round(spot / step) * step
    expiry, years = next_expiry(
        now, monthly=cfg.monthly_expiry, weekday=cfg.expiry_weekday
    )
    base_oi = cfg.chain_oi
    base_iv = cfg.chain_iv
    rows: list[OptionRow] = []

    for offset in range(-12, 13):
        strike = atm + offset * step
        if strike <= 0:
            continue
        moneyness = (strike - spot) / spot
        iv_skew = base_iv + 1.6 * abs(moneyness) ** 1.15 + (0.22 * max(0.0, -moneyness))
        wave = 1.0 + 0.16 * math.sin(strike / step * 0.55)
        call_profile = math.exp(-(((strike - (spot + 2.2 * step)) / (5.5 * step)) ** 2))
        put_profile = math.exp(-(((strike - (spot - 2.2 * step)) / (5.5 * step)) ** 2))
        ce_oi = int(round(base_oi * call_profile * wave / 500.0) * 500)
        pe_oi = int(round(base_oi * put_profile * (2.0 - wave) / 500.0) * 500)
        ce_ltp = round(black_scholes(spot, strike, years, iv_skew, is_call=True), 2)
        pe_ltp = round(black_scholes(spot, strike, years, iv_skew, is_call=False), 2)
        rows.append(
            OptionRow(
                strike=float(strike),
                ce_ltp=max(ce_ltp, 0.5),
                ce_oi=ce_oi,
                ce_oi_change=int(ce_oi * 0.04 * math.sin(strike)),
                ce_iv=round(iv_skew * 100, 2),
                pe_ltp=max(pe_ltp, 0.5),
                pe_oi=pe_oi,
                pe_oi_change=int(-pe_oi * 0.035 * math.sin(strike)),
                pe_iv=round(iv_skew * 100, 2),
            )
        )

    chain = OptionChain(
        symbol=symbol,
        spot=spot,
        source="synthetic",
        expiry=expiry,
        rows=rows,
        updated_at=now.isoformat(timespec="seconds"),
    )
    return _analytics(chain)
