"""yfinance-based market data and fundamentals fetcher.

Pulls historical prices, the three financial statements, and key metrics
(short interest, institutional ownership) for a ticker. yfinance is a
synchronous library, so calls are dispatched to a thread executor to keep the
fetcher interface asynchronous.

All payloads are normalised to plain JSON-serialisable structures so they can
be cached in SQLite and re-hydrated into pandas DataFrames on demand.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pandas as pd

from freealpharadar.config import settings
from freealpharadar.fetchers.base import BaseFetcher
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


def _df_to_records(df: Any) -> List[Dict[str, Any]]:
    """Convert a (possibly empty/None) DataFrame to JSON-friendly records.

    Index values become an ``index`` column so the orientation is preserved
    through JSON serialisation.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return []
    out = df.copy()
    # Statements come transposed (rows = line items, cols = period). Transpose
    # so each record is one reporting period.
    out = out.transpose()
    out.index = [str(i) for i in out.index]
    out = out.reset_index().rename(columns={"index": "period"})
    out.columns = [str(c) for c in out.columns]
    records: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        rec: Dict[str, Any] = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (pd.Timestamp,)):
                rec[col] = val.isoformat()
            else:
                try:
                    rec[col] = float(val)
                except (TypeError, ValueError):
                    rec[col] = str(val)
        records.append(rec)
    return records


class YFinanceFetcher(BaseFetcher):
    """Fetch prices, fundamentals and key metrics via yfinance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(source="yfinance", ttl=settings.ttl.fundamentals, **kwargs)

    async def _fetch_remote(self, key: str, **kwargs: Any) -> Dict[str, Any]:
        """Fetch a full data bundle for a ticker in a worker thread."""
        return await asyncio.to_thread(self._fetch_sync, key)

    def _fetch_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous yfinance pull, executed off the event loop."""
        import yfinance as yf  # imported lazily so the package imports offline

        # A browser-impersonating curl_cffi session greatly improves yfinance's
        # success rate from datacenter/CI IPs (Yahoo often blocks plain
        # requests there). Falls back to yfinance's default session if absent.
        tk = yf.Ticker(ticker, session=_browser_session())

        info: Dict[str, Any] = {}
        try:
            info = dict(tk.get_info())
        except Exception as exc:  # noqa: BLE001
            logger.debug("info() failed for %s: %s", ticker, exc)

        history_records: List[Dict[str, Any]] = []
        try:
            # Full available price history (monthly) so older names get their
            # complete record; recent IPOs simply return what exists.
            hist = tk.history(period="max", interval="1mo", auto_adjust=True)
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                for _, row in hist.iterrows():
                    date_val = row.get("Date") or row.get("Datetime")
                    history_records.append(
                        {
                            "date": str(date_val),
                            "close": _safe_float(row.get("Close")),
                            "volume": _safe_float(row.get("Volume")),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("history() failed for %s: %s", ticker, exc)

        # Free, key-less fallback when Yahoo returns nothing (common in CI):
        # Stooq monthly closes. Restores price history + a last close so the
        # price-momentum factor and a market-cap approximation still work.
        if not history_records:
            history_records = _stooq_history(ticker)

        payload: Dict[str, Any] = {
            "ticker": ticker.upper(),
            "info": _clean_info(info),
            "history": history_records,
            "income_statement": _safe_records(tk, "income_stmt"),
            "balance_sheet": _safe_records(tk, "balance_sheet"),
            "cash_flow": _safe_records(tk, "cashflow"),
            "key_metrics": {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "held_percent_institutions": info.get("heldPercentInstitutions"),
                "held_percent_insiders": info.get("heldPercentInsiders"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "trailing_pe": info.get("trailingPE"),
                "price_to_book": info.get("priceToBook"),
                "gross_margins": info.get("grossMargins"),
                "profit_margins": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
            },
        }
        return payload


def _browser_session() -> Any:
    """Return a Chrome-impersonating ``curl_cffi`` session, or ``None``.

    Passing this to ``yf.Ticker`` markedly improves success from datacenter/CI
    IPs. Returns ``None`` (yfinance's default session) when ``curl_cffi`` isn't
    installed, so behaviour is unchanged in that case.
    """
    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(impersonate="chrome")
    except Exception:  # noqa: BLE001
        return None


def _stooq_history(ticker: str) -> List[Dict[str, Any]]:
    """Fetch monthly close history from Stooq (free, key-less) as a fallback.

    Stooq serves US equities as ``<ticker>.us``. Returns ``[]`` on any failure
    so the caller simply ends up with no price history.
    """
    import csv
    import io
    import urllib.request

    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=m"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stooq fallback failed for %s: %s", ticker, exc)
        return []

    records: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(body))
    for row in reader:
        close = _safe_float(row.get("Close"))
        if close is not None:
            records.append(
                {
                    "date": row.get("Date", ""),
                    "close": close,
                    "volume": _safe_float(row.get("Volume")),
                }
            )
    if records:
        logger.info("Stooq supplied %d monthly closes for %s", len(records), ticker)
    return records


def _safe_records(tk: Any, attr: str) -> List[Dict[str, Any]]:
    """Return JSON records for a yfinance statement attribute, never raising."""
    try:
        return _df_to_records(getattr(tk, attr))
    except Exception as exc:  # noqa: BLE001
        logger.debug("statement %s failed: %s", attr, exc)
        return []


def _safe_float(value: Any) -> Any:
    """Coerce to float, returning ``None`` on failure or NaN."""
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _clean_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a compact, JSON-safe subset of yfinance's verbose ``info`` dict."""
    keep = {
        "longName",
        "shortName",
        "symbol",
        "sector",
        "industry",
        "longBusinessSummary",
        "country",
        "fullTimeEmployees",
        "marketCap",
        "enterpriseValue",
        "totalRevenue",
        "grossProfits",
        "ebitda",
        "freeCashflow",
        "website",
    }
    cleaned: Dict[str, Any] = {}
    for k in keep:
        v = info.get(k)
        if isinstance(v, float) and pd.isna(v):
            v = None
        cleaned[k] = v
    return cleaned
