"""Probe every free data-source endpoint and report whether it works.

Run on a networked machine (Codespaces / local) to confirm the live sources are
reachable and returning data before relying on them:

    python scripts/check_sources.py

It checks yfinance (quote), yfinance news, SEC EDGAR (ticker map + company
facts), and the optional patent providers (PatentsView and Lens) when their
tokens are set. No keys are required for the key-less sources; patents are
reported as SKIP when no provider token is configured. Exit code is non-zero if
any *required* (key-less) source fails, so it doubles as a smoke test.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Dict, Tuple

import requests

from freealpharadar.config import (
    LENS_API_TOKEN,
    LENS_ENDPOINT,
    PATENTSVIEW_API_KEY,
    PATENTSVIEW_ENDPOINT,
    SEC_COMPANY_FACTS,
    SEC_TICKER_MAP,
    SEC_USER_AGENT,
)

TIMEOUT = 30
_SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}


def _ok(detail: str) -> Tuple[str, str]:
    return "OK", detail


def _fail(detail: str) -> Tuple[str, str]:
    return "FAIL", detail


def check_yfinance() -> Tuple[str, str]:
    try:
        import yfinance as yf

        cap = (yf.Ticker("AAPL").get_info() or {}).get("marketCap")
        if cap:
            return _ok(f"AAPL market cap ${cap / 1e12:.2f}T")
        return _fail("yfinance returned no marketCap (Yahoo may be blocking)")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def check_yfinance_news() -> Tuple[str, str]:
    try:
        import yfinance as yf

        items = yf.Ticker("AAPL").news or []
        return _ok(f"{len(items)} news items for AAPL")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def check_sec() -> Tuple[str, str]:
    try:
        r = requests.get(SEC_TICKER_MAP, headers=_SEC_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        rows = r.json()
        cik = "0000320193"  # Apple, zero-padded to 10 digits.
        f = requests.get(
            SEC_COMPANY_FACTS.format(cik=cik), headers=_SEC_HEADERS, timeout=TIMEOUT
        )
        f.raise_for_status()
        concepts = f.json().get("facts", {}).get("us-gaap", {})
        return _ok(f"ticker map {len(rows)} rows; AAPL facts {len(concepts)} concepts")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def check_patentsview() -> Tuple[str, str]:
    if not PATENTSVIEW_API_KEY:
        return "SKIP", "FAR_PATENTSVIEW_API_KEY not set"
    try:
        import json

        q = {"_text_phrase": {"assignees.assignee_organization": "Apple Inc."}}
        params = {
            "q": json.dumps(q),
            "f": json.dumps(["patent_id", "patent_title", "patent_date"]),
            "o": json.dumps({"size": 5}),
        }
        r = requests.get(
            PATENTSVIEW_ENDPOINT,
            params=params,
            headers={"X-Api-Key": PATENTSVIEW_API_KEY},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        pats = r.json().get("patents", []) or []
        return _ok(f"{len(pats)} patents returned")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def check_lens() -> Tuple[str, str]:
    if not LENS_API_TOKEN:
        return "SKIP", "FAR_LENS_API_TOKEN not set"
    try:
        body = {
            "query": {"match_phrase": {"applicant.name": "Apple Inc."}},
            "size": 5,
            "include": ["biblio.invention_title", "date_published"],
        }
        r = requests.post(
            LENS_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Bearer {LENS_API_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=TIMEOUT,
        )
        if r.status_code in (401, 403):
            return _fail(f"{r.status_code} auth error (check the token)")
        if r.status_code == 429:
            return _fail("429 rate-limited (free tier quota)")
        r.raise_for_status()
        data = r.json().get("data", []) or []
        return _ok(f"{len(data)} patents returned")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


# Key-less sources whose failure should fail the whole run.
REQUIRED = ("yfinance", "yfinance-news", "SEC EDGAR")

CHECKS: Dict[str, Callable[[], Tuple[str, str]]] = {
    "yfinance": check_yfinance,
    "yfinance-news": check_yfinance_news,
    "SEC EDGAR": check_sec,
    "PatentsView": check_patentsview,
    "Lens.org": check_lens,
}


def main() -> int:
    print("Probing FreeAlphaRadar data sources…\n")
    failed_required = False
    for name, fn in CHECKS.items():
        t0 = time.monotonic()
        status, detail = fn()
        dt = time.monotonic() - t0
        icon = {"OK": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "?")
        print(f"{icon} {name:<14} {status:<4} ({dt:4.1f}s)  {detail}")
        if status == "FAIL" and name in REQUIRED:
            failed_required = True
    print()
    if failed_required:
        print("One or more REQUIRED (key-less) sources failed.")
        return 1
    print("All required sources OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
