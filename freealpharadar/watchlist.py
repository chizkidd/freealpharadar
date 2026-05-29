"""Watchlist management and change detection.

Watchlist membership lives in SQLite (see :mod:`freealpharadar.database`). This
module layers re-scoring and change detection on top: the "Check for Changes"
action re-scores watchlisted companies, diffs each against its last stored
score, writes a human-readable changelog to
``watchlist_changes/<ticker>_<date>.txt``, and returns the diffs for display.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from freealpharadar.config import WATCHLIST_CHANGES_DIR
from freealpharadar.database import get_db
from freealpharadar.service import run_pipeline
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


@dataclass
class WatchlistChange:
    """A detected change for a single watchlisted company.

    Attributes:
        ticker: Ticker symbol.
        previous_score: Last stored composite score, if any.
        new_score: Freshly computed composite score.
        delta: ``new_score - previous_score`` (``None`` on first run).
        changelog_path: Path to the written changelog file.
        details: Free-form human-readable lines describing the change.
    """

    ticker: str
    previous_score: Optional[float]
    new_score: float
    delta: Optional[float]
    changelog_path: str
    details: List[str] = field(default_factory=list)


def check_watchlist_changes(
    weights: Optional[Dict[str, float]] = None,
    manual_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    force_refresh: bool = True,
) -> List[WatchlistChange]:
    """Re-score every watchlisted company and report what changed.

    Args:
        weights: Factor-weight overrides to use when re-scoring.
        manual_signals: Optional manual CSV signals.
        force_refresh: Whether to bypass caches when re-fetching (default
            ``True`` so "check for changes" really refreshes).

    Returns:
        A list of :class:`WatchlistChange`, one per watchlisted ticker.
    """
    db = get_db()
    tickers = db.get_watchlist()
    if not tickers:
        return []

    # Capture previous scores *before* re-scoring overwrites the latest row.
    previous: Dict[str, Optional[float]] = {}
    for tk in tickers:
        prev = db.latest_score(tk)
        previous[tk] = prev.get("score") if prev else None

    output = run_pipeline(
        tickers,
        weights=weights,
        manual_signals=manual_signals,
        force_refresh=force_refresh,
        persist=True,
    )
    results_by_ticker = {r.ticker: r for r in output.results}

    changes: List[WatchlistChange] = []
    today = _dt.date.today().isoformat()
    WATCHLIST_CHANGES_DIR.mkdir(parents=True, exist_ok=True)

    for tk in tickers:
        res = results_by_ticker.get(tk)
        if res is None:
            continue
        prev_score = previous.get(tk)
        delta = None if prev_score is None else round(res.score - prev_score, 2)

        details = _build_details(tk, prev_score, res.score, delta, res)
        path = _write_changelog(tk, today, details)
        changes.append(
            WatchlistChange(
                ticker=tk,
                previous_score=prev_score,
                new_score=res.score,
                delta=delta,
                changelog_path=str(path),
                details=details,
            )
        )

    return changes


def _build_details(
    ticker: str,
    prev_score: Optional[float],
    new_score: float,
    delta: Optional[float],
    result: Any,
) -> List[str]:
    """Compose the human-readable changelog lines for one company."""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"FreeAlphaRadar watchlist changelog -- {ticker}",
        f"Generated: {now}",
        "",
    ]
    if prev_score is None:
        lines.append(f"First recorded score: {new_score:.2f} (no prior baseline).")
    else:
        direction = "↑" if (delta or 0) > 0 else ("↓" if (delta or 0) < 0 else "→")
        lines.append(
            f"Score: {prev_score:.2f} -> {new_score:.2f} " f"({direction} {delta:+.2f})"
        )
    lines.append("")
    lines.append("Top factor contributions:")
    for contrib in result.contributions[:8]:
        raw = "n/a" if contrib.raw is None else f"{contrib.raw:.4g}"
        lines.append(
            f"  - {contrib.label}: raw={raw}, z={contrib.zscore:+.2f}, "
            f"contribution={contrib.contribution:+.3f}"
        )
    return lines


def _write_changelog(ticker: str, date: str, lines: List[str]) -> Path:
    """Write changelog lines to ``watchlist_changes/<ticker>_<date>.txt``."""
    path = WATCHLIST_CHANGES_DIR / f"{ticker}_{date}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote watchlist changelog: %s", path)
    return path
