"""FreeAlphaRadar — Streamlit entrypoint.

Run locally with::

    streamlit run streamlit_app.py

The app is stateless and zero-config: it requires no API keys and no secrets.
On first launch it seeds a local SQLite cache with deterministic sample data so
the dashboard is fully populated and usable **even with no network access**.
Clicking "Refresh Data & Re-score" then pulls live data from the free public
sources (yfinance, SEC EDGAR, PatentsView, GDELT) where reachable, always
falling back to cache on failure.
"""

from __future__ import annotations

import streamlit as st

from freealpharadar.ml.finbert import FinBERTSentiment
from freealpharadar.sample_data import cache_is_empty, seed_cache, seed_from_snapshot
from freealpharadar.service import PipelineOutput, run_pipeline
from freealpharadar.ui import (
    render_deep_dive,
    render_radar_screen,
    render_sidebar,
    render_watchlist,
)
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="FreeAlphaRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _ensure_seeded() -> bool:
    """Warm the cache once per process when it is empty.

    Prefers the committed prewarm snapshot of real, scheduler-refreshed data so
    a freshly-booted hosted instance shows live figures immediately; falls back
    to the deterministic synthetic sample when no snapshot is present.
    """
    if cache_is_empty():
        # Use the committed prewarm snapshot only if it's reasonably populated;
        # a thin/failed snapshot falls back to the synthetic sample so the app
        # is never a wall of "n/a".
        if seed_from_snapshot(min_coverage=0.3) == 0:
            seed_cache()
    return True


@st.cache_resource(show_spinner=False)
def _get_analyzer() -> FinBERTSentiment:
    """Process-wide shared FinBERT analyser (lazy-loads its model)."""
    return FinBERTSentiment()


def _run_and_store(state) -> None:
    """Run the scoring pipeline and stash the output in session state."""
    progress = st.progress(0.0, text="Gathering data…")

    def _cb(done: int, total: int, ticker: str) -> None:
        progress.progress(
            done / max(total, 1), text=f"Processed {ticker} ({done}/{total})"
        )

    output = run_pipeline(
        state.tickers,
        weights=state.weights,
        manual_signals=state.manual_signals,
        force_refresh=state.force_refresh,
        run_ml=state.run_ml,
        progress_cb=_cb,
    )
    progress.empty()
    st.session_state["output"] = output
    st.session_state["companies_by_ticker"] = {c.ticker: c for c in output.companies}
    st.session_state["results_by_ticker"] = {r.ticker: r for r in output.results}


def main() -> None:
    """Render the full application."""
    _ensure_seeded()
    state = render_sidebar()

    # First load: auto-run against cached/sample data so the app is never blank.
    if "output" not in st.session_state or state.run_requested:
        with st.spinner("Scoring universe…"):
            _run_and_store(state)

    output: PipelineOutput = st.session_state["output"]

    if output.warnings:
        with st.expander(f"⚠️ {len(output.warnings)} data warning(s)", expanded=False):
            for w in output.warnings:
                st.warning(w)

    tab_radar, tab_deep, tab_watch = st.tabs(
        ["🛰️ Radar Screen", "🔬 Deep Dive", "⭐ Watchlist"]
    )

    with tab_radar:
        selected = render_radar_screen(output.results)
        if selected:
            st.session_state["selected_ticker"] = selected

    with tab_deep:
        ticker = st.session_state.get("selected_ticker")
        if ticker is None and output.results:
            ticker = output.results[0].ticker
        company = st.session_state.get("companies_by_ticker", {}).get(ticker)
        result = st.session_state.get("results_by_ticker", {}).get(ticker)
        render_deep_dive(company, result, analyzer=_get_analyzer())

    with tab_watch:
        render_watchlist(
            output.results,
            weights=state.weights,
            manual_signals=state.manual_signals,
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: yfinance · SEC EDGAR · PatentsView · GDELT — all free, no keys. "
        "Not investment advice."
    )


if __name__ == "__main__":
    main()
