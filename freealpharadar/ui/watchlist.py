"""The Watchlist view.

Lets the user add/remove tickers from the SQLite-backed watchlist and run
"Check for Changes", which re-scores watchlisted companies, writes a changelog
file per company, and displays the score deltas inline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from freealpharadar.database import get_db
from freealpharadar.scoring import ScoreResult
from freealpharadar.watchlist import check_watchlist_changes


def render_watchlist(
    results: List[ScoreResult],
    weights: Optional[Dict[str, float]] = None,
    manual_signals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Render watchlist management and change-checking.

    Args:
        results: Current scored universe (used to populate the add dropdown).
        weights: Factor weights to reuse when re-scoring the watchlist.
        manual_signals: Manual CSV signals to reuse when re-scoring.
    """
    st.header("⭐ Watchlist")
    db = get_db()
    watchlist = db.get_watchlist()

    # ---- Add / remove ----------------------------------------------- #
    c1, c2 = st.columns(2)
    with c1:
        options = [r.ticker for r in results if r.ticker not in watchlist]
        if options:
            to_add = st.selectbox("Add a scored company", ["—"] + options)
            if to_add != "—" and st.button(f"➕ Add {to_add}"):
                db.add_to_watchlist(to_add)
                st.rerun()
        manual_add = st.text_input("…or add a ticker directly").strip().upper()
        if manual_add and st.button("➕ Add ticker"):
            db.add_to_watchlist(manual_add)
            st.rerun()
    with c2:
        if watchlist:
            to_remove = st.selectbox("Remove from watchlist", ["—"] + watchlist)
            if to_remove != "—" and st.button(f"🗑️ Remove {to_remove}"):
                db.remove_from_watchlist(to_remove)
                st.rerun()

    if not watchlist:
        st.info("Your watchlist is empty. Add companies above.")
        return

    st.subheader("Current watchlist")
    st.write(", ".join(watchlist))

    # ---- Check for changes ------------------------------------------ #
    if st.button("🔔 Check for Changes (re-score watchlist)", type="primary"):
        with st.spinner("Re-scoring watchlist…"):
            changes = check_watchlist_changes(
                weights=weights, manual_signals=manual_signals
            )
        if not changes:
            st.warning("Nothing to check.")
            return
        for ch in changes:
            delta_txt = "first run" if ch.delta is None else f"{ch.delta:+.2f}"
            with st.expander(
                f"{ch.ticker}: {ch.new_score:.1f} ({delta_txt})", expanded=True
            ):
                st.text("\n".join(ch.details))
                st.caption(f"Changelog written to: `{ch.changelog_path}`")
