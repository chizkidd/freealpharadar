"""Sidebar controls: universe, factor weights, filters and manual upload.

Renders the global controls and returns a small settings object the main app
uses to drive the pipeline and filter results. All widgets have sensible
defaults so the app is fully usable without touching anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

from freealpharadar.config import settings
from freealpharadar.fetchers.manual_csv import load_manual_csv
from freealpharadar.scoring.factors import FACTORS, FactorGroup


@dataclass
class SidebarState:
    """User selections gathered from the sidebar.

    Attributes:
        tickers: The universe to score.
        weights: Per-factor weights (0-3, default 1.0).
        manual_signals: Parsed manual CSV signals (possibly empty).
        force_refresh: Whether the next run should bypass caches.
        run_ml: Whether to run FinBERT/clustering enrichment.
        run_requested: True when the user clicked "Run / Refresh".
    """

    tickers: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    manual_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    force_refresh: bool = False
    run_ml: bool = True
    run_requested: bool = False


def render_sidebar() -> SidebarState:
    """Render sidebar widgets and return the resulting :class:`SidebarState`."""
    st.sidebar.title("📡 FreeAlphaRadar")
    st.sidebar.caption("Zero-cost alpha discovery. No API keys, no sign-ups, no fees.")

    if settings.offline:
        st.sidebar.info("🛜 Offline mode — serving cached/sample data only.")

    # ---- Universe ---------------------------------------------------- #
    st.sidebar.subheader("Universe")
    default_universe = ", ".join(settings.default_universe)
    raw = st.sidebar.text_area(
        "Tickers (comma or space separated)",
        value=default_universe,
        height=90,
        help="The set of companies to ingest and score.",
    )
    tickers = _parse_tickers(raw)

    # ---- Manual signals upload -------------------------------------- #
    st.sidebar.subheader("Manual signals (optional)")
    upload = st.sidebar.file_uploader(
        "Upload manual_upload_template.csv",
        type=["csv"],
        help="Optional employee/culture/product-moat signals. Ignored if absent.",
    )
    manual_signals = load_manual_csv(upload) if upload is not None else {}
    if manual_signals:
        st.sidebar.success(f"Loaded manual signals for {len(manual_signals)} tickers.")

    # ---- Run controls ----------------------------------------------- #
    st.sidebar.subheader("Run")
    run_ml = st.sidebar.checkbox(
        "Run AI enrichment (FinBERT + clustering)",
        value=True,
        help="Disable for a faster, purely rule-based run.",
    )
    force_refresh = st.sidebar.checkbox(
        "Force refresh (ignore cache TTL)",
        value=False,
    )
    run_requested = st.sidebar.button("🔄 Refresh Data & Re-score", type="primary")

    # ---- Factor weights --------------------------------------------- #
    weights = _render_weight_sliders()

    return SidebarState(
        tickers=tickers,
        weights=weights,
        manual_signals=manual_signals,
        force_refresh=force_refresh,
        run_ml=run_ml,
        run_requested=run_requested,
    )


def _render_weight_sliders() -> Dict[str, float]:
    """Render grouped factor-weight sliders inside an expander."""
    weights: Dict[str, float] = {}
    with st.sidebar.expander("⚖️ Factor weights", expanded=False):
        st.caption("Default is equal weight (1.0). Set 0 to disable a factor.")
        for group in FactorGroup:
            st.markdown(f"**{group.value}**")
            for spec in FACTORS:
                if spec.group is not group:
                    continue
                weights[spec.name] = st.slider(
                    spec.label,
                    min_value=0.0,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key=f"w_{spec.name}",
                    help=spec.description,
                )
    return weights


def _parse_tickers(raw: str) -> List[str]:
    """Parse a free-form ticker string into a clean, de-duplicated list."""
    parts = [p.strip().upper() for p in raw.replace(",", " ").split()]
    seen: List[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return seen
