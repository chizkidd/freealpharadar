"""Streamlit dashboard views.

Each module exposes a single ``render_*`` function that takes already-computed
data and draws one view. Keeping rendering separate from data orchestration
makes the views easy to reason about and keeps :mod:`streamlit_app` thin.
"""

from __future__ import annotations

from freealpharadar.ui.deep_dive import render_deep_dive
from freealpharadar.ui.radar_screen import render_radar_screen
from freealpharadar.ui.sidebar import render_sidebar
from freealpharadar.ui.watchlist import render_watchlist

__all__ = [
    "render_radar_screen",
    "render_deep_dive",
    "render_watchlist",
    "render_sidebar",
]
