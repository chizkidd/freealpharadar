"""Market-wide discovery: scan all SEC filers for under-the-radar names.

A two-stage, **offline** funnel (never runs inside the Streamlit app):

1. :mod:`freealpharadar.discovery.screen` -- a cheap fundamentals-only screen
   over the *entire* bulk warehouse (~8,000 filers) producing a shortlist.
2. :mod:`freealpharadar.discovery.discover` -- runs the existing full 35-factor
   pipeline on the shortlist, ranks it, and promotes the top-N into the app's
   ``universe.txt`` + prewarm snapshot, plus a dated report.
"""

from __future__ import annotations

from freealpharadar.discovery.discover import DiscoveryResult, run_discovery
from freealpharadar.discovery.screen import ScreenConfig, screen_candidates

__all__ = ["ScreenConfig", "screen_candidates", "run_discovery", "DiscoveryResult"]
