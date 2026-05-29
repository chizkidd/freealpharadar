"""FreeAlphaRadar: zero-cost alpha discovery engine.

A Streamlit application that ingests financial, alternative, and qualitative
data from entirely free, public, no-key data sources to surface a ranked,
transparently scored list of high-potential, under-the-radar companies.

The package is organised into a handful of focused sub-packages:

* :mod:`freealpharadar.fetchers` -- data ingestion (yfinance, SEC EDGAR,
  PatentsView, GDELT, manual CSV) with caching and offline fallbacks.
* :mod:`freealpharadar.scoring` -- the multi-factor "Moat + Momentum +
  Misvaluation" scoring engine.
* :mod:`freealpharadar.ml` -- zero-shot ML enhancements (FinBERT sentiment,
  PCA/K-Means clustering, optional XGBoost skeleton).
* :mod:`freealpharadar.ui` -- Streamlit dashboard views.
* :mod:`freealpharadar.utils` -- logging and shared helpers.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "FreeAlphaRadar contributors"

__all__ = ["__version__", "__author__"]
