"""Bulk SEC fundamentals warehouse (optional, offline).

Downloads SEC's free, key-less **Financial Statement Data Sets** (quarterly
ZIPs, 2009->present, every XBRL filer including delisted companies), loads the
handful of fundamentals FreeAlphaRadar scores on into a partitioned Parquet
store, and exposes a small DuckDB-backed query API.

This package is a research/screening **sidecar**: it is never imported by the
Streamlit app, its data store lives under a gitignored ``data/warehouse/``, and
its dependencies (``duckdb``, ``pyarrow``) are opt-in via
``requirements-warehouse.txt``.
"""

from __future__ import annotations

from freealpharadar.warehouse.loader import build_warehouse, load_quarter
from freealpharadar.warehouse.store import WarehouseStore

__all__ = ["build_warehouse", "load_quarter", "WarehouseStore"]
