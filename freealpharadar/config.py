"""Application configuration for FreeAlphaRadar.

Configuration is intentionally code-first and requires **zero environment
setup**: there is no ``.env`` file and no secrets. Every value here has a sane
default. The few knobs that *can* be tuned are exposed as environment variables
purely for convenience (e.g. running in Colab or CI) and all have safe
fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
WATCHLIST_CHANGES_DIR: Path = PROJECT_ROOT / "watchlist_changes"

DEFAULT_DB_PATH: Path = DATA_DIR / "freealpharadar.sqlite"

# Editable default screening universe. ``universe.txt`` at the repo root is the
# source of truth so the list can be changed without code edits; the hard-coded
# fallback below guarantees the app still works if the file is missing.
UNIVERSE_FILE: Path = PROJECT_ROOT / "universe.txt"

_FALLBACK_UNIVERSE: List[str] = [
    "PLTR",
    "BE",
    "SNDK",
    "IONQ",
    "RKLB",
    "OKLO",
    "SMR",
    "ASTS",
    "TEM",
    "RXRX",
    "PATH",
    "CRWD",
]


def _load_default_universe() -> List[str]:
    """Load the default universe from ``universe.txt``.

    The file allows one-or-more whitespace/comma-separated tickers per line,
    blank lines, full-line ``#`` comments and inline ``# ...`` comments. Order
    is preserved and duplicates removed. Falls back to :data:`_FALLBACK_UNIVERSE`
    when the file is absent or yields nothing.
    """
    try:
        raw = UNIVERSE_FILE.read_text(encoding="utf-8")
    except OSError:
        return list(_FALLBACK_UNIVERSE)

    tickers: List[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = line.split("#", 1)[0]  # drop comments (full-line and inline)
        for token in line.replace(",", " ").split():
            sym = token.strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                tickers.append(sym)
    return tickers or list(_FALLBACK_UNIVERSE)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Cache TTLs (in seconds). Each data source can be tuned independently so that
# we respect free-tier rate limits while keeping the dashboard responsive.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CacheTTL:
    """Time-to-live (seconds) for each cached data source."""

    prices: int = _env_int("FAR_TTL_PRICES", 60 * 60 * 6)  # 6 hours
    fundamentals: int = _env_int("FAR_TTL_FUNDAMENTALS", 60 * 60 * 24)  # 1 day
    sec: int = _env_int("FAR_TTL_SEC", 60 * 60 * 24 * 7)  # 1 week
    patents: int = _env_int("FAR_TTL_PATENTS", 60 * 60 * 24 * 7)  # 1 week
    news: int = _env_int("FAR_TTL_NEWS", 60 * 60 * 12)  # 12 hours


# --------------------------------------------------------------------------- #
# External endpoints. All free, all key-less.
# --------------------------------------------------------------------------- #
PATENTSVIEW_ENDPOINT = "https://search.patentsview.org/api/v1/patent/"
PATENTSVIEW_LEGACY_ENDPOINT = "https://api.patentsview.org/patents/query"
# PatentsView's current Search API requires a *free* API key (request one at
# https://patentsview.org/apis/keyrequest). It's optional: without it the app
# simply shows no patent data — the zero-config promise is preserved.
PATENTSVIEW_API_KEY = os.environ.get("FAR_PATENTSVIEW_API_KEY", "")

# Lens.org is an alternative, global patent provider. The patent fetcher is
# provider-agnostic: it uses PatentsView when its key is set, otherwise Lens
# when a Lens token is set, otherwise it skips patents entirely (zero-config).
LENS_ENDPOINT = "https://api.lens.org/patent/search"
LENS_API_TOKEN = os.environ.get("FAR_LENS_API_TOKEN", "")

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC requires a descriptive User-Agent. No key, but identification is polite.
SEC_USER_AGENT = os.environ.get(
    "FAR_SEC_USER_AGENT",
    "FreeAlphaRadar research tool (contact: freealpharadar@example.com)",
)


@dataclass
class Settings:
    """Top-level application settings.

    Attributes:
        db_path: Location of the SQLite cache/database file.
        ttl: Per-source cache time-to-live configuration.
        offline: When ``True`` no network calls are attempted; the app serves
            cached/sample data only. Auto-detected at runtime but can be forced
            via the ``FAR_OFFLINE`` environment variable.
        http_timeout: Per-request timeout in seconds for HTTP fetchers.
        max_retries: Number of retry attempts for transient network failures.
        request_concurrency: Max simultaneous in-flight HTTP requests.
        default_universe: Tickers loaded when the user has not supplied any.
        finbert_model: HuggingFace model id used for sentiment analysis.
    """

    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FAR_DB_PATH", str(DEFAULT_DB_PATH))
        )
    )
    ttl: CacheTTL = field(default_factory=CacheTTL)
    offline: bool = _env_bool("FAR_OFFLINE", False)
    http_timeout: int = _env_int("FAR_HTTP_TIMEOUT", 30)
    max_retries: int = _env_int("FAR_MAX_RETRIES", 4)
    request_concurrency: int = _env_int("FAR_CONCURRENCY", 5)
    finbert_model: str = os.environ.get("FAR_FINBERT_MODEL", "ProsusAI/finbert")
    default_universe: List[str] = field(default_factory=_load_default_universe)

    def ensure_dirs(self) -> None:
        """Create all directories the application writes to."""
        for directory in (DATA_DIR, SAMPLE_DATA_DIR, WATCHLIST_CHANGES_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# A module-level singleton is convenient for Streamlit's execution model.
settings = Settings()
settings.ensure_dirs()
