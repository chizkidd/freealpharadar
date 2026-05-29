"""Base fetcher with caching, retries and offline fallbacks.

Concrete fetchers subclass :class:`BaseFetcher` and implement
:meth:`BaseFetcher._fetch_remote`. The base class handles the cross-cutting
concerns shared by every data source: cache freshness checks, exponential
backoff retries, offline detection, and graceful fallback to cached data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp

from freealpharadar.config import settings
from freealpharadar.database import Database, get_db
from freealpharadar.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Outcome of a fetch operation.

    Attributes:
        source: Logical source name.
        key: Identifier within the source (e.g. ticker).
        payload: The fetched (or cached) data, or ``None`` on total failure.
        from_cache: ``True`` when the payload came from the cache rather than a
            fresh network call.
        stale: ``True`` when cached data was used because the live fetch failed.
        warning: Human-readable, non-fatal warning suitable for ``st.warning``.
        age: Age of the returned payload in seconds (0 for fresh fetches).
    """

    source: str
    key: str
    payload: Optional[Any] = None
    from_cache: bool = False
    stale: bool = False
    warning: Optional[str] = None
    age: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether any usable payload (fresh or cached) is available."""
        return self.payload is not None


class BaseFetcher:
    """Abstract base for all data fetchers.

    Args:
        source: Unique logical name for this source, used as the cache
            namespace (e.g. ``"patentsview"``).
        ttl: Cache time-to-live in seconds.
        db: Database handle. Defaults to the shared singleton.
    """

    def __init__(self, source: str, ttl: int, db: Optional[Database] = None) -> None:
        self.source = source
        self.ttl = ttl
        self.db = db or get_db()

    # ------------------------------------------------------------------ #
    # To be implemented by subclasses
    # ------------------------------------------------------------------ #
    async def _fetch_remote(self, key: str, **kwargs: Any) -> Any:
        """Perform the live network fetch for ``key``.

        Subclasses must override this. It should raise on failure; the base
        class is responsible for retries and fallback handling.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def fetch(
        self, key: str, force_refresh: bool = False, **kwargs: Any
    ) -> FetchResult:
        """Fetch data for ``key``, honouring cache, retries and fallbacks.

        The resolution order is:

        1. If a fresh cache entry exists and ``force_refresh`` is ``False``,
           return it.
        2. If offline, return the cached entry (even if stale) or an empty
           result with a warning.
        3. Otherwise attempt a live fetch with retries; on success cache and
           return it; on failure fall back to any cached entry.

        Args:
            key: Identifier to fetch (typically a ticker).
            force_refresh: Bypass the freshness check and re-fetch.
            **kwargs: Passed through to :meth:`_fetch_remote`.

        Returns:
            A :class:`FetchResult`.
        """
        key = key.strip()
        cached = self.db.get_cache(self.source, key)

        if not force_refresh and cached is not None and cached["age"] < self.ttl:
            logger.debug("[%s] cache hit (fresh) for %s", self.source, key)
            return FetchResult(
                source=self.source,
                key=key,
                payload=cached["payload"],
                from_cache=True,
                age=cached["age"],
            )

        if settings.offline:
            return self._offline_result(key, cached)

        try:
            payload = await self._fetch_with_retries(key, **kwargs)
            self.db.set_cache(self.source, key, payload)
            logger.info("[%s] fetched fresh data for %s", self.source, key)
            return FetchResult(source=self.source, key=key, payload=payload)
        except Exception as exc:  # noqa: BLE001 -- we want to catch everything
            logger.warning("[%s] live fetch failed for %s: %s", self.source, key, exc)
            return self._fallback_result(key, cached, exc)

    async def _fetch_with_retries(self, key: str, **kwargs: Any) -> Any:
        """Call :meth:`_fetch_remote` with exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, settings.max_retries + 1):
            try:
                return await self._fetch_remote(key, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                backoff = 2 ** (attempt - 1)
                logger.debug(
                    "[%s] attempt %d/%d for %s failed (%s); retrying in %ds",
                    self.source,
                    attempt,
                    settings.max_retries,
                    key,
                    exc,
                    backoff,
                )
                if attempt < settings.max_retries:
                    await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------ #
    # Fallback helpers
    # ------------------------------------------------------------------ #
    def _offline_result(
        self, key: str, cached: Optional[Dict[str, Any]]
    ) -> FetchResult:
        """Build a result when running in forced-offline mode."""
        if cached is not None:
            return FetchResult(
                source=self.source,
                key=key,
                payload=cached["payload"],
                from_cache=True,
                stale=cached["age"] >= self.ttl,
                age=cached["age"],
                warning=(f"Offline mode: serving cached {self.source} data for {key}."),
            )
        return FetchResult(
            source=self.source,
            key=key,
            payload=None,
            warning=(f"Offline mode and no cached {self.source} data for {key}."),
        )

    def _fallback_result(
        self, key: str, cached: Optional[Dict[str, Any]], exc: Exception
    ) -> FetchResult:
        """Build a result after a failed live fetch, falling back to cache."""
        if cached is not None:
            return FetchResult(
                source=self.source,
                key=key,
                payload=cached["payload"],
                from_cache=True,
                stale=True,
                age=cached["age"],
                warning=(
                    f"Could not refresh {self.source} for {key} ({exc}). "
                    f"Showing cached data ({cached['age'] / 3600:.1f}h old)."
                ),
            )
        return FetchResult(
            source=self.source,
            key=key,
            payload=None,
            warning=f"{self.source} unavailable for {key} and no cache exists ({exc}).",
        )

    # ------------------------------------------------------------------ #
    # Shared HTTP helper
    # ------------------------------------------------------------------ #
    async def _http_get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """GET ``url`` and return parsed JSON.

        Args:
            url: Target URL.
            params: Query string parameters.
            headers: Extra request headers.

        Returns:
            Parsed JSON payload.

        Raises:
            aiohttp.ClientError: On any HTTP-level failure.
        """
        timeout = aiohttp.ClientTimeout(total=settings.http_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    async def _http_get_text(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """GET ``url`` and return the response body as text."""
        timeout = aiohttp.ClientTimeout(total=settings.http_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.text()
