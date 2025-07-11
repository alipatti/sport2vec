from typing import Any, Iterable, Optional, Callable
from dataclasses import KW_ONLY, dataclass, field
import time

from tqdm import tqdm
import requests
import requests_cache


type JSON = Any


def _rate_limit[T](
    iterable: Iterable[T],
    delay_ms: int | float = 600,
    limit_when: Callable[[T], bool] = lambda x: True,  # always throttle by default
):
    """Rate-limit an iterator"""
    next_throttled_item = time.monotonic()

    for item in iterable:
        yield item

        if not limit_when(item):
            continue

        sleep_time = next_throttled_item - time.monotonic()
        next_throttled_item = time.monotonic() + delay_ms / 1000

        if sleep_time > 0:
            time.sleep(sleep_time)


@dataclass
class API:
    base_url: str
    _ = KW_ONLY
    headers: dict = field(default_factory=lambda: {})
    session = requests_cache.CachedSession(
        ".http_cache",
    )

    def _request(
        self,
        endpoint: str,
        params: dict[Any, Any] = {},
    ) -> tuple[bool, Optional[JSON]]:  # returns json
        try:
            with self.session.get(
                self.base_url + endpoint,
                params=params,
                headers=self.headers,
                timeout=5,
            ) as r:
                return (r.from_cache, r.json())

        except requests.JSONDecodeError:
            return (False, None)

    def request(
        self,
        endpoint: str,
        params: dict = {},
    ) -> JSON:
        return self._request(endpoint, params)[1]

    def requests(
        self,
        endpoint: str,
        params: Iterable[dict],
        *,
        delay_ms=750,
        progress=True,
    ) -> Iterable[JSON]:
        """Makes many requests while respecting throttle."""
        params = tqdm(params) if progress else params

        queries = (self._request(endpoint, p) for p in params)

        return (
            json
            for _, json in _rate_limit(
                queries,
                delay_ms=delay_ms,
                # don't throttle when using cache
                limit_when=lambda x: not x[0],
            )
            if json is not None
        )
