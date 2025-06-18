from typing import Any, Iterable, Optional, Callable
import time

import requests
import requests_cache


def rate_limit[T](
    iterable: Iterable[T],
    delay_ms: int | float = 500,
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


type JSON = Any

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://stats.nba.com/",
}

SESSION = requests_cache.CachedSession(".http_cache")


def _single_api_request(
    endpoint: str,
    params: dict[Any, Any] = {},
    *,
    base_url="https://stats.nba.com/stats/",
    headers: dict = HEADERS,
) -> tuple[bool, Optional[JSON]]:  # returns json
    try:
        with SESSION.get(base_url + endpoint, params=params, headers=headers) as r:
            return (r.from_cache, r.json())

    except requests.JSONDecodeError:
        return (False, None)


def api_requests(
    endpoint: str,
    params: dict | Iterable[dict],
    delay_ms=500,
    **kwargs,
) -> Iterable[JSON]:
    if isinstance(params, dict):
        params = [params]

    queries = (_single_api_request(endpoint, p, **kwargs) for p in params)

    return (
        json
        for _, json in rate_limit(
            queries,
            delay_ms=delay_ms,
            # don't throttle when using cache
            limit_when=lambda x: not x[0],
        )
        if json is not None
    )
