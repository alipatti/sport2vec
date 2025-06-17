import time
from typing import Callable, Iterable


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
