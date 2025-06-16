import time
from typing import Iterable


def rate_limit(iterable: Iterable, delay_ms: int | float = 500):
    """Rate-limit an iterator"""
    next_time = time.monotonic()

    for item in iterable:
        yield item

        next_time += delay_ms / 1000
        sleep_time = next_time - time.monotonic()

        if sleep_time > 0:
            time.sleep(sleep_time)
