"""Retry with exponential backoff."""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

from avl.providers.base import ProviderError

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    logger: logging.Logger,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(). Retry only retryable ProviderErrors. Waits base, 2x base, 4x base ... (+/-10%)."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except ProviderError as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            wait = base_delay * (2 ** (attempt - 1)) * random.uniform(0.9, 1.1)
            logger.warning("Attempt %d/%d failed (%s). Waiting %.1fs.", attempt, max_retries, exc, wait)
            sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover
