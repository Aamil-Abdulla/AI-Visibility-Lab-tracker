"""Provider interface. Add a new AI provider by subclassing Provider."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from avl.config import ProviderSettings


class ProviderError(Exception):
    """Error from an AI provider. `retryable` tells the retry logic what to do."""

    def __init__(self, message: str, retryable: bool, stop_run: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.stop_run = stop_run  # True = no point continuing today (for example daily quota used up)


def is_daily_quota_error(text: str) -> bool:
    """True when an error message says a PER-DAY limit was hit (waiting seconds or minutes will not help)."""
    t = text.lower()
    return any(k in t for k in ("perday", "per day", "(tpd)", "(rpd)"))


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str = "unknown"  # "stop" = complete, "length" = cut off by max_tokens


class Provider(ABC):
    name: str = "base"

    def __init__(self, settings: ProviderSettings):
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.model

    @abstractmethod
    def complete(self, prompt: str) -> Completion:
        """Send one question, return a Completion. Raise ProviderError on failure."""
