from __future__ import annotations

from avl.config import ProviderSettings
from avl.providers.base import Provider, ProviderError
from avl.providers.gemini_provider import GeminiProvider
from avl.providers.groq_provider import GroqProvider

# To add another provider: create its file, add one line here, add a [providers.NAME] block in settings.toml.
REGISTRY: dict[str, type[Provider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def get_provider(settings: ProviderSettings) -> Provider:
    try:
        cls = REGISTRY[settings.name]
    except KeyError:
        raise ProviderError(f"No provider code registered for '{settings.name}'", retryable=False) from None
    return cls(settings)


__all__ = ["Provider", "ProviderError", "get_provider", "REGISTRY"]
