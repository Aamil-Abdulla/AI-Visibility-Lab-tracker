from __future__ import annotations

import os

from avl.config import ProviderSettings
from avl.providers.base import Completion, Provider, ProviderError, is_daily_quota_error


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise ProviderError(f"{settings.api_key_env} is not set. Check your .env file.", retryable=False)
        from groq import Groq  # imported here so tests and `check` work without the SDK

        self._client = Groq(api_key=api_key)

    def complete(self, prompt: str) -> Completion:
        extra = {}
        if self.settings.reasoning_effort:
            # Sent as an extra body field so it works with any SDK version.
            extra["extra_body"] = {"reasoning_effort": self.settings.reasoning_effort}
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                **extra,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            # Retry on rate limits (429), server errors (5xx) and network errors (no status).
            # Do not retry client errors such as bad key (401) or bad model name (404).
            retryable = status is None or status == 429 or (isinstance(status, int) and status >= 500)
            if is_daily_quota_error(str(exc)):
                raise ProviderError(
                    f"Daily free quota used up: {type(exc).__name__}: {exc}",
                    retryable=False,
                    stop_run=True,
                ) from exc
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=retryable) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise ProviderError(
                "Empty answer. The model may have used all max_tokens on reasoning. Raise max_tokens.",
                retryable=False,
            )
        return Completion(text=text, finish_reason=str(choice.finish_reason or "unknown"))
