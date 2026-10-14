from __future__ import annotations

import os

from avl.config import ProviderSettings
from avl.providers.base import Completion, Provider, ProviderError, is_daily_quota_error


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        if settings.model.startswith("SET_"):
            raise ProviderError(
                "No Gemini model chosen. Run list_gemini_models.py, then put GEMINI_MODEL=<id> in your .env file.",
                retryable=False,
            )
        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise ProviderError(f"{settings.api_key_env} is not set. Check your .env file.", retryable=False)
        from google import genai  # pip install google-genai (imported here so tests/check work without it)

        self._client = genai.Client(api_key=api_key)

    def complete(self, prompt: str) -> Completion:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.settings.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.settings.temperature,
                    max_output_tokens=self.settings.max_tokens,
                ),
            )
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            # Retry on rate limits (429), server errors (5xx) and network errors (no code).
            # Do not retry client errors such as a bad key or bad model name (4xx).
            retryable = code is None or code == 429 or (isinstance(code, int) and code >= 500)
            if is_daily_quota_error(str(exc)):
                raise ProviderError(
                    f"Daily free quota used up: {type(exc).__name__}: {exc}",
                    retryable=False,
                    stop_run=True,
                ) from exc
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=retryable) from exc

        reason = "unknown"
        if response.candidates:
            fr = response.candidates[0].finish_reason
            name = getattr(fr, "name", str(fr)) if fr is not None else "unknown"
            # Use the same words as the Groq provider: "stop" = complete, "length" = cut off.
            reason = {"STOP": "stop", "MAX_TOKENS": "length"}.get(name, name.lower())

        try:
            text = response.text or ""
        except Exception:  # blocked or empty responses can raise when reading .text
            text = ""
        if not text.strip():
            raise ProviderError(
                f"Empty answer (finish reason: {reason}). The model may have used all max_tokens "
                "on thinking, or the answer was blocked. Raise max_tokens or check the question.",
                retryable=False,
            )
        return Completion(text=text, finish_reason=reason)
