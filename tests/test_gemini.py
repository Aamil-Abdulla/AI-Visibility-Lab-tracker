from types import SimpleNamespace

import pytest

from avl.config import ProviderSettings
from avl.providers.base import ProviderError
from avl.providers.gemini_provider import GeminiProvider


def make(monkeypatch, model="some-model"):
    monkeypatch.setenv("TEST_G", "dummy")
    return GeminiProvider(ProviderSettings("gemini", model, "TEST_G"))


def fake_client(text, finish):
    resp = SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))])
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))


def test_placeholder_model_is_refused(monkeypatch):
    with pytest.raises(ProviderError) as e:
        make(monkeypatch, model="SET_IN_ENV_GEMINI_MODEL")
    assert not e.value.retryable


def test_finish_reasons_use_shared_words(monkeypatch):
    p = make(monkeypatch)
    p._client = fake_client("hello", "STOP")
    assert p.complete("q").finish_reason == "stop"
    p._client = fake_client("hello", "MAX_TOKENS")
    assert p.complete("q").finish_reason == "length"


def test_empty_answer_is_an_error(monkeypatch):
    p = make(monkeypatch)
    p._client = fake_client("", "MAX_TOKENS")
    with pytest.raises(ProviderError):
        p.complete("q")


def test_error_codes_decide_retry(monkeypatch):
    p = make(monkeypatch)

    def boom(code):
        def f(**kw):
            err = Exception("x")
            err.code = code
            raise err
        return SimpleNamespace(models=SimpleNamespace(generate_content=f))

    p._client = boom(429)
    with pytest.raises(ProviderError) as e:
        p.complete("q")
    assert e.value.retryable
    p._client = boom(404)
    with pytest.raises(ProviderError) as e:
        p.complete("q")
    assert not e.value.retryable
