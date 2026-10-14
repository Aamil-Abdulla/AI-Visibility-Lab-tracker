import logging
from types import SimpleNamespace

import pytest

from avl.config import Brand, Config, ProviderSettings, Question, RunSettings
from avl.providers.base import Provider, ProviderError, Completion, is_daily_quota_error
from avl.providers.gemini_provider import GeminiProvider
from avl.providers.groq_provider import GroqProvider
from avl.runner import Runner
from avl.storage import CsvStore

LOG = logging.getLogger("test")

# Real error texts from the project logs (shortened).
GEMINI_DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
                "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
GROQ_DAILY = "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 197726"
GROQ_MINUTE = "Rate limit reached for model on tokens per minute (TPM): Limit 8000, Used 4666"


def test_detects_daily_not_minute():
    assert is_daily_quota_error(GEMINI_DAILY)
    assert is_daily_quota_error(GROQ_DAILY)
    assert not is_daily_quota_error(GROQ_MINUTE)
    assert not is_daily_quota_error("503 UNAVAILABLE high demand")


def _boom_client_gemini(msg, code=429):
    def f(**kw):
        e = Exception(msg)
        e.code = code
        raise e
    return SimpleNamespace(models=SimpleNamespace(generate_content=f))


def test_gemini_daily_quota_stops_run(monkeypatch):
    monkeypatch.setenv("K", "x")
    p = GeminiProvider(ProviderSettings("gemini", "m", "K"))
    p._client = _boom_client_gemini(GEMINI_DAILY)
    with pytest.raises(ProviderError) as e:
        p.complete("q")
    assert e.value.stop_run and not e.value.retryable


def test_groq_minute_limit_still_retryable(monkeypatch):
    monkeypatch.setenv("K", "x")
    p = GroqProvider(ProviderSettings("groq", "m", "K"))

    def f(**kw):
        e = Exception(GROQ_MINUTE)
        e.status_code = 429
        raise e
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=f)))
    with pytest.raises(ProviderError) as e:
        p.complete("q")
    assert e.value.retryable and not e.value.stop_run


class Fake(Provider):
    name = "fake"

    def __init__(self, script):
        super().__init__(ProviderSettings("fake", "fake-model", "X"))
        self.script = list(script)
        self.asked = []

    def complete(self, prompt):
        self.asked.append(prompt)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return Completion(item, "stop")


def cfg(tmp_path):
    run = RunSettings("fake", 2, 0, 3, 0, 5, tmp_path / "a.csv", tmp_path / "runs", tmp_path / "l.log")
    return Config(run, {}, [Question("q1", "A?"), Question("q2", "B?")], [Brand("Etsy")])


def test_run_first_order(tmp_path):
    c = cfg(tmp_path)
    prov = Fake(["x"] * 4)
    Runner(c, prov, CsvStore(c.run.output_file), LOG, sleep=lambda s: None).run("r", run_first=True)
    assert prov.asked == ["A?", "B?", "A?", "B?"]


def test_question_first_order_is_default(tmp_path):
    c = cfg(tmp_path)
    prov = Fake(["x"] * 4)
    Runner(c, prov, CsvStore(c.run.output_file), LOG, sleep=lambda s: None).run("r")
    assert prov.asked == ["A?", "A?", "B?", "B?"]


def test_stop_run_aborts_immediately(tmp_path):
    c = cfg(tmp_path)
    prov = Fake(["ok", ProviderError("daily", False, stop_run=True), "never"])
    s = Runner(c, prov, CsvStore(c.run.output_file), LOG, sleep=lambda s: None).run("r")
    assert s.status == "aborted" and s.saved == 1 and s.failed == 1 and len(prov.asked) == 2
