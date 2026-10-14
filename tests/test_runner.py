import csv
import logging
from pathlib import Path

import pytest

from avl.config import Brand, Config, ProviderSettings, Question, RunSettings
from avl.providers.base import Completion, Provider, ProviderError
from avl.retry import call_with_retry
from avl.runner import Runner
from avl.storage import CsvStore, StorageError

LOG = logging.getLogger("test")


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, script):
        super().__init__(ProviderSettings("fake", "fake-model", "X"))
        self.script = list(script)  # each item: str answer or ProviderError
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return Completion(*item)
        return Completion(item, "stop")


def make_config(tmp_path: Path, max_consecutive=3) -> Config:
    run = RunSettings("fake", 2, 0, 2, 0, max_consecutive, tmp_path / "a.csv", tmp_path / "runs", tmp_path / "l.log")
    return Config(run, {}, [Question("q1", "Where?"), Question("q2", "Best?")], [Brand("Etsy")])


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_saves_rows_with_brands_and_domains(tmp_path):
    cfg = make_config(tmp_path)
    prov = FakeProvider(["Try Etsy at etsy.com"] * 4)
    s = Runner(cfg, prov, CsvStore(cfg.run.output_file), LOG, sleep=lambda x: None).run("r1")
    assert (s.saved, s.failed, s.status) == (4, 0, "completed")
    rows = read_rows(cfg.run.output_file)
    assert len(rows) == 4
    assert rows[0]["brands_found"] == "Etsy" and rows[0]["domains_found"] == "etsy.com"
    assert (cfg.run.manifest_dir / "r1.json").exists()


def test_retry_then_success(tmp_path):
    cfg = make_config(tmp_path)
    prov = FakeProvider([ProviderError("429", True), "ok"] + ["ok"] * 3)
    s = Runner(cfg, prov, CsvStore(cfg.run.output_file), LOG, sleep=lambda x: None).run("r1")
    assert s.saved == 4 and s.failed == 0


def test_non_retryable_not_retried(tmp_path):
    def boom():
        raise ProviderError("bad key", False)
    calls = []
    with pytest.raises(ProviderError):
        call_with_retry(lambda: (calls.append(1), boom()), max_retries=3, base_delay=0, logger=LOG, sleep=lambda x: None)
    assert len(calls) == 1


def test_abort_after_consecutive_failures(tmp_path):
    cfg = make_config(tmp_path, max_consecutive=2)
    prov = FakeProvider([ProviderError("bad key", False)] * 5)
    s = Runner(cfg, prov, CsvStore(cfg.run.output_file), LOG, sleep=lambda x: None).run("r1")
    assert s.status == "aborted" and s.failed == 2 and prov.calls == 2


def test_resume_skips_saved_rows(tmp_path):
    cfg = make_config(tmp_path)
    store = CsvStore(cfg.run.output_file)
    first = FakeProvider(["a", "b", ProviderError("x", False), ProviderError("x", False)])
    Runner(cfg, first, store, LOG, sleep=lambda x: None).run("r1")  # saves 2, fails 2
    second = FakeProvider(["c", "d"])
    s = Runner(cfg, second, store, LOG, sleep=lambda x: None).run("r1", resume=True)
    assert (s.saved, s.skipped) == (2, 2)
    assert len(read_rows(cfg.run.output_file)) == 4


def test_finish_reason_saved_and_truncation_counted(tmp_path):
    cfg = make_config(tmp_path)
    prov = FakeProvider(["fine", ("cut off here", "length"), "fine", "fine"])
    s = Runner(cfg, prov, CsvStore(cfg.run.output_file), LOG, sleep=lambda x: None).run("r1")
    rows = read_rows(cfg.run.output_file)
    assert [r["finish_reason"] for r in rows] == ["stop", "length", "stop", "stop"]
    assert s.truncated == 1


def test_completed_with_errors_status(tmp_path):
    cfg = make_config(tmp_path)
    prov = FakeProvider(["a", ProviderError("x", False), "b", "c"])
    s = Runner(cfg, prov, CsvStore(cfg.run.output_file), LOG, sleep=lambda x: None).run("r1")
    assert s.status == "completed_with_errors" and s.failed == 1


def test_old_csv_blocked_then_migrated(tmp_path):
    path = tmp_path / "old.csv"
    old_header = "run_id,timestamp,day,provider,model,question_id,question,run_number,answer_text,brands_found,domains_found"
    path.write_text(old_header + "\nr1,t,d,groq,m,q1,Q?,1,hello,,\n", encoding="utf-8")
    store = CsvStore(path)
    with pytest.raises(StorageError):
        store.check_format()
    assert store.migrate() == 1
    store.check_format()
    rows = read_rows(path)
    assert rows[0]["finish_reason"] == "unknown" and rows[0]["answer_text"] == "hello"
    assert list(tmp_path.glob("old.csv.backup-*"))
    assert store.migrate() == 0  # idempotent
    assert store.completed("r1") == {("q1", 1)}


def test_groq_provider_sends_reasoning_effort_and_reads_finish_reason(monkeypatch):
    from types import SimpleNamespace
    from avl.providers.groq_provider import GroqProvider

    monkeypatch.setenv("TEST_KEY", "dummy")
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(content="answer text")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    prov = GroqProvider(ProviderSettings("groq", "m", "TEST_KEY", reasoning_effort="low"))
    prov._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    c = prov.complete("hi")
    assert (c.text, c.finish_reason) == ("answer text", "length")
    assert seen["extra_body"] == {"reasoning_effort": "low"}

    seen.clear()
    prov2 = GroqProvider(ProviderSettings("groq", "m", "TEST_KEY"))
    prov2._client = prov._client
    prov2.complete("hi")
    assert "extra_body" not in seen
