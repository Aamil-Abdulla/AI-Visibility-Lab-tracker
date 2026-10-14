"""Runs the question x runs loop for one provider."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from avl.config import Config
from avl.extract import BrandMatcher, find_domains
from avl.models import AnswerRecord
from avl.providers.base import Provider, ProviderError
from avl.retry import call_with_retry
from avl.storage import CsvStore


@dataclass
class RunSummary:
    run_id: str
    planned: int
    saved: int = 0
    failed: int = 0
    skipped: int = 0
    truncated: int = 0  # answers cut off at max_tokens (finish_reason = length)
    status: str = "completed"  # completed | completed_with_errors | interrupted | aborted


class Runner:
    def __init__(
        self,
        config: Config,
        provider: Provider,
        store: CsvStore,
        logger: logging.Logger,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.provider = provider
        self.store = store
        self.log = logger
        self.sleep = sleep
        self.matcher = BrandMatcher(config.brands)

    def run(
        self,
        run_id: str,
        *,
        question_limit: int | None = None,
        runs: int | None = None,
        resume: bool = False,
        run_first: bool = False,
    ) -> RunSummary:
        cfg = self.config.run
        questions = self.config.questions[:question_limit] if question_limit else self.config.questions
        runs = runs or cfg.runs_per_question
        done = self.store.completed(run_id) if resume else set()
        summary = RunSummary(run_id=run_id, planned=len(questions) * runs)
        started = _utcnow()
        consecutive_failures = 0

        self.log.info("run_id=%s provider=%s model=%s planned=%d", run_id, self.provider.name,
                      self.provider.model, summary.planned)
        if run_first:
            # Every question gets run 1, then every question gets run 2, and so on.
            # If a daily quota stops you early, the questions stay evenly covered.
            tasks = [(q, n) for n in range(1, runs + 1) for q in questions]
        else:
            tasks = [(q, n) for q in questions for n in range(1, runs + 1)]
        try:
            for q, n in tasks:
                if (q.id, n) in done:
                    summary.skipped += 1
                    continue
                self.log.info("[%s run %d/%d] asking", q.id, n, runs)
                try:
                    completion = call_with_retry(
                        lambda: self.provider.complete(q.text),
                        max_retries=cfg.max_retries,
                        base_delay=cfg.retry_base_delay,
                        logger=self.log,
                        sleep=self.sleep,
                    )
                except ProviderError as exc:
                    summary.failed += 1
                    consecutive_failures += 1
                    self.log.error("[%s run %d] failed, not saved: %s", q.id, n, exc)
                    if exc.stop_run:
                        self.log.error("Daily quota used up. Stopping now. Resume after the quota resets.")
                        summary.status = "aborted"
                        return summary
                    if consecutive_failures >= cfg.max_consecutive_failures:
                        self.log.error("%d failures in a row. Stopping early.", consecutive_failures)
                        summary.status = "aborted"
                        return summary
                    continue

                consecutive_failures = 0
                answer = completion.text
                if completion.finish_reason == "length":
                    summary.truncated += 1
                    self.log.warning("[%s run %d] answer was cut off at max_tokens", q.id, n)
                self.store.append(AnswerRecord(
                    run_id=run_id,
                    timestamp=_utcnow(),
                    day=datetime.now().date().isoformat(),
                    provider=self.provider.name,
                    model=self.provider.model,
                    question_id=q.id,
                    question=q.text,
                    run_number=n,
                    answer_text=answer,
                    brands_found=";".join(self.matcher.find(answer)),
                    domains_found=";".join(find_domains(answer)),
                    finish_reason=completion.finish_reason,
                ))
                summary.saved += 1
                self.sleep(cfg.delay_seconds)
            if summary.failed and summary.status == "completed":
                summary.status = "completed_with_errors"
        except KeyboardInterrupt:
            summary.status = "interrupted"
            self.log.warning("Interrupted by user.")
        finally:
            self._write_manifest(summary, started, runs, len(questions), run_first)
        return summary

    def _write_manifest(self, summary: RunSummary, started: str, runs: int, n_questions: int,
                        run_first: bool = False) -> None:
        """One JSON file per run_id: what was run, with which settings, and what happened."""
        path: Path = self.config.run.manifest_dir / f"{summary.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "run_id": summary.run_id,
            "provider": self.provider.name,
            "model": self.provider.model,
            "temperature": self.provider.settings.temperature,
            "max_tokens": self.provider.settings.max_tokens,
            "questions": n_questions,
            "runs_per_question": runs,
            "sessions": [],
        }
        manifest["sessions"].append({
            "started_utc": started,
            "finished_utc": _utcnow(),
            "reasoning_effort": self.provider.settings.reasoning_effort,
            "order": "run-first" if run_first else "question-first",
            **asdict(summary),
        })
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
