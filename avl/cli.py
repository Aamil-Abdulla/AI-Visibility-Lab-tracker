"""Command line: python -m avl check | run"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from avl.config import ConfigError, load_config
from avl.providers import ProviderError, get_provider
from avl.runner import Runner
from avl.storage import CsvStore, StorageError


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("avl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logfile = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    logfile.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def resume_hint(provider: str, run_id: str, args) -> str:
    """The full command to continue a run (the old hint left out the provider)."""
    parts = ["python -m avl run", f"--provider {provider}", f"--resume {run_id}"]
    if getattr(args, "runs", None):
        parts.append(f"--runs {args.runs}")
    if getattr(args, "run_first", False):
        parts.append("--run-first")
    if getattr(args, "reasoning_effort", None):
        parts.append(f"--reasoning-effort {args.reasoning_effort}")
    return " ".join(parts)


def resume_matches_provider(provider: str, run_id: str) -> bool:
    """Run IDs look like 'gemini-20261012-153323'. Refuse to resume one with a different provider."""
    return run_id.startswith(provider + "-")


def cmd_check(args) -> int:
    config = load_config(args.settings, args.questions)
    name = args.provider or config.run.default_provider
    p = config.providers.get(name)
    if p is None:
        print(f"Unknown provider: {name}")
        return 2
    targets = [b.name for b in config.brands if b.target]
    print("Config OK")
    print(f"  provider:       {name}")
    print(f"  model:          {p.model}   (verify this name on the provider's website)")
    print(f"  API key set:    {bool(os.environ.get(p.api_key_env))}  ({p.api_key_env})")
    print(f"  questions:      {len(config.questions)}")
    print(f"  brands:         {len(config.brands)} ({len(targets)} marked as target)")
    print(f"  runs/question:  {config.run.runs_per_question}")
    print(f"  total calls:    {len(config.questions) * config.run.runs_per_question}")
    if not targets:
        print("  WARNING: no target brands yet. Add the real shop names in config/questions.toml.")
    return 0


def cmd_run(args) -> int:
    config = load_config(args.settings, args.questions)
    log = _setup_logging(config.run.log_file)
    name = args.provider or config.run.default_provider
    if name not in config.providers:
        print(f"Unknown provider: {name}")
        return 2
    if args.resume and not resume_matches_provider(name, args.resume):
        log.error("Run %s belongs to a different provider than '%s'. Add --provider <name> to match it.",
                  args.resume, name)
        return 2
    settings = config.providers[name]
    if args.reasoning_effort:
        settings = replace(settings, reasoning_effort=args.reasoning_effort)
    try:
        provider = get_provider(settings)
    except ProviderError as exc:
        log.error("%s", exc)
        return 2

    run_id = args.resume or f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    store = CsvStore(config.run.output_file)
    try:
        store.check_format()  # fail before spending any API calls
    except StorageError as exc:
        log.error("%s", exc)
        return 2
    runner = Runner(config, provider, store, log)
    summary = runner.run(run_id, question_limit=args.limit_questions, runs=args.runs, resume=bool(args.resume),
                            run_first=args.run_first)

    log.info("%s. saved=%d failed=%d skipped=%d cut_off=%d file=%s", summary.status.upper(),
             summary.saved, summary.failed, summary.skipped, summary.truncated, config.run.output_file)
    if summary.failed or summary.status != "completed":
        log.info("To continue, run exactly this (on Windows use ./.venv/Scripts/python):")
        log.info("%s", resume_hint(name, run_id, args))
    if summary.status == "interrupted":
        return 130
    return 1 if (summary.failed or summary.status == "aborted") else 0


def cmd_migrate(args) -> int:
    config = load_config(args.settings, args.questions)
    store = CsvStore(config.run.output_file)
    n = store.migrate()
    if n:
        print(f"Migrated {n} rows in {config.run.output_file} (backup saved next to it).")
    else:
        print("Nothing to migrate. The CSV already has the current columns (or does not exist yet).")
    return 0


def cmd_report(args) -> int:
    import pandas as pd

    from avl import report

    config = load_config(args.settings, args.questions)
    path = config.run.output_file
    if not path.exists():
        print(f"No data file at {path}")
        return 2
    df = pd.read_csv(path)
    report.build(df, config, Path(args.out), max_run=args.max_run)
    print(f"Report written to {args.out}/ (start with summary.md)")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="avl", description="AI Visibility Lab tracker")
    parser.add_argument("--settings", default="config/settings.toml")
    parser.add_argument("--questions", default="config/questions.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="validate config, no API calls")
    p_check.add_argument("--provider")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="ask the questions and save answers")
    p_run.add_argument("--provider")
    p_run.add_argument("--limit-questions", type=int, help="only the first N questions (testing)")
    p_run.add_argument("--runs", type=int, help="runs per question (overrides config)")
    p_run.add_argument("--resume", metavar="RUN_ID", help="continue an earlier run")
    p_run.add_argument("--run-first", action="store_true",
                       help="ask every question once, then every question again (best when a daily quota may stop you)")
    p_run.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                       help="reasoning models only; overrides settings.toml. Leave unset to keep the model default")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="build tables and charts from the saved answers (no API calls)")
    p_rep.add_argument("--max-run", type=int, default=3, help="use runs 1..N per question (default 3)")
    p_rep.add_argument("--out", default="reports")
    p_rep.set_defaults(func=cmd_report)

    p_mig = sub.add_parser("migrate", help="upgrade an older answers.csv to the current columns")
    p_mig.set_defaults(func=cmd_migrate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2
