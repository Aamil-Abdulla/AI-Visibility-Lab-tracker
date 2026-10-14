"""Load and validate configuration from TOML files (no extra dependencies, Python 3.11+)."""
from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


@dataclass(frozen=True)
class RunSettings:
    default_provider: str
    runs_per_question: int
    delay_seconds: float
    max_retries: int
    retry_base_delay: float
    max_consecutive_failures: int
    output_file: Path
    manifest_dir: Path
    log_file: Path


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    model: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 800
    reasoning_effort: str | None = None  # optional: "low" | "medium" | "high" (reasoning models only)


@dataclass(frozen=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True)
class Brand:
    name: str
    aliases: tuple[str, ...] = ()
    target: bool = False


@dataclass(frozen=True)
class Config:
    run: RunSettings
    providers: dict[str, ProviderSettings]
    questions: list[Question]
    brands: list[Brand]


def _read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def load_config(
    settings_path: str | Path = "config/settings.toml",
    questions_path: str | Path = "config/questions.toml",
) -> Config:
    settings = _read_toml(Path(settings_path))
    content = _read_toml(Path(questions_path))

    try:
        r = settings["run"]
        run = RunSettings(
            default_provider=str(r["default_provider"]),
            runs_per_question=int(r["runs_per_question"]),
            delay_seconds=float(r["delay_seconds"]),
            max_retries=int(r["max_retries"]),
            retry_base_delay=float(r["retry_base_delay"]),
            max_consecutive_failures=int(r["max_consecutive_failures"]),
            output_file=Path(r["output_file"]),
            manifest_dir=Path(r["manifest_dir"]),
            log_file=Path(r["log_file"]),
        )
        providers: dict[str, ProviderSettings] = {}
        for name, p in settings.get("providers", {}).items():
            providers[name] = ProviderSettings(
                name=name,
                # Optional override, e.g. GROQ_MODEL in .env, so you can change a model name without editing files.
                model=os.environ.get(f"{name.upper()}_MODEL") or str(p["model"]),
                api_key_env=str(p["api_key_env"]),
                temperature=float(p.get("temperature", 0.7)),
                max_tokens=int(p.get("max_tokens", 800)),
                reasoning_effort=p.get("reasoning_effort"),
            )
        questions = [Question(id=str(q["id"]), text=str(q["text"])) for q in content.get("questions", [])]
        brands = [
            Brand(
                name=str(b["name"]),
                aliases=tuple(str(a) for a in b.get("aliases", [])),
                target=bool(b.get("target", False)),
            )
            for b in content.get("brands", [])
        ]
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Bad value in config: {exc}") from exc

    _validate(run, providers, questions, brands)
    return Config(run=run, providers=providers, questions=questions, brands=brands)


def _validate(run, providers, questions, brands) -> None:
    if run.runs_per_question < 1:
        raise ConfigError("runs_per_question must be at least 1")
    if run.max_retries < 1:
        raise ConfigError("max_retries must be at least 1")
    if run.delay_seconds < 0 or run.retry_base_delay < 0:
        raise ConfigError("delay values cannot be negative")
    if run.max_consecutive_failures < 1:
        raise ConfigError("max_consecutive_failures must be at least 1")
    if run.default_provider not in providers:
        raise ConfigError(f"default_provider '{run.default_provider}' is not defined under [providers]")
    for prov in providers.values():
        if prov.reasoning_effort not in (None, "low", "medium", "high"):
            raise ConfigError(f"reasoning_effort for {prov.name} must be low, medium or high")
    if not questions:
        raise ConfigError("No questions found in questions.toml")
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ConfigError("Question ids must be unique")
    if not brands:
        raise ConfigError("No brands found in questions.toml")
