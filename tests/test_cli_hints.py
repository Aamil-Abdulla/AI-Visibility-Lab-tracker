from types import SimpleNamespace

from avl.cli import resume_hint, resume_matches_provider


def test_hint_includes_provider_and_options():
    args = SimpleNamespace(runs=3, run_first=True, reasoning_effort=None)
    hint = resume_hint("gemini", "gemini-20261012-153323", args)
    assert hint == "python -m avl run --provider gemini --resume gemini-20261012-153323 --runs 3 --run-first"


def test_hint_minimal():
    args = SimpleNamespace(runs=None, run_first=False, reasoning_effort=None)
    assert resume_hint("groq", "groq-1", args) == "python -m avl run --provider groq --resume groq-1"


def test_provider_mismatch_detected():
    assert resume_matches_provider("gemini", "gemini-20261012-153323")
    assert not resume_matches_provider("groq", "gemini-20261012-153323")
