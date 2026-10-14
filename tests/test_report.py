import pandas as pd

from avl.config import Brand, Config, Question, RunSettings
from avl.extract import BrandMatcher
from avl.report import answer_status, build, candidate_brands, prepare, stability
from pathlib import Path


def cfg():
    run = RunSettings("groq", 3, 0, 1, 0, 3, Path("x"), Path("y"), Path("z"))
    return Config(run, {}, [Question("q1", "?")], [Brand("Etsy"), Brand("Zazzle"), Brand("Picanova", target=True)])


def row(prov, q, n, text, fr="stop", day="2026-10-13", run="r1"):
    return dict(run_id=run, timestamp="t", day=day, provider=prov, model=prov + "-m", question_id=q, question="?",
                run_number=n, answer_text=text, brands_found="", domains_found="", finish_reason=fr)


def test_answer_status():
    assert answer_status("length", "x") == "cut_off"
    assert answer_status("stop", "x") == "complete"
    assert answer_status("unknown", "a" * 13000 + " click") == "suspected_cut_off"
    assert answer_status("unknown", "a" * 13000 + " Happy printing! \U0001F680") == "complete"
    assert answer_status("unknown", "short answer without end") == "complete"


def test_max_run_filters_and_providers_stay_separate():
    df = pd.DataFrame([row("groq", "q1", 1, "Etsy"), row("groq", "q1", 4, "Etsy"), row("gemini", "q1", 1, "Zazzle")])
    d = prepare(df, BrandMatcher(cfg().brands), 3)
    assert len(d) == 2 and set(d.provider) == {"groq", "gemini"}


def test_stability_uses_text_order():
    df = pd.DataFrame([row("groq", "q1", 1, "Zazzle then Etsy"), row("groq", "q1", 2, "Etsy then Zazzle"),
                       row("groq", "q1", 3, "Zazzle first again, Etsy")])
    d = prepare(df, BrandMatcher(cfg().brands), 3)
    s = stability(d).iloc[0]
    assert s.most_common_first_brand == "Zazzle" and s.changes_between_runs == 2 and s.possible_changes == 2


def test_candidates_find_unlisted_names_and_domains():
    text = "## 1. **CanvasJet (canvasjet.ae)**\nsome text\n2. **Etsy**"
    df = pd.DataFrame([row("gemini", "q1", 1, text), row("gemini", "q1", 2, text)])
    m = BrandMatcher(cfg().brands)
    c = candidate_brands(prepare(df, m, 3), m)
    assert ("name", "CanvasJet") in set(zip(c.kind, c.candidate))
    assert ("domain", "canvasjet.ae") in set(zip(c.kind, c.candidate))
    assert "Etsy" not in set(c.candidate)


def test_build_writes_files_and_target_zero(tmp_path):
    df = pd.DataFrame([row("groq", "q1", n, "Try Etsy and Zazzle.") for n in (1, 2, 3)]
                      + [row("gemini", "q1", n, "Try Zazzle.") for n in (1, 2)])
    t = build(df, cfg(), tmp_path, max_run=3)
    for f in ("coverage", "mention_rate", "share_of_voice", "top_domains", "stability", "target_check",
              "candidate_brands"):
        assert (tmp_path / f"{f}.csv").exists()
    assert (tmp_path / "summary.md").exists() and (tmp_path / "top_brands_groq.png").exists()
    tc = t["target_check"]
    assert int(tc[(tc.provider == "groq") & (tc.brand == "(ANY target brand)")].answers_with_brand.iloc[0]) == 0
    assert "do NOT have exactly 3" in (tmp_path / "summary.md").read_text(encoding="utf-8")  # gemini has 2
