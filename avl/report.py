"""Report: recompute every metric from the RAW answer text with the CURRENT brand list.

Nothing here calls an API. All numbers can be traced back to rows in answers.csv.
Small pilot: report counts, keep each provider in its own table.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from avl.config import Config
from avl.extract import BrandMatcher, find_domains

COMPLETE_ENDINGS = ".!?)|*"
HEADING_NAME = re.compile(r"^\s*(?:#{1,4}\s*)?\d+[.)]\s*\*\*(.+?)\*\*", re.M)


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF


def answer_status(finish_reason: str, text: str) -> str:
    """complete | cut_off | suspected_cut_off | other.
    finish_reason is exact for newer rows. Older rows ('unknown') get a cautious guess."""
    if finish_reason == "length":
        return "cut_off"
    if finish_reason == "stop":
        return "complete"
    if finish_reason == "unknown":
        t = text.strip()
        if len(t) >= 12000 and t and not (t[-1] in COMPLETE_ENDINGS or _is_emoji(t[-1])):
            return "suspected_cut_off"
        return "complete"
    return "other"


def prepare(df: pd.DataFrame, matcher: BrandMatcher, max_run: int) -> pd.DataFrame:
    d = df.fillna("").copy()
    d["run_number"] = d["run_number"].astype(int)
    d = d[d["run_number"] <= max_run].copy()
    d["brands"] = d["answer_text"].apply(matcher.find_ordered)
    d["domains"] = d["answer_text"].apply(find_domains)
    d["status"] = [answer_status(f, t) for f, t in zip(d["finish_reason"], d["answer_text"])]
    return d.reset_index(drop=True)


def coverage(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prov, qid), g in d.groupby(["provider", "question_id"]):
        rows.append(dict(
            provider=prov, question_id=qid, answers=len(g),
            runs_present=",".join(str(n) for n in sorted(g.run_number.unique())),
            complete=int((g.status == "complete").sum()),
            cut_off=int((g.status == "cut_off").sum()),
            suspected_cut_off=int((g.status == "suspected_cut_off").sum()),
            run_ids=";".join(sorted(g.run_id.unique())),
            dates=";".join(sorted(g.day.unique())),
        ))
    return pd.DataFrame(rows)


def mention_rates(d: pd.DataFrame, targets: set[str], subset: str) -> pd.DataFrame:
    cols = ["subset", "provider", "question_id", "brand", "answers_with_brand",
            "answers_total", "mention_rate", "is_target"]
    rows = []
    for (prov, qid), g in d.groupby(["provider", "question_id"]):
        total = len(g)
        counts = Counter(b for bs in g.brands for b in set(bs))
        for brand, c in counts.items():
            rows.append(dict(subset=subset, provider=prov, question_id=qid, brand=brand,
                             answers_with_brand=c, answers_total=total,
                             mention_rate=round(c / total, 3), is_target=brand in targets))
    return pd.DataFrame(rows, columns=cols)


def share_of_voice(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prov, g in d.groupby("provider"):
        counts = Counter(b for bs in g.brands for b in set(bs))
        total = sum(counts.values())
        for brand, c in counts.most_common():
            rows.append(dict(provider=prov, brand=brand, mentions=c, all_mentions=total,
                             share_of_voice=round(c / total, 3)))
    return pd.DataFrame(rows, columns=["provider", "brand", "mentions", "all_mentions", "share_of_voice"])


def top_domains(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prov, g in d.groupby("provider"):
        counts = Counter(dom for ds in g.domains for dom in set(ds))
        for dom, c in counts.most_common():
            rows.append(dict(provider=prov, domain=dom, answers_with_domain=c,
                             answers_total=len(g), label="as stated by the model"))
    return pd.DataFrame(rows, columns=["provider", "domain", "answers_with_domain", "answers_total", "label"])


def stability(d: pd.DataFrame) -> pd.DataFrame:
    """How often the FIRST-named listed brand changes between runs of the same question."""
    rows = []
    for (prov, qid), g in d.groupby(["provider", "question_id"]):
        firsts = [bs[0] if bs else "" for bs in g.sort_values("run_number").brands]
        top, n = Counter(firsts).most_common(1)[0]
        rows.append(dict(
            provider=prov, question_id=qid, answers=len(firsts),
            most_common_first_brand=top or "(none from my list)",
            agreement=round(n / len(firsts), 3),
            changes_between_runs=sum(1 for a, b in zip(firsts, firsts[1:]) if a != b),
            possible_changes=max(len(firsts) - 1, 0),
        ))
    return pd.DataFrame(rows)


def target_check(d: pd.DataFrame, targets: set[str]) -> pd.DataFrame:
    rows = []
    for prov, g in d.groupby("provider"):
        total = len(g)
        any_hit = int(g.brands.apply(lambda bs: bool(targets & set(bs))).sum())
        rows.append(dict(provider=prov, brand="(ANY target brand)", answers_with_brand=any_hit, answers_total=total))
        counts = Counter(b for bs in g.brands for b in set(bs) if b in targets)
        for brand in sorted(targets):
            rows.append(dict(provider=prov, brand=brand, answers_with_brand=counts.get(brand, 0), answers_total=total))
    return pd.DataFrame(rows, columns=["provider", "brand", "answers_with_brand", "answers_total"])


def clean_name(raw: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", raw).strip(" :*-–—").strip()


def candidate_brands(d: pd.DataFrame, matcher: BrandMatcher, min_answers: int = 2) -> pd.DataFrame:
    """Names and domains models used that are NOT on my brand list. Candidates only: review by hand."""
    rows = []
    for prov, g in d.groupby("provider"):
        names, doms = Counter(), Counter()
        for text, ds in zip(g.answer_text, g.domains):
            found = set()
            for raw in HEADING_NAME.findall(text):
                name = clean_name(raw)
                if 2 <= len(name) <= 40 and not matcher.find(name):
                    found.add(name)
            names.update(found)
            doms.update({x for x in ds if not matcher.find(x)})
        for kind, counter in (("name", names), ("domain", doms)):
            for cand, c in counter.most_common():
                if c >= min_answers:
                    rows.append(dict(provider=prov, kind=kind, candidate=cand, answers=c))
    return pd.DataFrame(rows, columns=["provider", "kind", "candidate", "answers"])


def _chart(d: pd.DataFrame, prov: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = d[d.provider == prov]
    counts = Counter(b for bs in g.brands for b in set(bs))
    top = counts.most_common(10)
    if not top:
        return
    names = [n for n, _ in top][::-1]
    rates = [c / len(g) for _, c in top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, rates)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of answers naming the brand")
    dates = f"{g.day.min()} to {g.day.max()}" if g.day.min() != g.day.max() else g.day.min()
    ax.set_title(f"{prov} ({g.model.iloc[0]}): {len(g)} answers, data dates {dates}", fontsize=10)
    fig.text(0.01, 0.01, "API pilot, not the ChatGPT app or Google AI Overviews. Mixes all questions.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / f"top_brands_{prov}.png", dpi=150)
    plt.close(fig)


def build(df: pd.DataFrame, config: Config, out_dir: Path, max_run: int = 3) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matcher = BrandMatcher(config.brands)
    targets = {b.name for b in config.brands if b.target}
    d = prepare(df, matcher, max_run)
    complete = d[d.status == "complete"]

    tables = {
        "coverage": coverage(d),
        "mention_rate": pd.concat([mention_rates(d, targets, "all_answers"),
                                   mention_rates(complete, targets, "complete_only")], ignore_index=True),
        "share_of_voice": share_of_voice(d),
        "top_domains": top_domains(d),
        "stability": stability(d),
        "target_check": target_check(d, targets),
        "candidate_brands": candidate_brands(d, matcher),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    for prov in d.provider.unique():
        _chart(d, prov, out_dir)
    (out_dir / "summary.md").write_text(_summary(d, tables, max_run), encoding="utf-8")
    return tables


def _summary(d: pd.DataFrame, tables: dict[str, pd.DataFrame], max_run: int) -> str:
    lines = ["# API pilot summary (all numbers are counts from saved answers)", "",
             f"Only runs 1 to {max_run} per question are used, so every question is compared fairly.",
             "Each provider is reported separately. API models are not the ChatGPT app or Google AI Overviews.", ""]
    for prov, g in d.groupby("provider"):
        counts = Counter(b for bs in g.brands for b in set(bs))
        tc = tables["target_check"]
        hit = tc[(tc.provider == prov) & (tc.brand == "(ANY target brand)")].iloc[0]
        lines += [
            f"## {prov} ({g.model.iloc[0]})",
            f"- Answers: {len(g)} | questions: {g.question_id.nunique()} | dates: {', '.join(sorted(g.day.unique()))}",
            f"- Complete: {(g.status == 'complete').sum()} | cut off: {(g.status == 'cut_off').sum()} | "
            f"suspected cut off (older rows): {(g.status == 'suspected_cut_off').sum()}",
            f"- Answers naming any target (TCG-related) brand: {int(hit.answers_with_brand)} of {int(hit.answers_total)}",
            "- Most-named brands (answers naming each, all questions mixed): "
            + ", ".join(f"{b} {c}" for b, c in counts.most_common(8)),
            "",
        ]
    cov = tables["coverage"]
    short = cov[cov.answers != max_run]
    lines.append("## Coverage check")
    if short.empty:
        lines.append(f"- Every provider/question pair has exactly {max_run} answers.")
    else:
        lines.append(f"- These provider/question pairs do NOT have exactly {max_run} answers:")
        lines += [f"  - {r.provider} {r.question_id}: {r.answers} (runs {r.runs_present})" for r in short.itertuples()]
    lines += ["", "Brands found with the current brand list; see candidate_brands.csv for names/domains not on it."]
    return "\n".join(lines) + "\n"
