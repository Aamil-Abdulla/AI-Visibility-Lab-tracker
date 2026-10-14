"""AI Visibility Lab: report from the raw answers.

Run:  ./.venv/Scripts/python report.py            (uses runs 1-3 of every question)
      ./.venv/Scripts/python report.py --max-run 5

Everything is recomputed from the raw answer_text, not from the stored brands_found column.
Each provider (model) gets its own folder and its own tables. They are never mixed.
This is a PILOT: counts of answers, not market share.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from avl.config import load_config
from avl.extract import BrandMatcher, find_domains

BOLD = re.compile(r"\*\*([^*\n]{2,60})\*\*")
END_OK = set(".!?)\"'*|`>:")  # an answer ending with one of these probably finished


def ordered_brands(text: str, matcher: BrandMatcher) -> list[str]:
    """Listed brands found in text, in order of first appearance."""
    hits = []
    for name, pattern in matcher._patterns:  # same patterns the tracker uses
        m = pattern.search(text)
        if m:
            hits.append((m.start(), name))
    return [name for _, name in sorted(hits)]


def bold_names(text: str) -> set[str]:
    out = set()
    for raw in BOLD.findall(text):
        name = re.sub(r"^\s*\d+[.)]\s*", "", raw).strip(" :-\u2013\u2014")
        if name and len(name.split()) <= 4:
            out.add(name)
    return out


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def analyse(df: pd.DataFrame, cfg, out_dir: Path, plot: bool = True) -> list[str]:
    matcher = BrandMatcher(cfg.brands)
    targets = [b.name for b in cfg.brands if b.target]
    qtext = {q.id: q.text for q in cfg.questions}
    n = len(df)
    texts = df["answer_text"].fillna("").tolist()
    qids = df["question_id"].tolist()

    brand_lists = [ordered_brands(t, matcher) for t in texts]
    dates = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    date_label = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d} (UTC)"
    models = ", ".join(sorted(df["model"].astype(str).unique()))
    lines = [f"Model(s): {models}", f"Answers used: {n}", f"Data dates: {date_label}",
             f"Runs per question used: {df['run_number'].min()}-{df['run_number'].max()}", ""]

    # 1. Brand table: mention rate and share of voice (answer level)
    counts = Counter(b for bl in brand_lists for b in set(bl))
    total_mentions = sum(counts.values())
    rows = [{"brand": b, "target": b in targets, "answers_naming_brand": c,
             "mention_rate_pct": pct(c, n), "share_of_voice_pct": pct(c, total_mentions)}
            for b, c in counts.most_common()]
    brand_df = pd.DataFrame(rows, columns=["brand", "target", "answers_naming_brand",
                                           "mention_rate_pct", "share_of_voice_pct"])
    brand_df.to_csv(out_dir / "brands_overall.csv", index=False)
    lines.append("TOP BRANDS (answers naming the brand / all answers; mixes all questions, NOT market share)")
    for r in rows[:12]:
        lines.append(f"  {r['brand']:<20} {r['answers_naming_brand']:>3} of {n}  ({r['mention_rate_pct']}%)")

    # 2. Brand x question mention rates
    per_q = []
    for qid in sorted(set(qids), key=lambda x: int(re.sub(r"\D", "", x) or 0)):
        idx = [i for i, q in enumerate(qids) if q == qid]
        c = Counter(b for i in idx for b in set(brand_lists[i]))
        for b, k in c.items():
            per_q.append({"question_id": qid, "question": qtext.get(qid, ""), "brand": b,
                          "answers_naming_brand": k, "answers": len(idx), "mention_rate_pct": pct(k, len(idx))})
    pq = pd.DataFrame(per_q, columns=["question_id", "question", "brand", "answers_naming_brand",
                                      "answers", "mention_rate_pct"])
    pq.to_csv(out_dir / "brands_by_question.csv", index=False)

    # 3. Target-brand check (TCG-related)
    lines += ["", "TARGET BRANDS (names from TCG's website)"]
    any_target = sum(1 for bl in brand_lists if any(b in targets for b in bl))
    lines.append(f"  Answers naming ANY target brand: {any_target} of {n}")
    for b in targets:
        if counts.get(b):
            lines.append(f"  {b}: {counts[b]} of {n}")

    # 4. Top domains (as stated by the model: may be invented)
    dom = Counter(d for t in texts for d in set(find_domains(t)))
    dom_df = pd.DataFrame([{"domain": d, "answers_naming_domain": c, "rate_pct": pct(c, n)}
                           for d, c in dom.most_common()], columns=["domain", "answers_naming_domain", "rate_pct"])
    dom_df.to_csv(out_dir / "domains.csv", index=False)
    lines += ["", "TOP DOMAINS (as stated by the model, unverified)"]
    for d, c in dom.most_common(10):
        lines.append(f"  {d:<28} {c} of {n}")

    # 5. Stability: does the FIRST brand named change between runs?
    st = []
    for qid in sorted(set(qids), key=lambda x: int(re.sub(r"\D", "", x) or 0)):
        firsts = [(brand_lists[i][0] if brand_lists[i] else "(none)") for i, q in enumerate(qids) if q == qid]
        top, k = Counter(firsts).most_common(1)[0]
        st.append({"question_id": qid, "answers": len(firsts), "most_common_first_brand": top,
                   "times_first": k, "distinct_first_brands": len(set(firsts))})
    st_df = pd.DataFrame(st, columns=["question_id", "answers", "most_common_first_brand",
                                      "times_first", "distinct_first_brands"])
    st_df.to_csv(out_dir / "stability_first_brand.csv", index=False)
    lines += ["", "STABILITY (first-named brand per question; 'distinct' = how many different first brands)"]
    for r in st:
        lines.append(f"  {r['question_id']:<4} {r['most_common_first_brand']:<18} first in {r['times_first']}/{r['answers']}"
                     f"  distinct={r['distinct_first_brands']}")

    # 6. Unlisted brand discovery (bold names not on the brand list). Noisy: verify by hand.
    seen = Counter()
    for t in texts:
        for name in bold_names(t):
            if not matcher.find(name):
                seen[name.lower()] += 1
    disc = pd.DataFrame([{"name_lowercase": k, "answers_naming": c} for k, c in seen.most_common()
                         if c >= 2], columns=["name_lowercase", "answers_naming"])
    disc.to_csv(out_dir / "unlisted_candidates.csv", index=False)
    lines += ["", "UNLISTED NAMES in bold, seen in 2+ answers (some are headings, not brands; check by hand)"]
    for k, c in list(seen.most_common(15)):
        if c >= 2:
            lines.append(f"  {k:<30} {c} answers")

    # 7. Cut-offs
    fr = df["finish_reason"].fillna("unknown").value_counts().to_dict()
    unknown_idx = [i for i, f in enumerate(df["finish_reason"].fillna("unknown")) if f == "unknown"]
    suspected = sum(1 for i in unknown_idx if texts[i].rstrip() and texts[i].rstrip()[-1] not in END_OK)
    lines += ["", "CUT-OFFS", f"  finish_reason counts: {fr}",
              f"  confirmed cut off (length): {fr.get('length', 0)}",
              f"  suspected cut off among 'unknown' rows (answer does not end like a finished text): {suspected}"]

    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = rows[:12][::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([r["brand"] for r in top], [r["mention_rate_pct"] for r in top],
                color=["#c0392b" if r["target"] else "#34609a" for r in top])
        ax.set_xlabel("% of answers naming the brand")
        ax.set_title(f"{models}: {n} answers, pilot\nData: {date_label}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "chart_top_brands.png", dpi=150)
        plt.close(fig)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/answers.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--max-run", type=int, default=3, help="use runs 1..N only (fair across providers)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    df["run_number"] = df["run_number"].astype(int)
    df = df[df["run_number"] <= args.max_run]

    all_lines: list[str] = []
    for provider, part in df.groupby("provider"):
        out_dir = Path(args.out) / provider
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"===== {provider.upper()} =====", *analyse(part, cfg, out_dir, plot=not args.no_plot), ""]
        all_lines += lines
    text = "\n".join(all_lines)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nTables and charts saved in {args.out}/<provider>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())