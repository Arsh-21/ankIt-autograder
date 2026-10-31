"""Compare ablation variants with a baseline, with confidence intervals. No model calls.

Reads items_<variant>.csv from one or more run folders and, for each variant,
reports MAE / bias / QWK and the difference from the baseline on the answers
both graded, with a 95% confidence interval. The interval comes from a paired
bootstrap over students (answers by one student are not independent, so whole
students are resampled). If the interval excludes 0, the difference is
unlikely to be chance.

A variant name that appears in two runs (e.g. n1 before and after a prompt
change) becomes n1@<run folder>; --base accepts that name too.

Usage:
    python eval/compare.py --runs eval/runs/20261029_030031 eval/runs/20261029_044753 --base n1
    python eval/compare.py --runs ... --base n1 --by-question   # adds a per-question MAE table
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import normalize_qid  # noqa: E402
from eval.metrics import qwk  # noqa: E402


def load(runs: list[str]) -> dict[str, dict[tuple, dict]]:
    """Load the graded answers of every variant in the given run folders.

    A variant name that appears in more than one run (e.g. n1 before and after
    a prompt change) is labelled variant@run_folder, so both versions can be
    compared.

    Args:
        runs (list[str]): Run folders that contain items_<variant>.csv files.

    Returns:
        dict[str, dict[tuple, dict]]: {variant: {(sheet, "Q<id>"): {"pred", "human"}}},
            with question ids normalised.
    """
    per_run = []
    for run in runs:
        found: dict[str, dict] = {}
        for f in sorted(Path(run).glob("items_*.csv")):
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    # question ids can differ between runs ("Q21" vs "21"): compare on the normalised id
                    found.setdefault(r["variant"], {})[(r["sheet"], "Q" + normalize_qid(r["qid"]))] = {
                        "pred": float(r["pred"]), "human": float(r["human"])}
        per_run.append((Path(run).name, found))
    count = defaultdict(int)
    for _, found in per_run:
        for v in found:
            count[v] += 1
    out: dict[str, dict[tuple, dict]] = {}
    for name, found in per_run:
        for v, rows in found.items():
            out[v if count[v] == 1 else f"{v}@{name}"] = rows
    return out


def stats(rows: dict[tuple, dict], keys: list[tuple]) -> tuple[float, float, float]:
    """Compute MAE, bias and QWK over a subset of answers.

    Args:
        rows (dict[tuple, dict]): One variant's answers, {(sheet, qid): {"pred", "human"}}.
        keys (list[tuple]): The (sheet, qid) keys to include; repeats count more than once.

    Returns:
        tuple[float, float, float]: (MAE, bias, QWK).
    """
    p = np.array([rows[k]["pred"] for k in keys])
    h = np.array([rows[k]["human"] for k in keys])
    return float(np.mean(np.abs(p - h))), float(np.mean(p - h)), qwk(p, h)


def main() -> None:
    """Print the comparison table (and optionally the per-question table) as Markdown."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", default="n1")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--by-question", action="store_true")
    args = ap.parse_args()

    data = load(args.runs)
    if args.base not in data:
        raise SystemExit(f"No variant {args.base!r}. Found: {sorted(data)}")
    base = data[args.base]
    print("Answers per variant: " + ", ".join(f"{v} {len(r)}" for v, r in sorted(data.items())))
    print(f"Baseline: {args.base}. Each variant is compared with the baseline on the answers both graded; "
          "95% CI from a paired bootstrap over students.\n")
    print("| variant | n | MAE | bias | QWK | ΔMAE vs base [95% CI] | ΔQWK vs base [95% CI] | closer / further / same |")
    print("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    rows_out = []
    for v, rows in data.items():
        keys = sorted(set(rows) & set(base))
        if not keys:
            rows_out.append((9e9, f"| {v} | 0 | | | | no answers in common with {args.base} | | |"))
            continue
        mae, bias, k = stats(rows, keys)
        if v == args.base:
            rows_out.append((mae, f"| **{v}** | {len(keys)} | {mae:.3f} | {bias:+.3f} | {k:.3f} | (baseline) | | |"))
            continue
        by_student = defaultdict(list)
        for key in keys:
            by_student[key[0]].append(key)
        students = sorted(by_student)
        d_mae, d_qwk = [], []
        for _ in range(args.reps):
            ks = [k2 for i in rng.choice(len(students), len(students)) for k2 in by_student[students[i]]]
            m1, _, q1 = stats(rows, ks)
            m0, _, q0 = stats(base, ks)
            d_mae.append(m1 - m0)
            d_qwk.append(q1 - q0)
        m0, _, q0 = stats(base, keys)
        e1 = [abs(rows[x]["pred"] - rows[x]["human"]) for x in keys]
        e0 = [abs(base[x]["pred"] - base[x]["human"]) for x in keys]
        closer = sum(a < b for a, b in zip(e1, e0, strict=True))
        further = sum(a > b for a, b in zip(e1, e0, strict=True))
        lo, hi = np.percentile(d_mae, [2.5, 97.5])
        qlo, qhi = np.nanpercentile(d_qwk, [2.5, 97.5])
        sig = " *" if hi < 0 or lo > 0 else ""
        rows_out.append((mae, f"| {v} | {len(keys)} | {mae:.3f} | {bias:+.3f} | {k:.3f} | {mae - m0:+.3f} "
                              f"[{lo:+.3f}, {hi:+.3f}]{sig} | {k - q0:+.3f} [{qlo:+.3f}, {qhi:+.3f}] | "
                              f"{closer} / {further} / {len(keys) - closer - further} |"))
    for _, line in sorted(rows_out, key=lambda t: t[0]):
        print(line)
    print("\n* = the interval excludes 0 (unlikely to be chance). Negative ΔMAE / positive ΔQWK = better.")
    order = sorted(data, key=lambda v: stats(data[v], sorted(set(data[v]) & set(base)) or list(data[v]))[0])
    keys = sorted(set(base))

    if args.by_question:
        qids = sorted({q for _, q in keys}, key=lambda q: (len(q), q))
        print("\nMAE per question\n")
        print("| variant | " + " | ".join(qids) + " |")
        print("|---|" + "---|" * len(qids))
        for v in order:
            cells = []
            for q in qids:
                ks = [k for k in keys if k[1] == q and k in data[v]]
                cells.append(f"{stats(data[v], ks)[0]:.2f}" if ks else "-")
            print(f"| {v} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
