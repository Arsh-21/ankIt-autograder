"""How much each setting changes the marks, compared with a base variant, without using the
teacher's marks at all.

The teacher's marks in the Mendeley dataset follow one teacher's policy (part
marks for attempts) and are not a ground truth. This table answers a question
that does not depend on them: if a teacher switched this setting on, how many
answers would get a different mark, in which direction, by how much, and at
what cost. Rows are paired on the same answers of the same students.

Columns:
  changed   share of answers whose mark differs from the base
  shift     mean signed change per answer (+ = more generous than the base)
  abs diff  mean absolute change per answer
  sheet     mean signed change per sheet total
  cost/sheet, s/sheet   model cost (USD) and time per sheet, from the run's summary.csv

Usage:
    python eval/shift.py --runs eval/runs/<run1> eval/runs/<run2> --base n1@<run1>
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.compare import load  # noqa: E402


def costs(runs: list[str]) -> dict[str, tuple[float, float, int]]:
    """Cost, time and sheet count per variant, read from each run's summary.csv.

    Args:
        runs (list[str]): Run folders.

    Returns:
        dict[str, tuple[float, float, int]]: {variant: (cost in USD, seconds, sheets)}; a variant
            found in several runs is keyed variant@run, as in compare.load.
    """
    per_run = []
    for run in runs:
        rows = {}
        f = Path(run) / "summary.csv"
        sheets: dict[str, set] = defaultdict(set)
        for items in Path(run).glob("items_*.csv"):
            with open(items, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    sheets[r["variant"]].add(r["sheet"])
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    rows[r["variant"]] = (float(r.get("cost_usd") or 0), float(r.get("seconds") or 0),
                                          len(sheets.get(r["variant"], ())) or 1)
        per_run.append((Path(run).name, rows))
    count = defaultdict(int)
    for _, rows in per_run:
        for v in rows:
            count[v] += 1
    return {(v if count[v] == 1 else f"{v}@{name}"): x for name, rows in per_run for v, x in rows.items()}


def main() -> None:
    """Print the change table for every variant against the base."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="variant to compare against, e.g. n1 or n1@<run folder>")
    args = ap.parse_args()
    data, money = load(args.runs), costs(args.runs)
    if args.base not in data:
        raise SystemExit(f"No variant '{args.base}'. Found: {sorted(data)}")
    base = data[args.base]
    print(f"Against {args.base}, on the answers both graded\n")
    print("| variant | answers | changed | shift | abs diff | sheet | cost/sheet | s/sheet |")
    print("|---|---|---|---|---|---|---|---|")
    for v in sorted(data):
        keys = [k for k in data[v] if k in base]
        if not keys:
            continue
        d = [data[v][k]["pred"] - base[k]["pred"] for k in keys]
        per_sheet: dict[str, float] = defaultdict(float)
        for k, x in zip(keys, d, strict=True):
            per_sheet[k[0]] += x
        c, s, n = money.get(v, (0.0, 0.0, 1))
        print(f"| {v} | {len(keys)} | {sum(1 for x in d if x != 0) / len(d):.0%} | {sum(d) / len(d):+.2f} | "
              f"{sum(abs(x) for x in d) / len(d):.2f} | {sum(per_sheet.values()) / len(per_sheet):+.1f} | "
              f"${c / n:.3f} | {s / n:.0f} |")


if __name__ == "__main__":
    main()