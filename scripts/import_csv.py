"""Convert a table of typed student answers with human scores into the CSV
format eval/run_ablation.py reads.

Output columns: id, question, model_answer, max_marks, student_answer,
human_score, human_feedback. The source column for each field is given on
the command line. Options for other files: --sep ";" for semicolon-separated
files, --max-col when each row has its own maximum, --feedback for a column
with the human's written feedback.

Example dataset: JorGPT (Spanish + English CS answers, 0-10 scores with
instructor feedback; Zenodo https://doi.org/10.5281/zenodo.18369061, CC BY 4.0).

Usage:
    python scripts/import_csv.py --src data/jorgpt/dataset_es.csv --out eval/datasets/jorgpt_es.csv
        --question question --answer student_answer --model-answer reference_answer
        --score human_score --max 10 --limit 150
"""

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    """Read the source table, keep scored non-empty answers, sample and write the CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--question", required=True, help="column with the question text")
    ap.add_argument("--answer", required=True, help="column with the student's answer")
    ap.add_argument("--score", required=True, help="column with the human score")
    ap.add_argument("--model-answer", default=None, help="column with the reference answer (optional)")
    ap.add_argument("--feedback", default=None, help="column with the human's feedback (optional)")
    ap.add_argument("--max", type=float, default=None, help="maximum score, same for every row")
    ap.add_argument("--max-col", default=None, help="or: column with each row's maximum")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--limit", type=int, default=None, help="random sample of this many rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.max is None and not args.max_col:
        raise SystemExit("Give --max or --max-col")

    with open(args.src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=args.sep))
    if not rows:
        raise SystemExit("The file has no rows")
    cols = [c for c in (args.question, args.answer, args.score, args.model_answer, args.feedback, args.max_col) if c]
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise SystemExit(f"No column(s) {missing}. The file has: {list(rows[0])}")
    rows = [r for r in rows if (r[args.answer] or "").strip() and (r[args.score] or "").strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "model_answer", "max_marks", "student_answer", "human_score", "human_feedback"])
        for i, r in enumerate(rows, start=1):
            mx = float(r[args.max_col]) if args.max_col else args.max
            w.writerow([i, r[args.question], r[args.model_answer] if args.model_answer else "", mx,
                        r[args.answer], float(r[args.score].replace(",", ".")),
                        r[args.feedback] if args.feedback else ""])
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
