"""Import SAF (Short Answer Feedback, communication networks, English;
Filighera et al., ACL 2022, CC BY 4.0) as a typed-answer test set.

Purpose of a second dataset: Mendeley is one teacher, one exam, handwritten.
SAF is a different subject, different graders, typed answers (so it tests the
grader alone, with no OCR), and its "test_unseen_questions" split has
questions the system has never seen: a check that nothing was tuned to the
Mendeley paper. It also has the graders' written feedback for every answer.

Scores in SAF are fractions of a point (0, 0.25, 0.5, ...). We rescale each
question to 4 marks so a quarter point becomes one mark; MAE is reported in
those 4-mark units (nMAE is scale-free).

Usage:
    pip install datasets
    python scripts/import_saf.py --split test_unseen_questions --limit 120
    python eval/run_ablation.py --dataset eval/datasets/saf_unseen_questions.csv --profile openrouter --only n1,n1_ref_none
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

REPO = "Short-Answer-Feedback/saf_communication_networks_english"


def main() -> None:
    """Download SAF, rescale scores, sample one split and write it as an eval CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_unseen_questions",
                    help="train | validation | test_unseen_answers | test_unseen_questions")
    ap.add_argument("--limit", type=int, default=120, help="random sample of this many answers (cost control)")
    ap.add_argument("--marks", type=float, default=4.0, help="rescale every question to this many marks")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets") from None

    data = load_dataset(REPO)
    best = defaultdict(float)  # the highest score any answer got = that question's maximum
    for split in data.values():
        for r in split:
            best[r["question"]] = max(best[r["question"]], float(r["score"]))
    rows = [r for r in data[args.split] if best[r["question"]] > 0]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    out = Path(args.out or f"eval/datasets/saf_{args.split.replace('test_', '')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "model_answer", "max_marks", "student_answer", "human_score", "human_feedback"])
        for i, r in enumerate(rows, start=1):
            scale = args.marks / best[r["question"]]
            w.writerow([i, r["question"], r["reference_answer"], args.marks, r["provided_answer"],
                        round(float(r["score"]) * scale * 4) / 4, r["answer_feedback"]])
    print(f"Wrote {len(rows)} answers to {len(set(r['question'] for r in rows))} questions to {out}")


if __name__ == "__main__":
    main()
