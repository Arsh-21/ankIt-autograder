"""Turn a few teacher-marked sheets into graded examples for the grader.

Motivation: on the Mendeley exam the teacher gives part marks for attempts
that the answer key would call wrong, so a key-based grader comes out ~0.6
marks per question below the teacher. Rather than guessing a strictness, the
grader is shown how this teacher marked real answers to the same question
(GLaH, 2405.19694: graded examples help; HITL-Maths, 2603.13083: the rubric is
the most sensitive part, so calibrate the rubric, not the model). Nothing is
trained: the examples only go into the prompt.

Writes <dataset>/rubric.json (the run's rubric + up to 3 examples per
question: the student's answer as read + the teacher's mark) and
<dataset>/calibration_sheets.txt. The sheets listed there are used only as
examples; evaluating on them would be leakage.

Usage:
    # 1. Grade the calibration students only (none of the test students):
    python eval/run_ablation.py --dataset eval/datasets/mendeley --profile openrouter --only n1
        --sheets Student_19,Student_24,Student_29,Student_31,Student_37,Student_42
    # 2. Build the examples from that run:
    python scripts/make_calibration.py --run eval/runs/<that run> --dataset eval/datasets/mendeley --questions 21-35
    # 3. Evaluate on the other students, excluding the calibration ones:
    python eval/run_ablation.py ... --only n1,n1_calibrated --exclude <the same students>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.utils import normalize_qid  # noqa: E402
from eval.run_ablation import parse_questions  # noqa: E402


def pick(cands: list[tuple[float, str]], k: int) -> list[tuple[float, str]]:
    """Pick up to k examples that cover different marks.

    Marks are taken in the order highest, lowest, then the ones in between,
    with the median-length answer for each mark. If there are fewer distinct
    marks than k, the remaining slots are filled from the other candidates.

    Args:
        cands (list[tuple[float, str]]): Candidate (mark, answer text) pairs.
        k (int): Maximum number of examples.

    Returns:
        list[tuple[float, str]]: The chosen (mark, answer text) pairs.
    """
    by_mark: dict[float, list[str]] = {}
    for mark, text in cands:
        by_mark.setdefault(mark, []).append(text)
    marks = sorted(by_mark, reverse=True)
    order = [marks[0], marks[-1]] + marks[1:-1] if len(marks) > 1 else marks
    chosen = []
    for m in order:
        texts = sorted(by_mark[m], key=len)
        chosen.append((m, texts[len(texts) // 2]))
        if len(chosen) == k:
            return chosen
    rest = [(m, t) for m, t in cands if (m, t) not in chosen]  # fewer distinct marks than k
    return chosen + rest[: k - len(chosen)]


def main() -> None:
    """Add teacher-marked examples to the run's rubric and save it in the dataset folder."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="eval/runs/<id> folder of the calibration run")
    ap.add_argument("--variant", default="n1", help="which variant's reports to read")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--questions", default=None, help="add examples only to these, e.g. 21-35")
    ap.add_argument("--per-question", type=int, default=3)
    args = ap.parse_args()
    run, ds = Path(args.run), Path(args.dataset)

    rubric = json.loads((run / f"rubric_{args.variant}.json").read_text(encoding="utf-8"))
    teacher: dict[str, dict[str, float]] = {}
    with open(ds / "human_marks.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teacher.setdefault(row["sheet"], {})[normalize_qid(row["question_id"])] = float(row["marks"])

    answers: dict[str, list[tuple[float, str]]] = {}
    sheets = []
    for rep_file in sorted((run / "reports" / args.variant).glob("*/report.json")):
        sheet = rep_file.parent.name
        if sheet not in teacher:
            continue
        sheets.append(sheet)
        for qr in json.loads(rep_file.read_text(encoding="utf-8"))["questions"]:
            qid = normalize_qid(qr["id"])
            text = (qr.get("answer_text") or "").strip()
            if text and qid in teacher[sheet] and "Not attempted" not in qr.get("info", []):
                answers.setdefault(qid, []).append((teacher[sheet][qid], text))
    if not sheets:
        raise SystemExit(f"No reports with teacher marks under {run / 'reports' / args.variant}")

    keep = parse_questions(args.questions)
    added = 0
    for q in rubric["questions"]:
        qid = normalize_qid(q["id"])
        if keep and qid not in keep:
            continue
        ex = pick(answers.get(qid, []), args.per_question)
        q["examples"] = [{"answer": t, "marks": m, "comment": "Mark given by the course teacher."} for m, t in ex]
        added += len(ex)

    (ds / "rubric.json").write_text(json.dumps(rubric, indent=2, ensure_ascii=False), encoding="utf-8")
    (ds / "calibration_sheets.txt").write_text(",".join(sheets) + "\n", encoding="utf-8")
    print(f"Wrote {ds / 'rubric.json'}: {added} teacher-marked examples from {len(sheets)} sheets")
    print(f"Leave these out when testing:  --exclude {','.join(sheets)}")


if __name__ == "__main__":
    main()
