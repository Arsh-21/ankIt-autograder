"""Convert the Mendeley dataset "A Dataset of Digitized Student Examination
Papers, Answer Keys, and Manual Evaluations for Automated Grading Research"
(Dinesh K P, CC BY 4.0, https://data.mendeley.com/datasets/sf3kvjwknt/1)
into the layout eval/run_ablation.py reads.

What is in it: 50 students' scanned handwritten answer sheets for a Data
Science exam. Part I: Q1-Q20 multiple choice, 1 mark each. Part II: Q21-Q35
short answers, 2 marks each (50 in total). Plus the question paper, the
answer key, the teacher's marked PDFs, and Teacher_manual_marks_Anonymized.csv.

How the teacher's marks are read:
  * Q21-Q35: the CSV holds the marks. An empty cell ("NC", left blank or not
    marked) or "NM" counts as 0, as in the dataset's own totals.
  * Q1-Q20: the CSV holds only the LETTER each student chose, plus the
    teacher's MCQ total. Comparing letters with answerkey.txt reproduces the
    teacher's total for just 13/50 students. Taking D ("All of the mentioned")
    as Q1's answer instead of B reproduces it for 32/50, so the teacher marked
    Q1 with D as the answer; this script follows the teacher (--strict-key keeps
    B). For the other 18 students the letters and the total disagree
    (transcription noise), so by default their MCQ rows are left out of the
    expected marks (--mcq all keeps them, --mcq none drops all MCQ rows). Their
    sheets are still graded in full.

Also writes fixed student splits (split_calibration/test/dev/ablation.txt).

Usage (from the project folder):
    python scripts/import_mendeley.py --src "data/mendeley_raw" --out eval/datasets/mendeley
"""

import argparse
import csv
import re
import shutil
from pathlib import Path


def _find(src: Path, pattern: str) -> Path:
    """Find the first file under src (recursively) that matches a glob pattern.

    Args:
        src (Path): Folder to search.
        pattern (str): Glob pattern, e.g. "Teacher_manual_marks*.csv".

    Returns:
        Path: The first match in sorted order.

    Raises:
        SystemExit: If nothing matches.
    """
    hits = sorted(src.rglob(pattern))
    if not hits:
        raise SystemExit(f"Could not find {pattern} under {src}")
    return hits[0]


def _to_float(x: object) -> float | None:
    """Convert a CSV cell to float.

    Args:
        x (object): The cell value.

    Returns:
        float | None: The number, or None if the cell is not numeric.
    """
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def main() -> None:
    """Write the question paper, rubric, model answer, human marks, sheets and splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder where you unzipped the Mendeley download")
    ap.add_argument("--out", default="eval/datasets/mendeley")
    ap.add_argument("--limit", type=int, default=None, help="only the first N students (by number)")
    ap.add_argument("--mcq", choices=["consistent", "all", "none"], default="consistent",
                    help="which students' Q1-Q20 marks to keep as expected marks (see above)")
    ap.add_argument("--strict-key", action="store_true", help="use answerkey.txt's B for Q1, not the teacher's D")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    question_txt = _find(src, "Question.txt")
    key_txt = _find(src, "answerkey.txt")
    marks_csv = _find(src, "Teacher_manual_marks*.csv")
    root = marks_csv.parent
    sheets_dir, marked_dir = root / "Student_Pdf", root / "Corrected_Pdf"
    if not sheets_dir.is_dir():
        raise SystemExit(f"No Student_Pdf folder next to {marks_csv.name}")

    # ---------------------------------------------------------------- answer key
    key = {}
    with open(key_txt, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key[int(row["Question_Number"])] = (row["Type"].strip(), row["Correct_Answer"].strip())
    accepted = {q: {ans.upper()} for q, (kind, ans) in key.items() if kind.upper() == "MCQ"}
    if not args.strict_key:
        accepted[1] = {"D"}  # the teacher's own marking (see the module docstring)

    # question paper as-is; marking scheme + model answer written from the key
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    shutil.copy(question_txt, out / "question_paper.txt")
    scheme = ["Marking scheme", "",
              "Part I (Q1-Q20): multiple choice, 1 mark each, all or nothing. The student writes the letter of "
              "the chosen option.", ""]
    model = ["Model answers", ""]
    for q in sorted(key):
        kind, ans = key[q]
        if kind.upper() == "MCQ":
            ok = " or ".join(sorted(accepted[q]))
            scheme.append(f"Q{q} (1 mark): correct option {ok}. Any other option or no answer: 0.")
            model.append(f"Q{q}. {ok}")
        else:
            scheme.append(f"Q{q} (2 marks): expected key points: {ans} Award 2 for a correct and complete answer, "
                          "1 for a partly correct or incomplete answer, 0 if wrong, irrelevant or blank.")
            model.append(f"Q{q}. {ans}")
        if q == 20:
            scheme += ["", "Part II (Q21-Q35): short answers, 2 marks each.", ""]
    (out / "rubric.txt").write_text("\n".join(scheme) + "\n", encoding="utf-8")
    (out / "model_answer.txt").write_text("\n".join(model) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- teacher marks
    with open(marks_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(re.sub(r"\D", "", r["Student_ID"]) or 0))
    if args.limit:
        rows = rows[:args.limit]

    human, problems, copied, inconsistent = [], [], 0, []
    for r in rows:
        sid = r["Student_ID"].strip()
        pdf = sheets_dir / f"{sid}.pdf"
        if not pdf.exists():
            problems.append(f"{sid}: no {pdf.name} in Student_Pdf")
            continue
        mcq, short = [], []
        for q in range(1, 36):
            cell = (r.get(str(q)) or "").strip()
            if q <= 20:
                mcq.append([sid, f"Q{q}", 1.0 if cell.upper() in accepted[q] else 0.0])
            else:
                mark = 0.0 if cell.upper() in ("", "NC", "NM") else _to_float(cell)
                if mark is None:
                    problems.append(f"{sid} Q{q}: unreadable mark {cell!r}, counted as 0")
                    mark = 0.0
                if mark > 2:  # a typo in the teacher's sheet (e.g. 4 for a 2-mark question): not usable
                    problems.append(f"{sid} Q{q}: mark {mark:g} is above the maximum 2, left out")
                    continue
                short.append([sid, f"Q{q}", mark])
        # sanity checks against the dataset's own totals
        mcq_ok = abs(sum(m for *_, m in mcq) - (_to_float(r.get("MCQ mark", "")) or 0)) < 1e-6
        if not mcq_ok:
            inconsistent.append(sid)
        theirs = _to_float(r.get("2 marks", ""))
        if theirs is not None and len(short) == 15 and abs(sum(m for *_, m in short) - theirs) > 1e-6:
            problems.append(f"{sid}: Q21-Q35 add up to {sum(m for *_, m in short):g} but the CSV total is {theirs:g}")
        if args.mcq == "all" or (args.mcq == "consistent" and mcq_ok):
            human += mcq
        human += short
        shutil.copy(pdf, out / "sheets" / pdf.name)
        copied += 1

    with open(out / "human_marks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "question_id", "marks"])
        w.writerows(human)
    if marked_dir.is_dir():  # the teacher's marked sheets, to compare by eye (not used by the runner)
        (out / "teacher_marked").mkdir(exist_ok=True)
        for r in rows:
            p = marked_dir / f"{r['Student_ID'].strip()}.pdf"
            if p.exists():
                shutil.copy(p, out / "teacher_marked" / p.name)

    # fixed splits, so every run uses the same students for the same purpose:
    #   calibration - only as teacher-marked examples (scripts/make_calibration.py)
    #   test        - held out: run once with the frozen setup; these are the numbers in the README
    #   ablation    - 20 students for comparing settings; dev = everything that is not calibration or test
    have = [r["Student_ID"].strip() for r in rows if (sheets_dir / f"{r['Student_ID'].strip()}.pdf").exists()]
    calib = [f"Student_{i}" for i in (19, 24, 29, 31, 37, 42)]
    test = [f"Student_{i}" for i in (40, 41, 43, 44, 45, 46, 47, 48, 49, 50)]
    dev = [s for s in have if s not in calib and s not in test]
    splits = {"calibration": [s for s in calib if s in have], "test": [s for s in test if s in have],
              "dev": dev, "ablation": dev[:20]}
    for name, members in splits.items():
        (out / f"split_{name}.txt").write_text(",".join(members) + "\n", encoding="utf-8")
    print(f"Wrote {copied} students ({len(human)} question marks) to {out}")
    print("Splits: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()) + "  (split_*.txt)")
    print(f"MCQ letters match the teacher's MCQ total for {copied - len(inconsistent)}/{copied} students; "
          f"MCQ expected marks kept for: {args.mcq}"
          + (f" (left out for {len(inconsistent)}: {', '.join(inconsistent)})" if args.mcq == "consistent" and inconsistent else ""))
    print("  question_paper.txt, rubric.txt, model_answer.txt, human_marks.csv, sheets/, teacher_marked/")
    if problems:
        print(f"{len(problems)} note(s):")
        for p in problems[:20]:
            print("  -", p)
    else:
        print("Short-answer totals match the dataset's own totals.")


if __name__ == "__main__":
    main()
