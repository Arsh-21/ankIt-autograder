"""Build samples/mendeley/: a real exam for the app's sample menus, from the
Mendeley dataset you imported with scripts/import_mendeley.py.

Dataset: "A Dataset of Digitized Student Examination Papers, Answer Keys, and
Manual Evaluations for Automated Grading Research", Dinesh K P, Mendeley Data,
CC BY 4.0, https://data.mendeley.com/datasets/sf3kvjwknt/1. The licence allows
sharing with credit; the credit is written into samples/mendeley/README.md.

Picks six students spread across the teacher's marks (lowest to highest on the
short answers), avoiding the calibration and held-out test students, and
writes:

  exam/question_paper.txt, marking_scheme.txt, model_answer.txt
  sheets/<student>.pdf          the scanned answer sheets, shrunk to 1600 px
  teacher_marked/<student>.pdf  the teacher's marked copies, to compare by eye
  class.zip                     the six sheets
  expected_marks.csv            the teacher's marks
  students.csv                  who is in the sample and their teacher totals
  sample.json                   tells the app what to show in its sample menus

Usage:
    python scripts/make_mendeley_samples.py --dataset eval/datasets/mendeley
"""

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "mendeley"
CREDIT = ("Dinesh K P, \"A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual "
          "Evaluations for Automated Grading Research\", Mendeley Data, V1, "
          "https://data.mendeley.com/datasets/sf3kvjwknt/1 (CC BY 4.0)")


def shrink_pdf(src: Path, dst: Path, long_side: int = 1600, quality: int = 80) -> int:
    """Re-save a scanned PDF with every page shrunk to a JPEG of the given size.

    The grader never sees more than 1600 px, so this keeps what matters and
    makes a 35-megapixel scan small enough for a git repository.

    Args:
        src (Path): The original PDF.
        dst (Path): Where to write the smaller PDF.
        long_side (int): Long side of each page image, in pixels.
        quality (int): JPEG quality.

    Returns:
        int: Number of pages written.
    """
    out = pymupdf.open()
    for page in pymupdf.open(src):
        dpi = max(30, round(72 * long_side / max(page.rect.width, page.rect.height)))
        pix = page.get_pixmap(dpi=dpi)
        buf = io.BytesIO()
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(buf, format="JPEG", quality=quality)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    out.save(dst, garbage=4, deflate=True)
    return len(out)


def pick_spread(totals: dict[str, float], k: int) -> list[str]:
    """Pick k students spread evenly from the lowest to the highest total.

    Args:
        totals (dict[str, float]): Total mark per student.
        k (int): How many to pick.

    Returns:
        list[str]: The chosen students, lowest total first.
    """
    order = sorted(totals, key=lambda s: (totals[s], s))
    if len(order) <= k:
        return order
    idx = sorted({round(i * (len(order) - 1) / (k - 1)) for i in range(k)})
    return [order[i] for i in idx]


def main() -> None:
    """Write samples/mendeley/ from the imported dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/datasets/mendeley", help="output folder of import_mendeley.py")
    ap.add_argument("--students", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated students to use instead of the automatic pick")
    args = ap.parse_args()
    ds = Path(args.dataset)
    if not (ds / "human_marks.csv").exists():
        raise SystemExit(f"{ds} has no human_marks.csv. Run scripts/import_mendeley.py first.")

    marks: dict[str, dict[str, float]] = {}
    with open(ds / "human_marks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            marks.setdefault(r["sheet"], {})[r["question_id"]] = float(r["marks"])
    held_out = set()
    for split in ("calibration", "test"):
        p = ds / f"split_{split}.txt"
        if p.exists():
            held_out |= {s.strip() for s in p.read_text(encoding="utf-8").split(",") if s.strip()}
    # short answers (Q21-Q35) are marked for every student; MCQ marks are missing for some
    short = {s: sum(v for q, v in m.items() if int(q.lstrip("Q")) >= 21) for s, m in marks.items()
             if s not in held_out and (ds / "sheets" / f"{s}.pdf").exists()}
    chosen = [s.strip() for s in args.only.split(",")] if args.only else pick_spread(short, args.students)

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("exam", "sheets", "teacher_marked"):
        (OUT / sub).mkdir(parents=True)
    shutil.copy(ds / "question_paper.txt", OUT / "exam" / "question_paper.txt")
    shutil.copy(ds / "rubric.txt", OUT / "exam" / "marking_scheme.txt")
    shutil.copy(ds / "model_answer.txt", OUT / "exam" / "model_answer.txt")

    rows, info = [], []
    for s in chosen:
        n = shrink_pdf(ds / "sheets" / f"{s}.pdf", OUT / "sheets" / f"{s}.pdf")
        marked = ds / "teacher_marked" / f"{s}.pdf"
        if marked.exists():
            shrink_pdf(marked, OUT / "teacher_marked" / f"{s}.pdf", long_side=1400, quality=70)
        rows += [[s, q, v] for q, v in sorted(marks[s].items(), key=lambda kv: int(kv[0].lstrip("Q")))]
        sa = sum(v for q, v in marks[s].items() if int(q.lstrip("Q")) >= 21)
        has_mcq = any(int(q.lstrip("Q")) <= 20 for q in marks[s])
        info.append([s, n, sa, sum(marks[s].values()) if has_mcq else ""])
        print(f"  {s}: {n} pages, teacher's short-answer marks {sa:g}/30")

    with zipfile.ZipFile(OUT / "class.zip", "w") as z:
        for s in chosen:
            z.write(OUT / "sheets" / f"{s}.pdf", arcname=f"{s}.pdf")
    with open(OUT / "expected_marks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "question_id", "marks"])
        w.writerows(rows)
    with open(OUT / "students.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "pages", "teacher_short_answers_of_30", "teacher_total_of_50"])
        w.writerows(info)

    manifest = {  # read by app.py to fill the sample menus
        "name": "Real exam: Data Science test (Mendeley, handwritten)",
        "order": 2,
        "exam": {"question_paper": ["exam/question_paper.txt"], "marking_scheme": ["exam/marking_scheme.txt"],
                 "model_answer": ["exam/model_answer.txt"]},
        "sheets": {f"{s}: real scan, teacher gave {sa:g}/30 on short answers": [f"sheets/{s}.pdf"]
                   for s, _, sa, _ in info},
        "class": "class.zip",
    }
    (OUT / "sample.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = ["# Real exam: Data Science test (Mendeley)", "",
             "Real scanned, handwritten answer sheets with the teacher's own marks. Part I is Q1-Q20 "
             "(multiple choice, 1 mark each), Part II is Q21-Q35 (short answers, 2 marks each), 50 in total.", "",
             "| Student | Pages | Teacher: short answers (of 30) | Teacher: total (of 50) |", "|---|---|---|---|"]
    lines += [f"| {s} | {n} | {sa:g} | {tot if tot == '' else f'{tot:g}'} |" for s, n, sa, tot in info]
    lines += ["", "A blank total means the dataset's MCQ letters and MCQ total disagree for that student, so "
              "only the short-answer marks are used (see `scripts/import_mendeley.py`).", "",
              "`teacher_marked/` has the teacher's marked copies, to compare with the app's marked pages.", "",
              f"Source: {CREDIT}. Pages are shrunk to 1600 px; nothing else is changed.",
              "", "Rebuild with `python scripts/make_mendeley_samples.py`."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(chosen)} students to {OUT}")


if __name__ == "__main__":
    main()
