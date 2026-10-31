"""Run the grading pipeline from the terminal, without the web UI.

Usage:
    # One sheet (several files = pages of that sheet, in order):
    python cli.py --qp samples/dl_quiz/exam/question_paper_scan.pdf --rubric samples/dl_quiz/exam/marking_scheme.pdf \
        --sheet samples/dl_quiz/sheets/rahul.pdf --profile mock

    # A whole class (each file, or each folder inside a zip, is one student):
    python cli.py --qp ... --rubric ... --batch sheets.zip

    # A rubric that was already checked or edited (skips the rubric agent):
    python cli.py --rubric-json my_rubric.json --sheet ...

    # Any setting from config.yaml `defaults:` can be overridden with --set:
    python cli.py ... --set n_samples=5 --set grader_input=text --set guard_detector=off
"""

import argparse
import json
import sys
import tempfile

import yaml

from agents.pipeline import build_rubric_stage, grade_batch, grade_sheet, read_sheet
from core.batch_io import collect_students
from core.llm import LLMError, start_usage
from core.report import save_batch, save_report, usage_line
from core.settings import RunSettings


def parse_sets(pairs: list[str]) -> dict:
    """Parse --set KEY=VALUE pairs into a settings dict.

    Values are read as YAML, so "3" becomes 3, "true" becomes True and "text"
    stays a string.

    Args:
        pairs (list[str]): Strings of the form "key=value".

    Returns:
        dict: {key: parsed value}.

    Raises:
        ValueError: If a pair has no "=".
    """
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--set needs key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)  # "3" -> 3, "true" -> True, "text" -> "text"
    return out


def main() -> None:
    """Build or load the rubric, grade one sheet or a batch, and save the reports."""
    ap = argparse.ArgumentParser(description="Grade answer sheets.")
    ap.add_argument("--qp", nargs="+", help="question paper file(s)")
    ap.add_argument("--rubric", nargs="*", default=[], help="rubric / marking scheme file(s)")
    ap.add_argument("--model-answer", nargs="*", default=[], help="model answer file(s)")
    ap.add_argument("--rubric-json", help="use this rubric JSON as-is (skip the rubric agent)")
    ap.add_argument("--sheet", nargs="+", help="ONE student's answer sheet file(s)")
    ap.add_argument("--batch", nargs="+", help="many students: files and/or zips")
    ap.add_argument("--profile", help="profile from config.yaml (mock, local, ollama_cloud, openrouter)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any setting")
    ap.add_argument("--save-rubric", help="write the drafted rubric JSON here and stop")
    args = ap.parse_args()

    def progress(frac: float, msg: str) -> None:
        """Print a progress line.

        Args:
            frac (float): Fraction done, 0-1.
            msg (str): What is happening now.
        """
        print(f"[{frac * 100:5.1f}%] {msg}", flush=True)

    try:
        settings = RunSettings.from_config(profile=args.profile, **parse_sets(args.set))
        usage = start_usage()
        if args.rubric_json:
            with open(args.rubric_json, encoding="utf-8") as f:
                rubric = json.load(f)
        else:
            if not (args.qp or args.rubric or args.model_answer):
                ap.error("give at least one of --qp, --rubric, --model-answer (or --rubric-json)")
            stage = build_rubric_stage(args.qp or [], args.rubric, args.model_answer, settings, progress)
            rubric = stage["rubric"]
            print(f"\nRubric ({stage['rubric_mode_used']}) has {len(rubric['questions'])} question(s).")
            for w in stage["warnings"]:
                print("  warning:", w)
        if args.save_rubric:
            with open(args.save_rubric, "w", encoding="utf-8") as f:
                json.dump(rubric, f, indent=2, ensure_ascii=False)
            print(f"Saved rubric to {args.save_rubric}. Edit it, then pass it with --rubric-json.")
            return

        if args.batch:
            with tempfile.TemporaryDirectory() as tmp:
                students = collect_students(args.batch, tmp)
                print(f"\n{len(students)} student(s) found.")
                batch = grade_batch(rubric, students, settings, progress)
                files = save_batch(batch)
            a = batch["analytics"]
            print(f"\nClass average {a['mean_total']:.2f} / {a['max_total']:g}; "
                  f"{a['sheets_needing_review']} sheet(s) need a check.")
        elif args.sheet:
            sheet = read_sheet(args.sheet, settings, progress)
            report = grade_sheet(rubric, sheet, settings, progress)
            files = save_report(report, sheet)
            print(f"\nTotal: {report['total']:g} / {report['max_total']:g}"
                  f"{'   (needs a human check)' if report['needs_review'] else ''}")
            for q in report["questions"]:
                print(f"  {q['id']}: {q['awarded']:g} / {q['max_marks']:g}  conf {q['confidence']:.2f}"
                      + (f"  CHECK: {'; '.join(q['review_reasons'])}" if q["needs_review"] else ""))
        else:
            ap.error("give --sheet or --batch (or --save-rubric)")
    except (LLMError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + usage_line(usage.as_dict()))
    print("Saved:")
    for k, v in files.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
