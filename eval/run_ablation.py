"""Run ablations: grade the same data under several settings and compare with human marks.

Each variant in the variants file changes one thing from the base settings.

Usage:
    python eval/run_ablation.py --dataset eval/datasets/demo --variants eval/variants.yaml --profile mock
    python eval/run_ablation.py --dataset eval/sample_eval.csv --variants eval/variants.yaml --only n1,n1_ref_none
    python eval/run_ablation.py --dataset eval/datasets/mendeley --profile openrouter --only n1,n1_ref_none \
        --exclude @eval/datasets/mendeley/split_test.txt --sample 10 --seed 0

--sample N grades a random subset of N sheets (folder datasets only). The pick
depends only on N, --seed and the sheet names left after --sheets / --exclude,
so separate runs with the same options grade the same students.

Two kinds of dataset:

1) A folder of answer sheets (e.g. handwritten scans):
     question_paper.*       required (any format the app reads)
     rubric.*               optional marking scheme
     model_answer.*         optional
     rubric.json            optional frozen rubric (variants with rubric_source: file)
     sheets/                one file per student, or one sub-folder of pages per student
     human_marks.csv        columns: sheet, question_id, marks
                            (sheet = file or folder name without extension;
                             question_id like Q1 / 1 / Q2a; use TOTAL for totals only)

2) A CSV of single answers (e.g. Mohler/Texas converted), columns:
     id, question, max_marks, student_answer, human_score
     optional: model_answer, rubric

Output in eval/runs/<time>/: summary.md + summary.csv (one row per variant),
items_<variant>.csv (every graded answer), questions_<variant>.jsonl (full
results incl. raw samples, for eval/offline.py), sheets_<variant>.jsonl.
OCR results are reused across variants (memory + disk cache), so only the
first variant pays for reading the scans.
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.pipeline import build_rubric_stage, grade_question, grade_sheet, read_sheet  # noqa: E402
from agents.rubric import build_rubric, validate_rubric  # noqa: E402
from core.config import resolve_profile_name  # noqa: E402
from core.ingest import Document  # noqa: E402
from core.llm import LLMError, start_usage  # noqa: E402
from core.report import save_report  # noqa: E402
from core.settings import RunSettings  # noqa: E402
from core.utils import normalize_qid  # noqa: E402
from eval.metrics import compute, fmt  # noqa: E402
from eval.single_call import single_call_grade  # noqa: E402

DOC_EXT = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------- dataset loading
def _find(folder: Path, stem: str) -> list[str]:
    """List the document files in a folder whose name (without extension) is `stem`.

    Args:
        folder (Path): Folder to search (not recursive).
        stem (str): Lower-case file name without extension, e.g. "rubric".

    Returns:
        list[str]: Matching file paths, sorted.
    """
    return sorted(str(p) for p in folder.iterdir()
                  if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in DOC_EXT)


def load_sheet_dataset(folder: Path) -> dict:
    """Load a folder dataset of answer sheets (layout in the module docstring).

    Args:
        folder (Path): The dataset folder.

    Returns:
        dict: "kind" ("sheets"), "qp", "rubric", "model_answer" (file lists),
            "rubric_json" (dict or None), "students" (list of (name, page files))
            and "human" ({sheet: {normalised qid: marks}}).

    Raises:
        ValueError: If the folder has no sheets/ sub-folder.
    """
    sheets_dir = folder / "sheets"
    if not sheets_dir.is_dir():
        raise ValueError(f"{folder} has no sheets/ folder")
    students = []
    for e in sorted(sheets_dir.iterdir()):
        if e.is_dir():
            pages = sorted(str(f) for f in e.iterdir() if f.is_file() and f.suffix.lower() in DOC_EXT)
            if pages:
                students.append((e.name, pages))
        elif e.suffix.lower() in DOC_EXT:
            students.append((e.stem, [str(e)]))
    human: dict[str, dict[str, float]] = {}
    with open(folder / "human_marks.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human.setdefault(row["sheet"].strip(), {})[normalize_qid(row["question_id"])] = float(row["marks"])
    rjson = folder / "rubric.json"
    return {"kind": "sheets", "qp": _find(folder, "question_paper"), "rubric": _find(folder, "rubric"),
            "model_answer": _find(folder, "model_answer"),
            "rubric_json": json.loads(rjson.read_text(encoding="utf-8")) if rjson.exists() else None,
            "students": students, "human": human}


def load_csv_dataset(path: Path) -> dict:
    """Load a CSV dataset of single typed answers.

    Args:
        path (Path): The CSV file.

    Returns:
        dict: "kind" ("csv") and "rows" (list of row dicts).
    """
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {"kind": "csv", "rows": rows}


def sample_students(students: list[tuple[str, list[str]]], n: int,
                    seed: int = 0) -> list[tuple[str, list[str]]]:
    """Pick n sheets at random, the same ones for the same n and seed.

    The names are sorted before sampling, so the pick does not depend on the
    order of the input list. The picked sheets keep their input order.

    Args:
        students (list[tuple[str, list[str]]]): (sheet name, page files) pairs,
            as in load_sheet_dataset's "students".
        n (int): Number of sheets to pick.
        seed (int): Seed for random.Random.

    Returns:
        list[tuple[str, list[str]]]: The picked pairs, in their input order.

    Raises:
        SystemExit: If n is below 1 or larger than the number of sheets.
    """
    if n < 1:
        raise SystemExit(f"--sample must be at least 1, got {n}")
    if n > len(students):
        raise SystemExit(f"--sample {n} is more than the {len(students)} sheet(s) available")
    picked = set(random.Random(seed).sample(sorted(name for name, _ in students), n))
    return [s for s in students if s[0] in picked]


# ---------------------------------------------------------------- running one variant
def parse_questions(spec: str | None) -> set[str] | None:
    """Parse a question selection such as '21-35' or '1,2,5-7'.

    Args:
        spec (str | None): Comma-separated ids and numeric ranges.

    Returns:
        set[str] | None: Normalised question ids, e.g. {'21', ..., '35'}; None
            (all questions) when spec is empty.
    """
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out |= {str(i) for i in range(int(a), int(b) + 1)}
        elif part:
            out.add(normalize_qid(part))
    return out


def get_rubric(ds: dict, variant: dict, settings: RunSettings, rubric_cache: dict,
               keep: set[str] | None, out_dir: Path) -> dict:
    """Build (or reuse) the rubric for a variant and keep only the chosen questions.

    The rubric is also written to out_dir/rubric_<variant>.json.

    Args:
        ds (dict): Sheet dataset from load_sheet_dataset.
        variant (dict): Variant entry from the variants file ("name", optional
            "inputs" and "rubric_source").
        settings (RunSettings): Settings for this variant.
        rubric_cache (dict): Rubrics already built, keyed by (inputs, rubric_mode, profile).
            Updated in place.
        keep (set[str] | None): Normalised question ids to keep; None keeps all.
        out_dir (Path): Run output folder.

    Returns:
        dict: The rubric, {"questions": [...]}.

    Raises:
        ValueError: If rubric_source is "file" and the dataset has no rubric.json,
            or if `keep` matches none of the rubric's questions.
    """
    inputs = variant.get("inputs", ["rubric", "model_answer"])
    if variant.get("rubric_source", "agent") == "file":
        if not ds["rubric_json"]:
            raise ValueError("rubric_source: file needs rubric.json in the dataset folder")
        rubric, _ = validate_rubric(ds["rubric_json"])
    else:
        key = (tuple(sorted(inputs)), settings.rubric_mode, settings.profile)
        if key not in rubric_cache:  # variants that differ only in grading reuse the same rubric
            rubric_cache[key] = build_rubric_stage(ds["qp"], ds["rubric"] if "rubric" in inputs else [],
                                                   ds["model_answer"] if "model_answer" in inputs else [],
                                                   settings)["rubric"]
        rubric = rubric_cache[key]
    if keep:
        rubric = {"questions": [q for q in rubric["questions"] if normalize_qid(q["id"]) in keep]}
        if not rubric["questions"]:
            raise ValueError("--questions matched none of the rubric's question ids")
    (out_dir / f"rubric_{variant['name']}.json").write_text(json.dumps(rubric, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")
    return rubric


def run_sheet_variant(ds: dict, variant: dict, settings: RunSettings, profile: str, limit: int | None,
                      sheet_cache: dict, out_dir: Path, rubric_cache: dict | None = None,
                      keep: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Grade every sheet of a folder dataset under one variant.

    Writes questions_<variant>.jsonl, sheets_<variant>.jsonl and (except for the
    single-call baseline) a report folder per sheet. Sheets without human marks
    or whose grading fails are skipped.

    Args:
        ds (dict): Sheet dataset from load_sheet_dataset.
        variant (dict): Variant entry from the variants file.
        settings (RunSettings): Settings for this variant.
        profile (str): Model profile name.
        limit (int | None): Grade at most this many sheets; None = all.
        sheet_cache (dict): OCR results shared across variants, keyed by
            (sheet, force_ocr, ocr_backend, language_hint). Updated in place.
        out_dir (Path): Run output folder.
        rubric_cache (dict | None): Rubrics shared across variants; None = no sharing.
        keep (set[str] | None): Normalised question ids to grade; None = all.

    Returns:
        tuple[list[dict], list[dict]]: (items, sheets). Items are per-answer rows
            ("variant", "sheet", "qid", "pred", "human", "max", "flagged", "reasons");
            sheets are per-sheet totals ("pred", "human", "max", "flagged").
    """
    rubric = get_rubric(ds, variant, settings, {} if rubric_cache is None else rubric_cache, keep, out_dir)
    items, sheets = [], []
    qfile = open(out_dir / f"questions_{variant['name']}.jsonl", "w", encoding="utf-8")
    sfile = open(out_dir / f"sheets_{variant['name']}.jsonl", "w", encoding="utf-8")
    for name, files in ds["students"][:limit]:
        human = ds["human"].get(name)
        if human is None:
            print(f"  (no human marks for {name}, skipped)")
            continue
        key = (name, settings.force_ocr, settings.ocr_backend, settings.language_hint)  # same OCR -> reuse it
        single = variant.get("mode") == "single_call"
        try:
            if key not in sheet_cache:
                sheet_cache[key] = read_sheet(files, settings)
            if single:
                rep = single_call_grade(rubric, sheet_cache[key], profile)
            else:
                rep = grade_sheet(rubric, sheet_cache[key], settings, overall_feedback=False)
        except (LLMError, ValueError) as e:
            print(f"  {name}: FAILED {e}")
            continue
        for q, qr in zip(rubric["questions"], rep["questions"], strict=True):
            h = human.get(normalize_qid(q["id"]))
            if h is None:
                continue
            items.append({"variant": variant["name"], "sheet": name, "qid": q["id"], "pred": qr["awarded"],
                          "human": h, "max": q["max_marks"], "flagged": qr["needs_review"],
                          "reasons": "; ".join(qr["review_reasons"])})
            qfile.write(json.dumps({"sheet": name, "qid": q["id"], "human": h, "scale": 1.0,
                                    "question": q, "result": qr}, ensure_ascii=False) + "\n")
        h_total = sum(human.get(normalize_qid(q["id"]), 0) for q in rubric["questions"])
        if "total" in human and not keep:
            h_total = human["total"]
        sheets.append({"pred": rep["total"], "human": h_total, "max": rep["max_total"], "flagged": rep["needs_review"]})
        sfile.write(json.dumps({"sheet": name, "report": rep}, ensure_ascii=False, default=str) + "\n")
        print(f"  {name}: {rep['total']:g} (human {h_total:g})")
        if not single:
            save_report(rep, sheet_cache[key], out_dir / "reports" / variant["name"] / name)
    qfile.close()
    sfile.close()
    return items, sheets


def run_csv_variant(ds: dict, variant: dict, settings: RunSettings, profile: str, limit: int | None,
                    out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Grade every row of a CSV dataset under one variant.

    Each row is graded as a one-question exam (Q1) with its own rubric; rows with
    the same question and references share a rubric. Marks are rescaled to the
    row's max_marks. Writes questions_<variant>.jsonl; failed rows are skipped.

    Args:
        ds (dict): CSV dataset from load_csv_dataset.
        variant (dict): Variant entry from the variants file.
        settings (RunSettings): Settings for this variant.
        profile (str): Model profile name.
        limit (int | None): Grade at most this many rows; None = all.
        out_dir (Path): Run output folder.

    Returns:
        tuple[list[dict], list[dict]]: (items, []). Items have the same keys as in
            run_sheet_variant; there are no per-sheet totals.
    """
    inputs = variant.get("inputs", ["rubric", "model_answer"])
    rubrics, items = {}, []
    empty = Document(name="csv", kind="text")
    qfile = open(out_dir / f"questions_{variant['name']}.jsonl", "w", encoding="utf-8")
    for r in ds["rows"][:limit]:
        mx = float(r["max_marks"])
        ma = r.get("model_answer", "") if "model_answer" in inputs else ""
        rb = r.get("rubric", "") if "rubric" in inputs else ""
        key = (r["question"], ma, rb, mx)
        try:
            if key not in rubrics:
                raw, _ = build_rubric(f"Q1. {r['question']} [{mx:g} marks]", rb,
                                      f"Q1. {ma}" if ma else "", settings.rubric_mode, profile)
                rubrics[key] = validate_rubric(raw)[0]["questions"][0] | {"id": "Q1"}
            q = rubrics[key]
            qr = grade_question(q, {"answer_text": r["student_answer"], "pages": []}, empty,
                                r["student_answer"], settings, profile)
        except (LLMError, ValueError) as e:
            print(f"  row {r.get('id')}: FAILED {e}")
            continue
        scale = mx / q["max_marks"] if q["max_marks"] else 1.0
        pred = qr["awarded"] * scale
        items.append({"variant": variant["name"], "sheet": r.get("id", ""), "qid": "Q1", "pred": pred,
                      "human": float(r["human_score"]), "max": mx, "flagged": qr["needs_review"],
                      "reasons": "; ".join(qr["review_reasons"])})
        qfile.write(json.dumps({"sheet": r.get("id", ""), "qid": "Q1", "human": float(r["human_score"]),
                                "scale": scale, "question": q, "result": qr}, ensure_ascii=False) + "\n")
    qfile.close()
    return items, []


# ---------------------------------------------------------------- main
def main() -> None:
    """Run the chosen variants on a dataset and save the per-variant results and summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset folder or CSV")
    ap.add_argument("--variants", default="eval/variants.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--limit", type=int, default=None, help="max sheets / rows")
    ap.add_argument("--questions", default=None, help="grade only these questions, e.g. 21-35 or 1,3,5-7")
    ap.add_argument("--sheets", default=None,
                    help="only these sheets: Student_1,Student_10 or @file.txt (e.g. @eval/datasets/mendeley/split_ablation.txt)")
    ap.add_argument("--exclude", default=None, help="skip these sheets (same forms as --sheets)")
    ap.add_argument("--groups", default=None,
                    help='extra result rows per group of questions, e.g. "MCQ=1-20,Short=21-35"')
    ap.add_argument("--sample", type=int, default=None,
                    help="grade a random subset of N sheets (after --sheets / --exclude)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --sample")
    args = ap.parse_args()

    profile = resolve_profile_name(args.profile)
    spec = yaml.safe_load(open(args.variants, encoding="utf-8"))
    base = spec.get("base") or {}
    variants = spec["variants"]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        variants = [v for v in variants if v["name"] in keep]

    path = Path(args.dataset)
    ds = load_csv_dataset(path) if path.suffix.lower() == ".csv" else load_sheet_dataset(path)
    if ds["kind"] == "sheets" and (args.sheets or args.exclude):
        def names(spec: str | None) -> set[str]:
            """Parse sheet names given as "A,B,C" or "@file.txt" (commas or new lines).

            Args:
                spec (str | None): The --sheets or --exclude value.

            Returns:
                set[str]: The sheet names; empty when spec is None or empty.
            """
            if spec and spec.startswith("@"):
                spec = Path(spec[1:]).read_text(encoding="utf-8")
            return {x.strip() for x in (spec or "").replace("\n", ",").split(",") if x.strip()}
        only, skip = names(args.sheets), names(args.exclude)
        ds["students"] = [(n, f) for n, f in ds["students"] if (not only or n in only) and n not in skip]
        missing = only - {n for n, _ in ds["students"]} - skip
        if missing:
            raise SystemExit(f"Not in the dataset: {', '.join(sorted(missing))}")
    if args.sample is not None:
        if ds["kind"] != "sheets":
            raise SystemExit("--sample works only with a folder dataset (use --limit for a CSV)")
        ds["students"] = sample_students(ds["students"], args.sample, args.seed)
        print(f"Sampled {args.sample} sheet(s) with seed {args.seed}: {', '.join(n for n, _ in ds['students'])}")
    out_dir = Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variants_used.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")

    sheet_cache, rubric_cache, summary = {}, {}, []
    keep = parse_questions(args.questions)
    groups = {}
    for part in (args.groups or "").split(","):
        if "=" in part:
            gname, spec = part.split("=", 1)
            groups[gname.strip()] = parse_questions(spec.replace(";", ","))
    for v in variants:
        settings = RunSettings.from_config(profile=profile, **(base | (v.get("settings") or {})))
        print(f"\n=== {v['name']} ===")
        usage = start_usage()
        t0 = time.time()
        if ds["kind"] == "csv":
            items, sheets = run_csv_variant(ds, v, settings, profile, args.limit, out_dir)
        else:
            items, sheets = run_sheet_variant(ds, v, settings, profile, args.limit, sheet_cache, out_dir,
                                              rubric_cache, keep)
        u = usage.as_dict()
        m = compute(items)
        row = {"variant": v["name"], **{k: m.get(k) for k in
               ["n", "MAE", "bias", "nMAE", "QWK", "within_1", "review", "flag_recall", "flag_precision"]}}
        if sheets:
            mt = compute(sheets)
            row.update(total_MAE=mt["MAE"], total_bias=mt["bias"], sheet_review=mt["review"])
        row.update(calls=u["api_calls"], cache_hits=u["cache_hits"], tokens_in=u["prompt_tokens"],
                   tokens_out=u["completion_tokens"], cost_usd=u["cost_usd"], seconds=round(time.time() - t0, 1))
        summary.append(row)
        for gname, ids in groups.items():  # same metrics on a subset of questions
            sub = [i for i in items if normalize_qid(i["qid"]) in ids]
            mg = compute(sub)
            summary.append({"variant": f"{v['name']} [{gname}]", **{k: mg.get(k) for k in
                            ["n", "MAE", "bias", "nMAE", "QWK", "within_1", "review", "flag_recall", "flag_precision"]}})
        with open(out_dir / f"items_{v['name']}.csv", "w", newline="", encoding="utf-8") as f:
            if items:
                w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
                w.writeheader()
                w.writerows(items)
        print("  " + "  ".join(f"{k}={fmt(row[k])}" for k in ["MAE", "bias", "QWK", "review", "cost_usd"] if k in row))

    keys = list(dict.fromkeys(k for r in summary for k in r))
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    md = [f"# Ablation run {out_dir.name}", "", f"Dataset: `{args.dataset}` · profile: `{profile}`", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in summary]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
