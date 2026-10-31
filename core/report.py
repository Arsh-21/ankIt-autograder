"""Writes grading results to files.

For each sheet: a Markdown report, the full JSON (with every raw grading
sample, for offline ablations), the answer-sheet transcript and the
annotated PDF. For a batch: a marks CSV, class analytics and a zip of all
reports."""

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path

from .annotate import annotate, save_pdf
from .config import ROOT
from .ingest import Document


def _f(x: float) -> str:
    """Format a number without trailing zeros (2.0 -> "2").

    Args:
        x (float): Number.

    Returns:
        str: Short form of x.
    """
    return f"{x:g}"


def _cell(s: str, n: int = 200) -> str:
    """Make text safe for a Markdown table cell.

    Args:
        s (str): Text; None is treated as empty.
        n (int): Maximum length.

    Returns:
        str: Text on one line with "|" escaped, cut to n characters.
    """
    return (s or "").replace("\n", " ").replace("|", "\\|")[:n]


def usage_line(u: dict) -> str:
    """Summarise model usage in one line.

    Args:
        u (dict): Output of Usage.as_dict().

    Returns:
        str: Calls, cache hits, tokens and cost.
    """
    cost = f"${u['cost_usd']:.4f} (₹{u['cost_inr']:.2f})"
    if u.get("calls_with_unknown_cost"):
        cost += f", {u['calls_with_unknown_cost']} call(s) with unknown price"
    return (f"Model calls: {u['api_calls']} (+{u['cache_hits']} from cache) · tokens in/out: "
            f"{u['prompt_tokens']}/{u['completion_tokens']} · cost {cost}")


def to_markdown(report: dict) -> str:
    """Render one sheet's report as Markdown.

    Args:
        report (dict): Sheet report from the pipeline.

    Returns:
        str: Summary table, then per-question criteria, feedback and checks.
    """
    s = report["meta"]["settings"]
    out = [f"# Result: {_f(report['total'])} / {_f(report['max_total'])}", ""]
    out.append(f"_Answer sheet: {report['meta']['answer_sheet']} · profile: {s['profile']} · strictness: "
               f"{s['strictness']} · samples: {s['n_samples']} · grader input: {s['grader_input']}_")
    if report["needs_review"]:
        out += ["", "> **Needs a human check.** See the flagged questions below."]
    for r in report["sheet_review_reasons"]:
        out.append(f"> - {r}")
    out += ["", "## Overall feedback", "", report["overall_feedback"] or "_none_", ""]

    out += ["## Summary", "", "| Question | Marks | Confidence | Check? | Why |", "|---|---|---|---|---|"]
    for q in report["questions"]:
        out.append(f"| {q['id']} | {_f(q['awarded'])} / {_f(q['max_marks'])} | {q['confidence']:.2f} | "
                   f"{'**yes**' if q['needs_review'] else ('checked by TA' if q.get('human_reviewed') else 'no')} | {_cell('; '.join(q['review_reasons']), 300)} |")
    out.append("")

    for q in report["questions"]:
        out += [f"## {q['id']}: {_f(q['awarded'])} / {_f(q['max_marks'])}", "", f"**Question:** {q['text']}", ""]
        if q["review_reasons"]:
            out += ["**Check:** " + "; ".join(q["review_reasons"]), ""]
        if q["info"]:
            out += ["**Notes:** " + "; ".join(q["info"]), ""]
        out += ["| Criterion | Marks | Evidence from the answer | Reason |", "|---|---|---|---|"]
        for c in q["criteria"]:
            ev = _cell(c["evidence"])
            out.append(f"| {_cell(c['description'], 300)} | {_f(c['awarded'])} / {_f(c['max'])} | "
                       f"{('“' + ev + '”') if ev else '–'} | {_cell(c['reason'], 300)} |")
        out += ["", f"**Feedback:** {q['feedback'] or '–'}", ""]
        if q["injection_findings"]:
            out += ["**Text aimed at the grader:**"] + [f"- ({f['kind']}) “{f['text']}”" for f in q["injection_findings"]] + [""]
        if q["recheck"] and q["recheck"]["discrepancies"]:
            out += ["<details><summary>OCR re-check</summary>", ""]
            for d in q["recheck"]["discrepancies"]:
                out.append(f"- read “{d.get('transcribed', '')}”, written “{d.get('actually_written', '')}” "
                           f"({d.get('confidence', '?')} confidence)")
            out += ["", "</details>", ""]
        if q["tool_observations"]:
            out += ["<details><summary>Tool checks</summary>", ""]
            for o in q["tool_observations"]:
                out += [f"- Thought: {o.get('thought', '')}", f"  Action: `{o['tool']}`", "```", o["input"][:800], "```",
                        "Observation:", "```", o["output"][:800], "```"]
            out += ["</details>", ""]
        if q["critique"]:
            out += ["<details><summary>Critic's notes</summary>", "", q["critique"], "", "</details>", ""]
        out += ["<details><summary>Student's answer (as read)</summary>", "", "```",
                q["answer_text"] or "(empty)", "```", "</details>", ""]

    if (report.get("unassigned_text") or "").strip():
        out += ["## Text not matched to any question", "", "```", report["unassigned_text"], "```", ""]
    if report["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    out.append(f"_{usage_line(report['usage'])}_")
    return "\n".join(out)


def _new_folder(prefix: str = "") -> Path:
    """Create a time-stamped folder under outputs/.

    Args:
        prefix (str): Prefix for the folder name, e.g. "batch_".

    Returns:
        Path: The new folder.
    """
    folder = ROOT / "outputs" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_report(report: dict, sheet: Document, folder: Path | None = None,
                annotated: list | None = None) -> dict:
    """Write report.md, report.json, the transcript and the annotated PDF.

    Args:
        report (dict): Sheet report from the pipeline.
        sheet (Document): The answer sheet.
        folder (Path | None): Output folder; a new one under outputs/ if None.
        annotated (list | None): Annotated page images; drawn here if None and
            annotation is on in the report's settings.

    Returns:
        dict: {"report_md", "report_json", "transcript_md", optional "annotated_pdf"} -> path str.
    """
    folder = folder or _new_folder()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"report_md": folder / "report.md", "report_json": folder / "report.json",
             "transcript_md": folder / "answer_sheet_transcript.md"}
    paths["report_md"].write_text(to_markdown(report), encoding="utf-8")
    paths["report_json"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["transcript_md"].write_text(sheet.transcript_markdown(), encoding="utf-8")
    if annotated is None and report["meta"]["settings"].get("annotate", True):
        annotated = annotate(report, sheet)
    if annotated:
        paths["annotated_pdf"] = Path(save_pdf(annotated, folder / "annotated_sheet.pdf"))
    return {k: str(v) for k, v in paths.items()}


def analytics_markdown(batch: dict) -> str:
    """Render class analytics as Markdown.

    Args:
        batch (dict): Batch result with "analytics", "results" and "usage".

    Returns:
        str: Class averages, a per-question table and the most-missed criteria.
    """
    a = batch["analytics"]
    out = [f"# Class results: {a['students']} sheets graded", "",
           f"Average total {a['mean_total']:.2f} / {a['max_total']:g} (median {a['median_total']:g}). "
           f"{a['sheets_needing_review']} sheet(s) need a human check.", ""]
    failed = [r for r in batch["results"] if r["error"]]
    if failed:
        out += ["**Could not grade:** " + "; ".join(f"{r['student']} ({r['error'][:120]})" for r in failed), ""]
    out += ["| Question | Average | Std | Full marks | Zero | Not attempted | Check |", "|---|---|---|---|---|---|---|"]
    for q in a["questions"]:
        out.append(f"| {q['id']} | {q['mean']:.2f} / {q['max_marks']:g} | {q['std']:.2f} | {q['full_pct']}% | "
                   f"{q['zero_pct']}% | {q['not_attempted']} | {q['needs_review']} |")
    out.append("")
    for q in a["questions"]:
        out += [f"## {q['id']}", "", f"_{q['text'][:300]}_", "", "**Most-missed criteria**", ""]
        for c in q["criteria"][:5]:
            out.append(f"- {c['missed_pct']}% missed: {c['description']}")
        if q["common_mistakes"]:
            out += ["", "**Common mistakes**", "", q["common_mistakes"]]
        out.append("")
    out.append(f"_{usage_line(batch['usage'])}_")
    return "\n".join(out)


def save_batch(batch: dict) -> dict:
    """Write the batch outputs: marks CSV, class analytics and all reports as a zip.

    Args:
        batch (dict): Batch result with "rubric", "results", "analytics" and "usage".

    Returns:
        dict: {"marks_csv", "analytics_md", "reports_zip"} -> path str.
    """
    folder = _new_folder("batch_")
    qids = [q["id"] for q in batch["rubric"]["questions"]]
    csv_path = folder / "marks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student"] + qids + ["total", "max_total", "needs_review", "review_reasons", "error"])
        for r in batch["results"]:
            rep = r["report"]
            if not rep:
                w.writerow([r["student"]] + [""] * len(qids) + ["", "", "yes", "", r["error"]])
                continue
            by = {q["id"]: q for q in rep["questions"]}
            reasons = rep["sheet_review_reasons"] + [f"{q['id']}: {x}" for q in rep["questions"] for x in q["review_reasons"]]
            w.writerow([r["student"]] + [_f(by[q]["awarded"]) for q in qids] +
                       [_f(rep["total"]), _f(rep["max_total"]), "yes" if rep["needs_review"] else "no",
                        " | ".join(reasons), ""])
    analytics_path = folder / "class_analytics.md"
    analytics_path.write_text(analytics_markdown(batch), encoding="utf-8")
    zip_path = folder / "all_reports.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r in batch["results"]:
            if r["report"]:
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in r["student"])
                sub = save_report(r["report"], r["sheet"], folder / "reports" / safe)
                for p in sub.values():
                    z.write(p, arcname=f"{safe}/{Path(p).name}")
    return {"marks_csv": str(csv_path), "analytics_md": str(analytics_path), "reports_zip": str(zip_path)}
