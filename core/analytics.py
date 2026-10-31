"""Class-level statistics for a batch (no AI): per-question averages, how
many sheets got full or zero marks, and which rubric criteria were missed most."""

import numpy as np


def class_analytics(rubric: dict, reports: list[dict]) -> dict:
    """Compute class statistics from the graded sheets of a batch.

    Args:
        rubric (dict): Rubric with "questions", each with "id", "text", "max_marks" and "criteria".
        reports (list[dict]): Sheet reports from the pipeline.

    Returns:
        dict: Keys "students", "max_total", "mean_total", "median_total",
            "sheets_needing_review" and "questions". Each question entry has "id",
            "text", "max_marks", "n", "mean", "std", "full_pct", "zero_pct",
            "not_attempted", "needs_review", "criteria" (sorted by "missed_by",
            most missed first) and "common_mistakes" (filled in later by the
            class-summary agent).
    """
    questions = []
    for q in rubric["questions"]:
        rows = [x for r in reports for x in r["questions"] if x["id"] == q["id"]]
        marks = [x["awarded"] for x in rows]
        n = len(rows)
        crit = []
        for c in q["criteria"]:
            missed = sum(1 for x in rows for cc in x["criteria"] if cc["id"] == c["id"] and cc["awarded"] < cc["max"])
            crit.append({"id": c["id"], "description": c["description"], "missed_by": missed,
                         "missed_pct": round(100 * missed / n, 1) if n else 0.0})
        crit.sort(key=lambda c: -c["missed_by"])
        questions.append({
            "id": q["id"], "text": q["text"], "max_marks": q["max_marks"], "n": n,
            "mean": float(np.mean(marks)) if marks else 0.0,
            "std": float(np.std(marks)) if marks else 0.0,
            "full_pct": round(100 * sum(m >= q["max_marks"] for m in marks) / n, 1) if n else 0.0,
            "zero_pct": round(100 * sum(m == 0 for m in marks) / n, 1) if n else 0.0,
            "not_attempted": sum(1 for x in rows if "Not attempted" in x["info"]),
            "needs_review": sum(1 for x in rows if x["needs_review"]),
            "criteria": crit,
            "common_mistakes": "",
        })
    totals = [r["total"] for r in reports]
    return {
        "students": len(reports),
        "max_total": sum(q["max_marks"] for q in rubric["questions"]),
        "mean_total": float(np.mean(totals)) if totals else 0.0,
        "median_total": float(np.median(totals)) if totals else 0.0,
        "sheets_needing_review": sum(1 for r in reports if r["needs_review"]),
        "questions": questions,
    }
