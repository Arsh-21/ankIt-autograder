"""Human review: a TA accepts or changes the marks of a question.

Design note: the papers this project builds on keep a human in charge of
final marks (the handwritten-maths study had a human verify every grade).
Each change is logged in the report (who, when, old and new marks), so the
record is kept in the JSON and the CSV."""

from datetime import datetime

from .utils import clamp, round_half


def _recompute(report: dict) -> dict:
    """Update a sheet's total and review flag after a change.

    Args:
        report (dict): Sheet report, changed in place.

    Returns:
        dict: The same report.
    """
    report["total"] = sum(q["awarded"] for q in report["questions"])
    report["needs_review"] = sheet_note_pending(report) or any(q["needs_review"] for q in report["questions"])
    return report


def sheet_note_pending(report: dict) -> bool:
    """Whether the sheet has a whole-sheet note (e.g. text matched to no question) nobody has checked yet.

    Args:
        report (dict): Sheet report.

    Returns:
        bool: True if there is such a note and it is not checked.
    """
    return bool(report.get("sheet_review_reasons")) and not report.get("sheet_checked")


def unaccept(report: dict, qid: str, reviewer: str = "TA") -> dict:
    """Undo "checked" on a question (its marks stay as they are); it is flagged again if it had reasons.

    Args:
        report (dict): Sheet report, changed in place.
        qid (str): Question id.
        reviewer (str): Name stored in the review log.

    Returns:
        dict: The updated report.
    """
    q = next(x for x in report["questions"] if x["id"] == qid)
    if q.get("human_reviewed"):
        q["human_reviewed"] = False
        q["needs_review"] = bool(q.get("review_reasons"))
        q.setdefault("review_log", []).append({"by": reviewer, "at": datetime.now().isoformat(timespec="seconds"),
                                               "action": "marked as not checked", "awarded": q["awarded"]})
    return _recompute(report)


def unaccept_sheet(report: dict) -> dict:
    """Undo "checked" on the whole-sheet notes.

    Args:
        report (dict): Sheet report, changed in place.

    Returns:
        dict: The updated report.
    """
    report["sheet_checked"] = False
    return _recompute(report)


def accept_sheet(report: dict, reviewer: str = "TA") -> dict:
    """Mark the whole-sheet notes (not the questions) as checked.

    Args:
        report (dict): Sheet report, changed in place.
        reviewer (str): Name stored in the review log.

    Returns:
        dict: The updated report.
    """
    if sheet_note_pending(report):
        report["sheet_checked"] = True
        report.setdefault("review_log", []).append({"by": reviewer, "at": datetime.now().isoformat(timespec="seconds"),
                                                   "action": "accepted the sheet notes"})
    return _recompute(report)


def accept(report: dict, qid: str, reviewer: str = "TA") -> dict:
    """Mark a flagged question as checked without changing its marks.

    Args:
        report (dict): Sheet report, changed in place.
        qid (str): Question id.
        reviewer (str): Name stored in the review log.

    Returns:
        dict: The updated report.

    Raises:
        StopIteration: If the report has no question with this id.
    """
    q = next(x for x in report["questions"] if x["id"] == qid)
    q["human_reviewed"] = True
    q["needs_review"] = False
    q.setdefault("review_log", []).append({"by": reviewer, "at": datetime.now().isoformat(timespec="seconds"),
                                           "action": "accepted", "awarded": q["awarded"]})
    return _recompute(report)


def override(report: dict, qid: str, criterion_marks: dict[str, float], feedback: str | None = None,
             reviewer: str = "TA") -> dict:
    """Set new marks per criterion and log the change.

    Marks are rounded to the nearest half and clamped to each criterion's
    maximum. Criteria not in criterion_marks keep their marks.

    Args:
        report (dict): Sheet report, changed in place.
        qid (str): Question id.
        criterion_marks (dict[str, float]): Criterion id -> new marks.
        feedback (str | None): New feedback text; None or blank keeps the old one.
        reviewer (str): Name stored in the review log.

    Returns:
        dict: The updated report.

    Raises:
        StopIteration: If the report has no question with this id.
    """
    q = next(x for x in report["questions"] if x["id"] == qid)
    before = q["awarded"]
    for c in q["criteria"]:
        if c["id"] in criterion_marks:
            new = clamp(round_half(float(criterion_marks[c["id"]])), 0, c["max"])
            if new != c["awarded"]:
                c["awarded"] = new
                c["verdict"] = "met" if new >= c["max"] else "partial" if new > 0 else "not_met"
                c["reason"] = (c.get("reason") or "") + " [changed by reviewer]"
    if feedback is not None and feedback.strip() and feedback != q["feedback"]:
        q["feedback"] = feedback.strip()
    q["awarded"] = sum(c["awarded"] for c in q["criteria"])
    q["human_reviewed"] = True
    q["needs_review"] = False
    q.setdefault("review_log", []).append({"by": reviewer, "at": datetime.now().isoformat(timespec="seconds"),
                                           "action": "changed", "from": before, "to": q["awarded"]})
    return _recompute(report)
