"""Aggregator: combines N independent grading samples into one grade. No model calls.

Per criterion, the N sampled marks are combined by `aggregation`:
  strictness - percentile set by strictness: lenient 75th, standard median,
               strict 25th (config.yaml)
  mean / median / max / min
Design note: "max" is not the default. Human-in-the-loop maths grading
(arXiv 2603.13083) took the max of 5 passes "to protect students", and it
pushed marks above the human range. The spread between samples becomes a
confidence score and a review trigger, as in the engineering-exam pipeline
(arXiv 2601.00730).
"""

import numpy as np

from core.config import thresholds
from core.utils import clamp, round_half


def combine(values: list[float], method: str, strictness: str) -> float:
    """Combines the sampled marks for one criterion into a single value.

    Args:
        values (list[float]): Marks given by each sample.
        method (str): "mean", "median", "max", "min" or "strictness".
        strictness (str): "lenient", "standard" or "strict"; picks the
            percentile when method is "strictness".

    Returns:
        float: The combined mark (0.0 for an empty list).
    """
    if not values:
        return 0.0
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "max":
        return float(max(values))
    if method == "min":
        return float(min(values))
    pct = thresholds()["strictness_percentile"][strictness]
    return float(np.percentile(values, pct))


def aggregate(q: dict, samples: list[dict], method: str = "strictness",
              strictness: str = "standard") -> dict:
    """Combines the grading samples for one question into a final grade.

    Each criterion is combined separately, rounded to half marks and clamped
    to its maximum. Feedback and reasons come from the sample whose total is
    closest to the final total.

    Args:
        q (dict): Rubric question with "max_marks" and "criteria".
        samples (list[dict]): Normalised grader outputs (see grader.normalize).
        method (str): Combination method passed to `combine`.
        strictness (str): Strictness level passed to `combine`.

    Returns:
        dict: The final grade, with keys "awarded", "max_marks",
            "sample_totals", "spread", "confidence", "rejections",
            "evidence_flags", "rubric_note", "addressed_to_grader",
            "addressed_text", "criteria" (list of per-criterion dicts) and
            "feedback".
    """
    max_marks = q["max_marks"]
    totals = [s["total"] for s in samples]
    spread = (max(totals) - min(totals)) if totals else 0.0

    final_crit = {}
    for c in q["criteria"]:
        marks = [s["criteria"][c["id"]]["marks"] for s in samples]
        final_crit[c["id"]] = clamp(round_half(combine(marks, method, strictness)), 0, c["marks"])
    final_total = sum(final_crit.values())

    # explanation comes from the sample whose total is closest to the final one
    chosen = min(samples, key=lambda s: abs(s["total"] - final_total)) if samples else None
    criteria_out = []
    for c in q["criteria"]:
        cid = c["id"]
        src = chosen["criteria"][cid] if chosen else {}
        if final_crit[cid] > 0 and not src.get("marks"):
            for s in samples:  # borrow the quote from a sample that gave marks here
                if s["criteria"][cid]["marks"] > 0:
                    src = s["criteria"][cid]
                    break
        criteria_out.append({
            "id": cid, "description": c["description"], "max": c["marks"], "awarded": final_crit[cid],
            "verdict": "met" if final_crit[cid] >= c["marks"] else "partial" if final_crit[cid] > 0 else "not_met",
            "evidence": src.get("evidence", "") if final_crit[cid] > 0 else "",
            "reason": src.get("reason", ""),
        })
    return {
        "awarded": final_total,
        "max_marks": max_marks,
        "sample_totals": totals,
        "spread": spread,
        "confidence": round(1 - spread / max_marks, 2) if max_marks else 1.0,
        "rejections": sum(s["rejections"] for s in samples),
        "evidence_flags": sum(s["evidence_flags"] for s in samples),
        "rubric_note": next((s.get("rubric_note", "") for s in samples if s.get("rubric_note")), ""),
        "addressed_to_grader": any(s["addressed_to_grader"] for s in samples),
        "addressed_text": next((s["addressed_text"] for s in samples if s["addressed_text"]), ""),
        "criteria": criteria_out,
        "feedback": chosen["feedback"] if chosen else "",
    }


def disagreed(q: dict, samples: list[dict]) -> bool:
    """Checks whether the sample totals are spread too far apart.

    Args:
        q (dict): Rubric question with "max_marks".
        samples (list[dict]): Normalised grader outputs.

    Returns:
        bool: True if the spread of totals exceeds the disagreement threshold
            (a fraction of max marks, from config.yaml).
    """
    totals = [s["total"] for s in samples]
    spread = (max(totals) - min(totals)) if totals else 0
    return spread > thresholds()["disagreement_threshold"] * q["max_marks"]


def needs_reflection(q: dict, samples: list[dict]) -> bool:
    """Decides whether the critic should review this question's samples.

    Args:
        q (dict): Rubric question with "max_marks".
        samples (list[dict]): Normalised grader outputs.

    Returns:
        bool: True if the samples disagreed, or any sample had a mark removed
            or flagged by the evidence check.
    """
    return disagreed(q, samples) or any(s["rejections"] or s["evidence_flags"] for s in samples)
