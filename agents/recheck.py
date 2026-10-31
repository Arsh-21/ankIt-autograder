"""OCR re-check agent, based on EDU-CIRCUIT-HW (arXiv 2602.00095).

Misread handwriting mostly hurts students: a misread "+" or digit makes a
right answer look wrong. So, only for answers that lost marks, a vision
model compares the transcription with the page image, focusing on what the
deductions depend on:
  * high-confidence misreading that affects a deduction -> use the corrected
    transcription and re-grade (the paper's "LLM re-grader" path)
  * low-confidence misreading -> flag for a human (the paper sent ~3-4% to TAs)
In the paper this raised point-level agreement by 4.5 to 8.9 points.
"""

from PIL import Image

from core.llm import chat_json, image_part

from .prompts import RECHECK_SYSTEM, RECHECK_USER


def deductions_text(q: dict, criteria_out: list[dict]) -> str:
    """Lists the criteria that lost marks, for the re-check prompt.

    Args:
        q (dict): Rubric question (not used; kept for the call signature).
        criteria_out (list[dict]): Aggregated criteria with "id", "awarded",
            "max", "description" and "reason".

    Returns:
        str: One line per criterion below full marks, or "(none)".
    """
    lines = []
    for c in criteria_out:
        if c["awarded"] < c["max"]:
            lines.append(f"- {c['id']} ({c['awarded']:g}/{c['max']:g}) {c['description']}: {c['reason']}")
    return "\n".join(lines) or "(none)"


def recheck(q: dict, answer: str, criteria_out: list[dict], images: list[Image.Image],
            profile: str | None = None) -> dict:
    """Compares the transcription of an answer with its page images.

    Args:
        q (dict): Rubric question with "id" and "text".
        answer (str): The transcribed answer that was graded.
        criteria_out (list[dict]): Aggregated criteria from the grade.
        images (list[Image.Image]): Page images of the answer.
        profile (str | None): Model profile name from config.yaml.

    Returns:
        dict: Keys "discrepancies" (list of dicts with "transcribed",
            "actually_written", "confidence", "affects_deduction") and
            "corrected_answer" ("" if nothing needs correcting).

    Raises:
        LLMError: If the model call fails or the reply is not valid JSON.
    """
    user = RECHECK_USER.format(qid=q["id"], question=q["text"],
                               deductions=deductions_text(q, criteria_out), answer=answer)
    messages = [{"role": "system", "content": RECHECK_SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": user}]
                 + [image_part(img) for img in images]}]
    out = chat_json("recheck", messages, profile=profile, max_tokens=6000)
    if not isinstance(out, dict):
        return {"discrepancies": [], "corrected_answer": ""}
    disc = [d for d in out.get("discrepancies") or [] if isinstance(d, dict)]
    return {"discrepancies": disc, "corrected_answer": str(out.get("corrected_answer") or "").strip()}
