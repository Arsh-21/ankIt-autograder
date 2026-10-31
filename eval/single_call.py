"""Single-call baseline: the whole answer sheet graded in one model call.

The model gets the rubric (questions, criteria, marks, model answers) and every
page image of the sheet, and returns a mark for each question. No OCR, no
segmenting, no per-question grading, no checks. It measures whether the
pipeline does better than simply asking a vision model to mark the sheet.
Used only by eval/run_ablation.py (variant `baseline_single_call`).

Usage:
    report = single_call_grade(rubric, sheet, profile="openrouter")
    # or through the ablation runner:
    python eval/run_ablation.py --dataset eval/datasets/demo --only baseline_single_call
"""

import json

from core.ingest import Document
from core.llm import chat_json, image_part
from core.utils import clamp, normalize_qid, round_half, to_float

SYSTEM = "You are an exam grader. You mark a student's answer sheet against the given rubric."

USER = """Mark this answer sheet. The rubric gives every question, its marking criteria, its maximum marks and a model answer.

<rubric>
{rubric}
</rubric>

{sheet_note}

Give every question a mark between 0 and its maximum, in half marks. A question the student did not answer gets 0.
Return only JSON:
{{"marks": [{{"question_id": "Q1", "marks": 0}}]}}"""


def single_call_grade(rubric: dict, sheet: Document, profile: str | None) -> dict:
    """Grade a whole sheet with one model call.

    Args:
        rubric (dict): Validated rubric, {"questions": [{"id", "text", "max_marks", "model_answer",
            "criteria"}, ...]}.
        sheet (Document): The student's sheet; page images are used when present, otherwise its text.
        profile (str | None): Model profile name from config.yaml (None = the default profile).

    Returns:
        dict: A small report shaped like grade_sheet's output, enough for the ablation
            runner: "questions" (list of {"id", "awarded", "max_marks", "needs_review",
            "review_reasons"}), "total", "max_total" and "needs_review" (always False).
            A question missing from the reply gets 0.

    Raises:
        LLMError: If the model call fails.
    """
    slim = [{"id": q["id"], "question": q["text"], "max_marks": q["max_marks"],
             "model_answer": q.get("model_answer", ""),
             "criteria": [{"description": c["description"], "marks": c["marks"]} for c in q["criteria"]]}
            for q in rubric["questions"]]
    images = [p.image for p in sheet.pages if p.image is not None]
    if images:
        note = f"The student's answer sheet is in the {len(images)} attached page image(s)."
        content = [{"type": "text", "text": USER.format(rubric=json.dumps(slim, ensure_ascii=False, indent=1),
                                                        sheet_note=note)}]
        content += [image_part(img) for img in images]
    else:
        note = f"The student's answer sheet:\n<answer_sheet>\n{sheet.full_text()}\n</answer_sheet>"
        content = USER.format(rubric=json.dumps(slim, ensure_ascii=False, indent=1), sheet_note=note)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    raw = chat_json("grader_vision" if images else "grader", messages, profile=profile, max_tokens=8000)

    given = {}
    for item in (raw.get("marks", []) if isinstance(raw, dict) else []):
        if isinstance(item, dict):
            given[normalize_qid(str(item.get("question_id", "")))] = to_float(item.get("marks"))
    questions = []
    for q in rubric["questions"]:
        mark = clamp(round_half(given.get(normalize_qid(q["id"]), 0.0)), 0, q["max_marks"])
        questions.append({"id": q["id"], "awarded": mark, "max_marks": q["max_marks"],
                          "needs_review": False, "review_reasons": []})
    total = sum(q["awarded"] for q in questions)
    return {"questions": questions, "total": total,
            "max_total": sum(q["max_marks"] for q in rubric["questions"]), "needs_review": False}
