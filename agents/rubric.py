"""Rubric agent: question paper + rubric + model answer -> structured criteria.

Modes (setting `rubric_mode`):
  as_is  - only STRUCTURE the TA's rubric: one criterion per marking point,
           same words, same marks. Nothing is split or reworded.
  expand - make vague points concrete and split marks into small steps.
  auto   - as_is when a rubric is given, expand when there is only a model
           answer (or nothing).

Design notes: the code needs structured criteria (ids, marks that add up)
to check quotes per criterion and combine samples, so some parsing is always
needed. Expanding is a judgement call: Human-in-the-loop maths grading
(arXiv 2603.13083) found small explicit steps work best, but Grade Like a
Human (arXiv 2405.19694) found finer rubrics hurt on simple questions.
So it is a setting, and an ablation.

The rubric is built once, before any student answer is seen, and can be
edited before grading. It never changes while grading, so a run of wrong
answers can't drag it off course.
"""

import json

from core.llm import chat_json
from core.utils import clamp, infer_dependencies, to_float

from .prompts import RUBRIC_AS_IS_USER, RUBRIC_EXPAND_USER, RUBRIC_SYSTEM

NOT_PROVIDED = "(not provided)"
# Used only when there is no question paper, so prompts with a question paper are unchanged.
NO_PAPER = ("(not provided: take the questions, their ids and their marks from the marking scheme and the "
            "model answer)")


def resolve_mode(mode: str, rubric_text: str) -> str:
    """Resolves the "auto" rubric mode.

    Args:
        mode (str): "auto", "as_is" or "expand".
        rubric_text (str): The TA's rubric text ("" if none).

    Returns:
        str: "as_is" if a rubric was given, else "expand"; any mode other
            than "auto" is returned unchanged.
    """
    if mode == "auto":
        return "as_is" if rubric_text.strip() else "expand"
    return mode


def build_rubric(question_paper: str, rubric: str = "", model_answer: str = "",
                 mode: str = "auto", profile: str | None = None) -> tuple[dict, str]:
    """Asks the model to turn the exam documents into rubric JSON.

    Args:
        question_paper (str): Question paper text.
        rubric (str): Marking scheme text, or "".
        model_answer (str): Model answer text, or "".
        mode (str): "auto", "as_is" or "expand".
        profile (str | None): Model profile name from config.yaml.

    Returns:
        tuple[dict, str]: The raw rubric JSON from the model (not yet
            validated) and the mode actually used.

    Raises:
        LLMError: If the model call fails or the reply is not valid JSON.
    """
    mode = resolve_mode(mode, rubric)
    template = RUBRIC_AS_IS_USER if mode == "as_is" else RUBRIC_EXPAND_USER
    messages = [
        {"role": "system", "content": RUBRIC_SYSTEM},
        {"role": "user", "content": template.format(
            question_paper=question_paper.strip() or NO_PAPER,
            rubric=rubric.strip() or NOT_PROVIDED,
            model_answer=model_answer.strip() or NOT_PROVIDED)},
    ]
    return chat_json("rubric", messages, profile=profile, max_tokens=16000), mode


def _clean_examples(raw: list | None, max_marks: float) -> list[dict]:
    """Cleans the optional staff-graded examples of a question.

    Examples are added by hand in the rubric editor as
    "examples": [{"answer": "...", "marks": 2, "comment": "..."}].

    Args:
        raw (list | None): The question's "examples" value.
        max_marks (float): Maximum marks for the question; example marks are
            clamped to it.

    Returns:
        list[dict]: Examples with "answer", "marks" and "comment"; entries
            with no answer text are dropped.
    """
    out = []
    for ex in raw or []:
        if not isinstance(ex, dict) or not str(ex.get("answer", "")).strip():
            continue
        out.append({"answer": str(ex["answer"]).strip(),
                    "marks": clamp(to_float(ex.get("marks")), 0, max_marks),
                    "comment": str(ex.get("comment", "") or "").strip()})
    return out


def validate_rubric(rub: dict | list | str) -> tuple[dict, list[str]]:
    """Checks and repairs a rubric, from the model or from manual edits.

    Fixes duplicate ids, drops criteria with no marks, adds one overall
    criterion where none exist, makes max_marks match the criteria total,
    and fills in "depends_on" (set by hand, or inferred from the question
    text).

    Args:
        rub (dict | list | str): Rubric as a dict with "questions", a list of
            questions, or a JSON string of either.

    Returns:
        tuple[dict, list[str]]: The clean rubric {"questions": [...]} and a
            list of warnings about what was changed.

    Raises:
        ValueError: If the rubric is not valid JSON or has no questions.
    """
    if isinstance(rub, str):
        try:
            rub = json.loads(rub)
        except json.JSONDecodeError as e:
            raise ValueError(f"The rubric is not valid JSON: {e}") from e
    if isinstance(rub, list):
        rub = {"questions": rub}
    qs = rub.get("questions") if isinstance(rub, dict) else None
    if not qs:
        raise ValueError("The rubric has no questions.")

    warnings, seen, clean = [], set(), []
    for i, q in enumerate(qs, start=1):
        qid = str(q.get("id") or f"Q{i}").strip()
        if qid in seen:
            qid = f"{qid}_{i}"
            warnings.append(f"Duplicate question id renamed to {qid}.")
        seen.add(qid)

        crits = []
        for j, c in enumerate(q.get("criteria") or [], start=1):
            marks = to_float(c.get("marks"))
            if marks <= 0:
                continue
            crits.append({"id": str(c.get("id") or f"{qid}.{j}"),
                          "description": str(c.get("description", "")).strip(),
                          "marks": marks,
                          "partial_allowed": bool(c.get("partial_allowed", True))})
        crit_sum = sum(c["marks"] for c in crits)
        max_marks = to_float(q.get("max_marks"), crit_sum)
        if not crits:
            crits = [{"id": f"{qid}.1", "description": "Correct and complete answer",
                      "marks": max_marks or 1, "partial_allowed": True}]
            crit_sum = crits[0]["marks"]
            warnings.append(f"{qid}: no criteria found, using one overall criterion.")
        if max_marks <= 0:
            max_marks = crit_sum
        if abs(crit_sum - max_marks) > 1e-6:
            warnings.append(f"{qid}: criteria add up to {crit_sum:g} but max marks is {max_marks:g}. "
                            f"Using {crit_sum:g}; edit the rubric if that's wrong.")
            max_marks = crit_sum
        notes = str(q.get("notes", "") or "")
        if "derived" in notes.lower():
            warnings.append(f"{qid}: criteria were derived by the model, not taken from your rubric.")
        clean.append({"id": qid, "text": str(q.get("text", "")).strip(), "max_marks": max_marks,
                      "model_answer": str(q.get("model_answer", "") or ""), "criteria": crits,
                      "notes": notes, "examples": _clean_examples(q.get("examples"), max_marks),
                      "depends_on": [str(d) for d in (q.get("depends_on") or []) if str(d).strip()]})
    # questions that build on earlier ones: kept if the teacher set them, else read from the question text
    ids = {q["id"] for q in clean}
    inferred = infer_dependencies(clean)
    for q in clean:
        q["depends_on"] = [d for d in q["depends_on"] if d in ids and d != q["id"]] or inferred.get(q["id"], [])
    return {"questions": clean}, warnings
