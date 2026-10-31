"""Segmenter (planner): decides which part of the answer sheet answers which question.

After this, each question is graded on its own and in parallel, so one long
answer can't crowd the others out of the model's context. The output is then
checked by plain code (the grounding check in pipeline.grade_question): every
segmented answer must actually be found in the sheet's transcription, so the
segmenter can't invent an answer for a blank question.

If the model's reply cannot be used, the sheet is split by its question
labels instead ("Q3.", "3)", "प्रश्न 3", "கேள்வி 3") and the sheet is flagged,
so one bad reply does not lose the whole sheet.
"""

import json
import re

from core.llm import LLMError, chat_json
from core.utils import looks_like_grid, normalize_qid

from .prompts import SEGMENT_GRID_HINT, SEGMENT_SYSTEM, SEGMENT_USER

FALLBACK_WARNING = ("The segmenter's reply could not be used, so answers were matched to questions by their "
                    "labels (Q1, 2), प्रश्न 3, ...). Check that each answer went to the right question.")

# A question label at the start of a line: "Q3.", "Q 3)", "3.", "3)", "Ans 3", "प्रश्न 3", "கேள்வி 3", "(3)"
_LABEL = re.compile(r"(?m)^\s*(?:Q(?:uestion)?|Ans(?:wer)?|प्रश्न|उत्तर|கேள்வி)?\s*\(?(\d{1,3}[a-z]?)\s*[.):](?=\s|$)")


def split_by_labels(ids: list[str], sheet_text: str) -> tuple[dict, str]:
    """Split a sheet at its question labels, without a model.

    A label for a question that was already found is kept as part of the
    current answer (it is usually a numbered point such as "1." in a list).

    Args:
        ids (list[str]): Question ids of the rubric, e.g. ["Q1", "Q2a"].
        sheet_text (str): Sheet transcription with [Page n] markers.

    Returns:
        tuple[dict, str]: {question id: {"answer_text", "pages"}} for every
            question (empty if its label was not found), and the text before
            the first label.
    """
    by_norm = {normalize_qid(q): q for q in ids}
    result = {q: {"answer_text": "", "pages": []} for q in ids}
    page, marks = 1, []
    for m in re.finditer(r"\[Page (\d+)\]|" + _LABEL.pattern[4:], sheet_text, flags=re.M):
        if m.group(1):
            page = int(m.group(1))
        else:
            qid = by_norm.get(normalize_qid(m.group(2)))
            if qid and qid not in {q for _, q, _ in marks}:  # a repeated "1." is a numbered point, not Q1
                marks.append((m.start(), qid, page))
    for i, (start, qid, pg) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(sheet_text)
        chunk = re.sub(r"(\s*\[Page \d+\])+\s*$", "", sheet_text[start:end])  # a page break just before the next answer
        text = re.sub(r"\[Page (\d+)\]\s*", "", chunk).strip()
        pages = {pg} | {int(p) for p in re.findall(r"\[Page (\d+)\]", chunk)}
        r = result[qid]
        r["answer_text"] = (r["answer_text"] + "\n\n" + text).strip()
        r["pages"] = sorted(set(r["pages"]) | pages)
    first = marks[0][0] if marks else len(sheet_text)
    return result, re.sub(r"\[Page \d+\]\s*", "", sheet_text[:first]).strip()


def segment(rubric: dict, sheet_text: str, profile: str | None = None) -> tuple[dict, str, list[str]]:
    """Splits the transcribed answer sheet into one answer per question.

    With a single question the whole sheet is its answer and no model is
    called. Text the model gives for the same question twice is joined.

    Args:
        rubric (dict): Validated rubric with "questions".
        sheet_text (str): Sheet transcription with [Page n] markers.
        profile (str | None): Model profile name from config.yaml.

    Returns:
        tuple[dict, str, list[str]]: {question id: {"answer_text", "pages"}}
            for every question, the text matched to no question, and
            warnings.

    A model reply that cannot be used is replaced by `split_by_labels`, with
    FALLBACK_WARNING added to the warnings; no error is raised for it.
    """
    questions = rubric["questions"]
    ids = [q["id"] for q in questions]
    warnings = []

    if len(questions) == 1:  # nothing to split
        return {ids[0]: {"answer_text": sheet_text, "pages": []}}, "", warnings

    q_list = json.dumps([{"id": q["id"], "text": q["text"][:300]} for q in questions],
                        ensure_ascii=False, indent=1)
    messages = [{"role": "system", "content": SEGMENT_SYSTEM},
                {"role": "user", "content": SEGMENT_USER.format(questions=q_list, sheet=sheet_text)
                 + (SEGMENT_GRID_HINT if looks_like_grid(sheet_text) else "")}]
    try:
        out = chat_json("segmenter", messages, profile=profile, max_tokens=12000)
    except LLMError:
        result, unassigned = split_by_labels(ids, sheet_text)
        return result, unassigned, warnings + [FALLBACK_WARNING]

    result = {qid: {"answer_text": "", "pages": []} for qid in ids}
    answers = out.get("answers", []) if isinstance(out, dict) else []
    for a in answers if isinstance(answers, list) else []:
        if not isinstance(a, dict):
            continue
        qid = str(a.get("question_id", "")).strip()
        text = str(a.get("answer_text") or "").strip()
        if qid not in result:
            warnings.append(f"Segmenter returned unknown question '{qid}'; its text was set aside.")
            continue
        if result[qid]["answer_text"] and text:  # same question answered in two places
            result[qid]["answer_text"] += "\n\n" + text
        elif text:
            result[qid]["answer_text"] = text
        pages = a.get("pages") or []
        result[qid]["pages"] = sorted({int(p) for p in pages if str(p).isdigit()} | set(result[qid]["pages"]))
    unassigned = str(out.get("unassigned", "") if isinstance(out, dict) else "")
    return result, unassigned, warnings
