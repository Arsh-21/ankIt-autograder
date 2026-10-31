"""A fake model for the `mock` profile.

It returns well-formed replies for every agent, with no download and no
internet, so the whole program can be run end to end as a check. It does
NOT grade: its rules are fixed (first criterion full marks, the rest half
or zero depending on the sample number) and its "OCR" returns the same
made-up text for every page. Any number produced with this profile is
meaningless.
"""

import json
import re


def _user_text(messages: list[dict]) -> str:
    """Get the text of the last message, joining text parts if it has images.

    Args:
        messages (list[dict]): Chat messages.

    Returns:
        str: Text of the last message.
    """
    content = messages[-1]["content"]
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content


def _tag(text: str, name: str) -> str:
    """Get the contents of the last <name>...</name> block.

    The last block is used because prompts mention the tag names in their
    instructions before the real block.

    Args:
        text (str): Prompt text.
        name (str): Tag name.

    Returns:
        str: Block contents, stripped; "" if the tag is not found.
    """
    start = text.rfind(f"<{name}>")
    if start == -1:
        return ""
    end = text.find(f"</{name}>", start)
    return text[start + len(name) + 2:end if end != -1 else None].strip()


_Q_LINE = re.compile(r"(?m)^[ \t]*(?:Q(?:uestion)?[ \t]*)?(\d+[a-z]?)[ \t]*[.):][ \t]*(.*)$")  # one line only
_MARKS = re.compile(r"\[(\d+(?:\.\d+)?)\s*marks?\]", re.I)


def _mock_rubric(text: str) -> str:
    """Build a two-criterion rubric per question line of the question paper.

    Args:
        text (str): Rubric prompt with <question_paper> and <model_answer> blocks.

    Returns:
        str: JSON {"questions": [...]}.
    """
    paper, model = _tag(text, "question_paper"), _tag(text, "model_answer")
    model_parts = {m.group(1): m.group(2) for m in _Q_LINE.finditer(model)}
    questions = []
    for m in _Q_LINE.finditer(paper):
        num, line = m.group(1), m.group(2)
        mk = _MARKS.search(line)
        total = float(mk.group(1)) if mk else 4.0
        qid = f"Q{num}"
        questions.append({
            "id": qid, "text": _MARKS.sub("", line).strip(), "max_marks": total,
            "model_answer": model_parts.get(num, ""),
            "criteria": [
                {"id": f"{qid}.1", "description": "(mock) states the main idea", "marks": total / 2,
                 "partial_allowed": True},
                {"id": f"{qid}.2", "description": "(mock) gives supporting detail", "marks": total - total / 2,
                 "partial_allowed": True}],
            "notes": "mock rubric"})
    if not questions:
        questions = [{"id": "Q1", "text": paper[:200], "max_marks": 4, "model_answer": model,
                      "criteria": [{"id": "Q1.1", "description": "(mock) correct answer", "marks": 4,
                                    "partial_allowed": True}], "notes": "mock rubric"}]
    return json.dumps({"questions": questions})


def _mock_segment(text: str) -> str:
    """Split the answer sheet at lines that start with a question number.

    Args:
        text (str): Segmenter prompt with <questions> and <answer_sheet> blocks.

    Returns:
        str: JSON {"answers": [...], "unassigned": str}.
    """
    qs = json.loads(_tag(text, "questions") or "[]")
    sheet = re.sub(r"\[Page \d+\]\n?", "", _tag(text, "answer_sheet"))
    found, matches = {}, list(_Q_LINE.finditer(sheet))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sheet)
        found[f"Q{m.group(1)}"] = sheet[m.start():end].strip()
    answers = [{"question_id": q["id"], "answer_text": found.get(q["id"], ""), "pages": [1]} for q in qs]
    first = matches[0].start() if matches else len(sheet)
    return json.dumps({"answers": answers, "unassigned": sheet[:first].strip()})


def _mock_grade(text: str, sample_idx: int) -> str:
    """Grade with fixed rules: first criterion met, the others partial or not met by sample.

    Args:
        text (str): Grader prompt with <criteria_json> and <student_answer> blocks.
        sample_idx (int): Sample number; odd samples give 0 for later criteria.

    Returns:
        str: JSON {"criteria", "feedback", "addressed_to_grader", "addressed_text"}.
    """
    criteria = json.loads(_tag(text, "criteria_json") or "[]")
    answer = _tag(text, "student_answer")
    clean = re.sub(r"\[removed:[^\]]*\]", "", answer)
    quote = " ".join(clean.split()[:8])
    out = []
    for i, c in enumerate(criteria):
        if not clean.strip() or clean.strip() == "(no answer written)":
            v, mk = "not_met", 0
        elif i == 0:
            v, mk = "met", c["marks"]
        elif sample_idx % 2 == 1:
            v, mk = "not_met", 0
        else:
            v, mk = "partial", c["marks"] / 2
        out.append({"criterion_id": c["id"], "verdict": v, "marks_awarded": mk,
                    "evidence": quote if mk else "", "reason": "(mock) placeholder reason"})
    addressed = "full marks" in answer.lower() or "dear sir" in answer.lower()
    return json.dumps({"criteria": out, "feedback": "(mock) Feedback would explain what was missing.",
                       "addressed_to_grader": addressed,
                       "addressed_text": "(mock) request for marks" if addressed else ""})


def _mock_tools(text: str) -> str:
    """Make one calc call for code answers, then finish.

    Args:
        text (str): Tool-agent prompt with <history> and <student_answer> blocks.

    Returns:
        str: JSON {"thought", "action", "input"}.
    """
    history, answer = _tag(text, "history").strip(), _tag(text, "student_answer")
    if history in ("", "(none yet)") and ("def " in answer or "```" in answer):
        return json.dumps({"thought": "(mock) check one sum", "action": "calc", "input": "(32 - 5 + 2*1)/1 + 1"})
    return json.dumps({"thought": "(mock) nothing else to check", "action": "finish", "input": ""})


def _mock_recheck(text: str) -> str:
    """Report one fixed OCR discrepancy ("= 28" read for "= 30") when present.

    Args:
        text (str): Re-check prompt with a <transcription> block.

    Returns:
        str: JSON {"discrepancies": [...], "corrected_answer": str}.
    """
    answer = _tag(text, "transcription")
    if "= 28" in answer:  # pretend the 28 was really a 30 on the page
        return json.dumps({"discrepancies": [{"transcribed": "= 28", "actually_written": "= 30",
                                              "confidence": "high", "affects_deduction": True}],
                           "corrected_answer": answer.replace("= 28", "= 30")})
    return json.dumps({"discrepancies": [], "corrected_answer": ""})


def mock_response(role: str, messages: list[dict], sample_idx: int = 0) -> str:
    """Return a fixed, well-formed reply for an agent role.

    Args:
        role (str): Agent role, e.g. "ocr", "rubric", "grader".
        messages (list[dict]): Chat messages sent to the model.
        sample_idx (int): Sample number, used by the grader rules.

    Returns:
        str: Reply text in the format the role expects.
    """
    text = _user_text(messages)
    if role == "ocr":
        page = re.search(r"page (\d+)", text)
        n = page.group(1) if page else "?"
        return (f"Name: Mock Student (mock OCR of page {n})\n"
                "Q1. Vanishing gradients happen when gradients shrink as they pass back through many layers.\n"
                "Q2. Output size = (32 - 5 + 2*1)/1 + 1 = 28\n"
                "Q3. [STRUCK: def soft(x)] [illegible]")
    if role == "rubric":
        return _mock_rubric(text)
    if role == "segmenter":
        return _mock_segment(text)
    if role in ("grader", "grader_vision"):
        return _mock_grade(text, sample_idx % 1000)
    if role == "tools":
        return _mock_tools(text)
    if role == "recheck":
        return _mock_recheck(text)
    if role == "locator":
        qids = re.findall(r"(?m)^- (\S+)$", text)
        return json.dumps({"positions": [{"question_id": q, "y": 150 + 750 * i // max(1, len(qids))}
                                         for i, q in enumerate(qids)]})
    if role == "critic":
        return "- (mock) Criterion 2 was graded inconsistently across attempts."
    if role == "summary":
        return "(mock) Overall feedback paragraph would go here."
    if role == "class_summary":
        return "(mock) ~3 students: common mistake would be listed here."
    return "(mock) ok"
