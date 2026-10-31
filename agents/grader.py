"""Grader agent: grades one answer against one question's criteria.

What the grader sees (setting `grader_input`):
  text  - the transcription only
  image - the page image(s) only
  both  - image(s) + transcription  (default for scanned pages)
Design note: EDU-CIRCUIT-HW (arXiv 2602.00095) found that even the best model
misread the handwriting on ~38% of solutions, and those errors silently
change grades; the engineering-exam (arXiv 2601.00730) and maths HITL
(arXiv 2603.13083) pipelines grade from the image itself. Ablation C tests
this.

After the model replies, plain code cleans the result (`normalize`):
  * "met" = full marks, "not_met" = 0, partial is clamped and rounded to 0.5
  * evidence check: every awarded mark needs a quote that really appears in
    the (non-crossed-out) answer; otherwise reject / flag / off
  * a quote that comes from text aimed at the grader never earns marks
"""

import json
import re

from PIL import Image

from core import guard
from core.config import thresholds
from core.guard import Finding
from core.llm import chat_json, image_part
from core.settings import RunSettings
from core.utils import clamp, evidence_supported, remove_struck, round_half, to_float

from .prompts import (
    ANSWER_BOTH,
    ANSWER_IMAGE,
    ANSWER_TEXT,
    ATTEMPT_CREDIT_RULE,
    CODE_RULE,
    CRITIQUE_BLOCK,
    EXAMPLES_BLOCK,
    GRADER_HEAD,
    GRADER_OUTPUT_GUARDED,
    GRADER_OUTPUT_PLAIN,
    GRADER_SYSTEM,
    GUARD_BOUNDARY,
    GUARD_RULE,
    GUARD_SYSTEM,
    JUDGEMENT_RULES,
    LINKED_BLOCK,
    STRICTNESS_POLICY,
    TOOL_BLOCK,
)


def _examples_text(q: dict) -> str:
    """Formats the staff-graded examples of a question for the grader prompt.

    Args:
        q (dict): Rubric question with "max_marks" and optional "examples".

    Returns:
        str: The examples, one block each, or "" if there are none.
    """
    parts = []
    for i, ex in enumerate(q.get("examples") or [], start=1):
        line = f"Example {i} (awarded {ex['marks']:g}/{q['max_marks']:g}):\n{ex['answer']}"
        if ex.get("comment"):
            line += f"\nStaff comment: {ex['comment']}"
        parts.append(line)
    return "\n\n".join(parts)


_CODE_RE = re.compile(r"```|^\s*(def |class |import |for .+:|while .+:|if .+:|#include|public static|print\()", re.M)


def _policy_rules(q: dict, settings: RunSettings, answer: str = "") -> str:
    """Builds the optional extra grading rules for the prompt.

    The code rule is added only when the answer looks like code, so prompts
    for other answers stay unchanged.

    Args:
        q (dict): Rubric question with "max_marks".
        settings (RunSettings): Run settings (grading_style, attempt_credit).
        answer (str): The student's answer, checked for code.

    Returns:
        str: Zero or more rule lines, each ending in a newline.
    """
    rules = JUDGEMENT_RULES if settings.grading_style == "judgement" else ""
    rules += CODE_RULE if _CODE_RE.search(answer or "") else ""
    if settings.attempt_credit > 0:
        floor = round_half(settings.attempt_credit * q["max_marks"])
        if floor > 0:
            rules += ATTEMPT_CREDIT_RULE.format(floor=floor, max_marks=q["max_marks"])
    return rules


def _output_format(settings: RunSettings) -> str:
    """Picks the JSON output spec for the grader prompt.

    Args:
        settings (RunSettings): Run settings (guard_prompt, grading_style).

    Returns:
        str: The output spec; with the "judgement" style it also asks for a
            "rubric_note" field.
    """
    out = GRADER_OUTPUT_GUARDED if settings.guard_prompt else GRADER_OUTPUT_PLAIN
    if settings.grading_style == "judgement":  # ask for the rubric disagreement note too
        out = out.replace('"feedback": "...",', '"feedback": "...", "rubric_note": "",', 1) \
                 .replace('"feedback": "..."}}', '"feedback": "...", "rubric_note": ""}}', 1)
    return out


def build_messages(q: dict, answer: str, settings: RunSettings, images: list[Image.Image], mode: str,
                   tool_text: str = "", critique: str = "", feedback_language: str = "English",
                   linked: str = "") -> list[dict]:
    """Builds the chat messages for one grading call.

    Args:
        q (dict): Rubric question with "id", "text", "max_marks", "criteria",
            and optional "model_answer" and "examples".
        answer (str): The student's answer text.
        settings (RunSettings): Run settings that shape the prompt.
        images (list[Image.Image]): Page images; attached only in "image" or
            "both" mode.
        mode (str): What the grader sees: "text", "image" or "both".
        tool_text (str): Formatted tool observations, or "".
        critique (str): Critic output for a re-grade, or "".
        feedback_language (str): Language the feedback should be written in.
        linked (str): Answers to linked questions with a note on how to use
            them, or "". Kept outside <student_answer>, so the note is read
            as ours and not as text the student wrote to the grader.

    Returns:
        list[dict]: A system message and a user message (text, or text plus
            image parts).
    """
    criteria_json = json.dumps(
        [{"id": c["id"], "description": c["description"], "marks": c["marks"],
          "partial_allowed": c["partial_allowed"]} for c in q["criteria"]],
        ensure_ascii=False, indent=1)
    examples = _examples_text(q) if settings.use_examples else ""
    boundary = GUARD_BOUNDARY if settings.guard_prompt else ""
    head = GRADER_HEAD.format(
        qid=q["id"], max_marks=q["max_marks"], question=q["text"],
        model_answer=q.get("model_answer") or "(none provided)", criteria_json=criteria_json,
        examples_block=EXAMPLES_BLOCK.format(examples=examples) if examples else "",
        strictness=settings.strictness, policy=STRICTNESS_POLICY[settings.strictness],
        quote_source="the page image" if mode == "image" else "the student's answer text below",
        guard_rule=GUARD_RULE if settings.guard_prompt else "", feedback_language=feedback_language,
        policy_rules=_policy_rules(q, settings, answer))
    body = {"text": ANSWER_TEXT, "both": ANSWER_BOTH, "image": ANSWER_IMAGE}[mode].format(
        qid=q["id"], boundary=boundary, answer=answer or "(no answer written)")
    tail = (TOOL_BLOCK.format(results=tool_text) if tool_text else "") + \
           (CRITIQUE_BLOCK.format(critique=critique) if critique else "") + \
           _output_format(settings)
    user_text = head + (LINKED_BLOCK.format(linked=linked) if linked else "") + body + tail
    system = GRADER_SYSTEM + (GUARD_SYSTEM if settings.guard_prompt else "")
    if images and mode in ("image", "both"):
        content = [{"type": "text", "text": user_text}] + [image_part(img) for img in images]
    else:
        content = user_text
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def normalize(q: dict, raw: object, gradable_text: str, findings: list[Finding], evidence_policy: str) -> dict:
    """Turns the model's JSON reply into a clean grade.

    Pure code with no model call, so it can be re-run offline on saved raw
    replies (eval/offline.py) with a different evidence policy.

    Args:
        q (dict): Rubric question with "criteria".
        raw (object): Parsed model reply; anything other than a dict counts as
            an empty reply.
        gradable_text (str): Answer text with crossed-out parts removed;
            evidence quotes must be found in it.
        findings (list[Finding]): Injection findings in the answer; a quote
            overlapping one never earns marks.
        evidence_policy (str): What to do with a quote not found in the
            answer: "reject" (remove the mark), "flag" or "off".

    Returns:
        dict: Keys "criteria" (criterion id -> verdict, marks, evidence,
            reason, rejected, evidence_flagged), "total", "feedback",
            "rejections", "evidence_flags", "rubric_note",
            "addressed_to_grader" and "addressed_text".
    """
    threshold = thresholds()["evidence_match_threshold"]
    raw = raw if isinstance(raw, dict) else {}
    items = raw.get("criteria", [])
    by_id = {str(i.get("criterion_id", "")).strip(): i
             for i in (items if isinstance(items, list) else []) if isinstance(i, dict)}

    crits, rejections, ev_flags = {}, 0, 0
    for c in q["criteria"]:
        item = by_id.get(c["id"], {})
        verdict = str(item.get("verdict", "not_met")).strip().lower()
        if verdict not in ("met", "partial", "not_met"):
            verdict = "not_met"
        if verdict == "met":
            marks = c["marks"]
        elif verdict == "partial" and c["partial_allowed"]:
            marks = round_half(clamp(to_float(item.get("marks_awarded")), 0, c["marks"]))
            if marks >= c["marks"]:
                marks = c["marks"] - 0.5 if c["marks"] > 0.5 else c["marks"]
        else:
            marks = 0.0
        if marks == 0:
            verdict = "not_met"
        evidence = str(item.get("evidence", "") or "").strip()
        rejected = flagged = False
        if marks > 0:
            from_injection = bool(findings) and guard.overlaps(evidence, findings)
            found = evidence_supported(evidence, gradable_text, threshold)
            if from_injection or (evidence_policy == "reject" and not found):
                rejected, marks, verdict = True, 0.0, "not_met"
                rejections += 1
            elif evidence_policy == "flag" and not found:
                flagged = True
                ev_flags += 1
        crits[c["id"]] = {"verdict": verdict, "marks": float(marks), "evidence": evidence,
                          "reason": str(item.get("reason", "") or ""), "rejected": rejected,
                          "evidence_flagged": flagged}
    return {
        "criteria": crits,
        "total": sum(v["marks"] for v in crits.values()),
        "feedback": str(raw.get("feedback", "") or ""),
        "rejections": rejections,
        "evidence_flags": ev_flags,
        "rubric_note": str(raw.get("rubric_note", "") or "").strip(),
        "addressed_to_grader": bool(raw.get("addressed_to_grader")),
        "addressed_text": str(raw.get("addressed_text", "") or ""),
    }


def grade_once(q: dict, answer: str, settings: RunSettings, profile: str | None, *, mode: str = "text",
               images: list[Image.Image] | None = None, findings: list[Finding] | None = None,
               tool_text: str = "", critique: str = "", sample_idx: int = 0, temperature: float = 0.0,
               feedback_language: str = "English", linked: str = "", linked_quotable: str = "") -> dict:
    """Runs one grading sample: builds the prompt, calls the model, cleans the reply.

    Args:
        q (dict): Rubric question.
        answer (str): The student's answer text.
        settings (RunSettings): Run settings.
        profile (str | None): Model profile name from config.yaml.
        mode (str): What the grader sees: "text", "image" or "both".
        images (list[Image.Image] | None): Page images for this answer.
        findings (list[Finding] | None): Injection findings in the answer.
        tool_text (str): Formatted tool observations, or "".
        critique (str): Critic output for a re-grade, or "".
        sample_idx (int): Sample number; keeps cached samples separate.
        temperature (float): Sampling temperature.
        feedback_language (str): Language the feedback should be written in.
        linked (str): Linked answers for the prompt (see `build_messages`).
        linked_quotable (str): Linked text the grader may quote as evidence
            (answers the student pointed to with "see Q3").

    Returns:
        dict: The output of `normalize`, plus "raw" (the model's reply).

    Raises:
        LLMError: If the model call fails or the reply is not valid JSON.
    """
    images = images or []
    messages = build_messages(q, answer, settings, images, mode, tool_text, critique, feedback_language, linked)
    role = "grader_vision" if images and mode in ("image", "both") else "grader"
    raw = chat_json(role, messages, profile=profile, temperature=temperature,
                    sample_idx=sample_idx + 1000 * settings.seed, max_tokens=4000)
    policy = settings.evidence_check
    if mode == "image" and policy == "reject":
        policy = "flag"  # quotes read from the image won't match the OCR text exactly
    gradable = remove_struck(answer + ("\n" + linked_quotable if linked_quotable else ""))
    result = normalize(q, raw, gradable, findings or [], policy)
    result["raw"] = raw
    return result
