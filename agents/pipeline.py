"""Orchestrator: runs the agents in order. The app, cli.py and the ablation
runner all call these functions.

Steps (optional steps in brackets):
  1. build_rubric_stage(): read the question paper, rubric and model answer
     (OCR where needed) and build the rubric JSON. It can be edited before
     grading.
  2. read_sheet(): load the answer sheet files and OCR the pages that need it.
  3. grade_sheet(): segment the sheet by question, then for each question in
     parallel (grade_question):
       a. grounding check: is the segmented answer really on the sheet?
       b. injection detector, cross-references, blank check
       c. [tools] -> N grader samples -> aggregate
       d. [critic + one re-grade if the samples disagree]
       e. [OCR re-check if marks were lost -> re-grade]
       f. [translate the feedback with Bodhan]
     then write the overall feedback and [locate answers for annotation].
  4. grade_batch(): grade_sheet() for every student, then class analytics and
     [a summary of common mistakes per question].

Callers start a usage counter with core.llm.start_usage() first; the report's
"usage" covers everything since then (OCR included). Progress goes to the
optional `progress(fraction, message)` callback, and finer-grained events to
core.trace (the app's live agent panel).
"""

import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from rapidfuzz import fuzz

from core import bodhan, guard, trace
from core.analytics import class_analytics
from core.config import max_parallel, model_for, resolve_profile_name, thresholds
from core.ingest import Document, load_many
from core.lang import detect_language, is_indic, resolve_feedback_language
from core.llm import LLMError, chat, current_usage, submit
from core.settings import RunSettings
from core.utils import (
    coverage,
    find_references,
    grid_answers,
    is_blank,
    is_page_number,
    normalize_qid,
    normalize_ws,
    only_question,
    strip_label,
)

from .aggregator import aggregate, disagreed, needs_reflection
from .critic import critique as run_critic
from .grader import grade_once
from .locator import locate
from .ocr import run_ocr
from .prompts import CLASS_SUMMARY_SYSTEM, CLASS_SUMMARY_USER, SUMMARY_SYSTEM, SUMMARY_USER
from .recheck import recheck
from .rubric import build_rubric, validate_rubric
from .segmenter import FALLBACK_WARNING, segment
from .tool_agent import format_observations, run_tools

NO_PAPER_WARNING = ("No question paper was given, so the questions and their marks were taken from the marking "
                    "scheme and model answer. Check the question text and marks in the table.")
NO_REFERENCE_WARNING = (
    "No rubric and no model answer were given. Without a reference, LLM graders tend to over-grade "
    "(Caraeni et al., arXiv 2411.05231; engineering-exam pipeline, arXiv 2601.00730). Check the rubric carefully.")


def _as_list(x: object) -> list:
    """Wraps a single value in a list.

    Args:
        x (object): None, a single item, or a list/tuple of items.

    Returns:
        list: [] for a falsy value, a copy of a list/tuple, else [x].
    """
    if not x:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _noop(*_: object) -> None:
    """Default progress callback that does nothing."""


# ================================================================ stage 1: rubric
def _read(files: str | list[str] | None, label: str, settings: RunSettings, profile: str,
          force_ocr: bool = False, progress: Callable[[float, str], None] | None = None) -> Document | None:
    """Loads one role's files into a Document and runs OCR where needed.

    Args:
        files (str | list[str] | None): A path or a list of paths; empty
            means not given.
        label (str): Name of the document for messages and prompts.
        settings (RunSettings): Run settings (language hint, OCR backend).
        profile (str): Model profile name.
        force_ocr (bool): OCR every page even if it has a text layer.
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback for OCR.

    Returns:
        Document | None: The read document, or None if no files were given.
    """
    files = _as_list(files)
    if not files:
        return None
    doc = load_many(files, force_ocr=force_ocr)
    trace.emit("ingest", f"Opened {label}: {len(doc.pages)} page(s)")
    run_ocr(doc, label, profile, progress=progress, language=settings.language_hint, backend=settings.ocr_backend)
    return doc


def build_rubric_stage(question_paper: str | list[str], rubric_files: str | list[str] | None,
                       model_answer_files: str | list[str] | None, settings: RunSettings,
                       progress: Callable[[float, str], None] | None = None) -> dict:
    """Reads the exam documents and drafts the rubric, before any answer is seen.

    Args:
        question_paper (str | list[str]): Path(s) to the question paper (may be
            empty if a marking scheme or model answer is given).
        rubric_files (str | list[str] | None): Path(s) to the marking scheme.
        model_answer_files (str | list[str] | None): Path(s) to the model
            answer.
        settings (RunSettings): Run settings.
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback.

    Returns:
        dict: Keys "rubric" (validated rubric), "warnings", "rubric_mode_used"
            and "texts" (the text read from each document).

    Raises:
        ValueError: If no document is given at all, or the rubric is unusable.
        LLMError: If a model call fails.
    """
    progress = progress or _noop
    profile = resolve_profile_name(settings.profile)
    if not (_as_list(question_paper) or _as_list(rubric_files) or _as_list(model_answer_files)):
        raise ValueError("Give at least one of the question paper, marking scheme or model answer.")
    warnings: list[str] = []
    texts = {}
    for i, (key, files, label) in enumerate([("question_paper", question_paper, "question paper"),
                                             ("rubric", rubric_files, "rubric"),
                                             ("model_answer", model_answer_files, "model answer")]):
        progress(0.05 + 0.25 * i, f"Reading {label}...")
        doc = _read(files, label, settings, profile)
        texts[key] = doc.full_text(with_markers=False) if doc else ""
        if doc:
            warnings.extend(doc.warnings)
    if not texts["question_paper"].strip():
        warnings.append(NO_PAPER_WARNING)
        trace.emit("rubric", "No question paper: questions taken from the marking scheme / model answer", "warn")
    if not texts["rubric"].strip() and not texts["model_answer"].strip():
        warnings.append(NO_REFERENCE_WARNING)
        trace.emit("rubric", "No rubric or model answer: expect over-grading", "warn")

    progress(0.8, "Building the rubric...")
    trace.emit("rubric", "Turning the marking scheme into criteria")
    raw, mode = build_rubric(texts["question_paper"], texts["rubric"], texts["model_answer"],
                             settings.rubric_mode, profile)
    rubric, rub_warn = validate_rubric(raw)
    warnings.extend(rub_warn)
    n_crit = sum(len(q["criteria"]) for q in rubric["questions"])
    trace.emit("rubric", f"{len(rubric['questions'])} questions, {n_crit} criteria ({mode})", "good")
    progress(1.0, "Rubric ready")
    return {"rubric": rubric, "warnings": warnings, "rubric_mode_used": mode, "texts": texts}


# ================================================================ stage 2: answer sheet
def read_sheet(files: str | list[str], settings: RunSettings,
               progress: Callable[[float, str], None] | None = None) -> Document:
    """Loads a student's answer sheet and runs OCR where needed.

    Args:
        files (str | list[str]): Path(s) to the sheet's files.
        settings (RunSettings): Run settings (force_ocr, OCR backend, ...).
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback.

    Returns:
        Document: The answer sheet with page text filled in.

    Raises:
        ValueError: If no files are given.
        LLMError: If an OCR model call fails.
    """
    profile = resolve_profile_name(settings.profile)
    if not _as_list(files):
        raise ValueError("No answer sheet given.")
    return _read(files, "student answer sheet", settings, profile, force_ocr=settings.force_ocr,
                 progress=progress)


# ================================================================ one question
def _page_images(sheet: Document, pages: list[int], cap: int) -> tuple[list[Image.Image], list[int], str]:
    """Picks the page images that go with one answer.

    Only OCR'd pages (scans and photos) count. If none of `pages` has an
    image, all such pages are used.

    Args:
        sheet (Document): The answer sheet.
        pages (list[int]): Page numbers the segmenter gave for this answer.
        cap (int): Maximum number of images to send to the grader.

    Returns:
        tuple[list[Image.Image], list[int], str]: The images, their page
            numbers, and a note if some pages were left out ("" otherwise).
    """
    with_img = {p.number: p.image for p in sheet.pages if p.image is not None and p.source == "ocr"}
    chosen = [n for n in pages if n in with_img] or sorted(with_img)
    note = ""
    if len(chosen) > cap:
        note = f"Answer spans {len(chosen)} pages; only the first {cap} images were sent to the grader."
        chosen = chosen[:cap]
    return [with_img[n] for n in chosen], chosen, note


def _input_mode(setting: str, has_images: bool) -> str:
    """Decides what the grader sees for one answer.

    Args:
        setting (str): The grader_input setting: "auto", "text", "image" or
            "both".
        has_images (bool): Whether page images exist for this answer.

    Returns:
        str: "text" when there are no images; "both" for "auto"; otherwise
            the setting itself.
    """
    if not has_images:
        return "text"
    return "both" if setting == "auto" else setting


def _grade_core(q: dict, answer: str, findings: list[guard.Finding], images: list[Image.Image], mode: str,
                settings: RunSettings, profile: str, feedback_language: str, offset: int = 0,
                linked: str = "", linked_quotable: str = "") -> dict:
    """Grades one answer: tools -> N samples -> aggregate -> optional critic + one re-grade.

    Args:
        q (dict): Rubric question.
        answer (str): The answer text to grade.
        findings (list[guard.Finding]): Injection findings in the answer.
        images (list[Image.Image]): Page images to send ([] in text mode).
        mode (str): What the grader sees: "text", "image" or "both".
        settings (RunSettings): Run settings.
        profile (str): Model profile name.
        feedback_language (str): Language for the grader's feedback.
        offset (int): Added to sample indices so a re-grade after the OCR
            re-check gets its own cached samples.
        linked (str): Answers to linked questions, shown to the grader
            outside the student's answer.
        linked_quotable (str): Linked text that may be quoted as evidence.

    Returns:
        dict: Keys "samples", "first" (aggregate of the samples), "final"
            (after the critic re-grade, else same as "first"), "critique",
            "regraded" (the re-grade sample or None) and "tool_obs".

    Raises:
        LLMError: If a model call fails.
    """
    tool_obs = []
    if settings.use_tools:
        tool_obs = run_tools(q, answer, profile, settings.guard_prompt)
        if tool_obs:
            trace.emit("tools", f"{q['id']}: ran {len(tool_obs)} check(s)")
    tool_text = format_observations(tool_obs)
    n = settings.n_samples
    temperature = settings.temperature if n > 1 else 0.0
    samples = []
    for i in range(n):
        s = grade_once(q, answer, settings, profile, mode=mode, images=images, findings=findings,
                       tool_text=tool_text, sample_idx=offset + i, temperature=temperature,
                       feedback_language=feedback_language, linked=linked, linked_quotable=linked_quotable)
        samples.append(s)
        trace.emit("grade", f"{q['id']}: sample {i + 1}/{n} gave {s['total']:g}/{q['max_marks']:g}"
                   + (f", {s['rejections']} unsupported mark(s) removed" if s["rejections"] else ""))
    first = aggregate(q, samples, settings.aggregation, settings.strictness)
    final, crit_text, regraded = first, "", None
    if settings.use_reflection and needs_reflection(q, samples):
        trace.emit("critic", f"{q['id']}: samples disagree {first['sample_totals']}, asking the critic", "warn")
        crit_text = run_critic(q, answer, samples, profile, settings.guard_prompt)
        regraded = grade_once(q, answer, settings, profile, mode=mode, images=images, findings=findings,
                              tool_text=tool_text, critique=crit_text, sample_idx=offset + 100,
                              temperature=0.0, feedback_language=feedback_language, linked=linked,
                              linked_quotable=linked_quotable)
        final = aggregate(q, [regraded], "mean", settings.strictness)
        trace.emit("critic", f"{q['id']}: re-graded after critique: {final['awarded']:g}")
    return {"samples": samples, "first": first, "final": final, "critique": crit_text,
            "regraded": regraded, "tool_obs": tool_obs}


def _result(q: dict, **kw: object) -> dict:
    """Builds a per-question result dict with defaults for every field.

    Args:
        q (dict): Rubric question with "id", "text" and "max_marks".
        **kw (object): Fields to override.

    Returns:
        dict: The result; "needs_review" is True when "review_reasons" is
            not empty.
    """
    base = {
        "id": q["id"], "text": q["text"], "max_marks": q["max_marks"], "awarded": 0.0,
        "first_pass_awarded": 0.0, "confidence": 1.0, "sample_totals": [], "needs_review": False,
        "review_reasons": [], "info": [], "criteria": [], "feedback": "", "grader_input": "text",
        "pages": [], "coverage": 1.0, "injection_findings": [], "addressed_text": "",
        "tool_observations": [], "critique": "", "recheck": None, "answer_text": "",
        "graded_text": "", "samples": [], "regraded_raw": None, "reflected": False,
        "language": "English", "feedback_language": "English", "human_reviewed": False,
    }
    base.update(kw)
    base["needs_review"] = bool(base["review_reasons"])
    return base


def _pages_with(text: str, sheet: Document) -> list[int]:
    """Pages of the sheet on which lines of a text appear.

    Args:
        text (str): Text taken from the sheet (e.g. the part matched to no question).
        sheet (Document): The answer sheet.

    Returns:
        list[int]: Page numbers, in order.
    """
    pages = []
    lines = [normalize_ws(x) for x in text.splitlines() if len(x.strip()) >= 12]
    for p in sheet.pages:
        body = normalize_ws(p.text or "")
        if any(line in body for line in lines):
            pages.append(p.number)
    return pages


_OPTION_ONLY = re.compile(r"^\(?[A-Da-d]\)?\.?$")


def _is_mcq(q: dict) -> bool:
    """Whether a rubric question is multiple choice: its model answer is one option letter,
    or its text lists options (a) and (b)."""
    if _OPTION_ONLY.match((q.get("model_answer") or "").strip()):
        return True
    text = q.get("text") or ""
    return "(a)" in text.lower() and "(b)" in text.lower()


def grade_question(q: dict, seg: dict, sheet: Document, sheet_text: str, settings: RunSettings,
                   profile: str) -> dict:
    """Grades one question of one sheet (step 3 in the module docstring).

    Args:
        q (dict): Rubric question.
        seg (dict): Segmenter output for this question: "answer_text",
            "pages" and optional "referenced" (linked answers).
        sheet (Document): The answer sheet.
        sheet_text (str): Plain sheet text, for the grounding check.
        settings (RunSettings): Run settings.
        profile (str): Model profile name.

    Returns:
        dict: The question result (see `_result`), with "awarded",
            "criteria", "feedback", "review_reasons", "info", "samples",
            "recheck" and related fields.

    Raises:
        LLMError: If a model call fails.
    """
    th = thresholds()
    strip_mode = settings.guard_detector == "strip"
    answer_raw = seg["answer_text"]
    review, info = [], []
    if seg.get("from_grid"):
        info.append("Answer read from a grid of answers on the sheet")

    # languages: what the student wrote in, and what the feedback should be in
    language = detect_language(answer_raw)
    if settings.language_hint != "auto" and language != "English":
        language = settings.language_hint  # e.g. Devanagari written in Marathi
    fb_language = resolve_feedback_language(settings.feedback_language, language)
    translate_after = settings.feedback_translator == "bodhan" and is_indic(fb_language)
    prompt_language = "English" if translate_after else fb_language
    if language != "English":
        info.append(f"Written in {language}")

    # 1. grounding: is the segmented answer really on the sheet? An "answer" that is only the
    #    question itself (copied, or handed back by the segmenter) is simply not attempted.
    echo = bool(answer_raw.strip()) and only_question(strip_label(answer_raw, q["id"]), q["text"])
    cov = coverage(answer_raw, sheet_text) if answer_raw.strip() else 1.0
    if cov < th["grounding_min_coverage"] and not echo:
        review.append(f"Only {cov:.0%} of the matched answer was found on the sheet (segmenter may have altered it)")

    # 2. injection detector (rules, not a model)
    findings = guard.scan(answer_raw) if settings.guard_detector != "off" else []
    answer = guard.strip(answer_raw, findings) if findings and strip_mode else answer_raw
    if findings:
        review.append("Text aimed at the grader found" + (" and removed before grading" if strip_mode else ""))
        trace.emit("guard", f"{q['id']}: text aimed at the grader: “{findings[0].text[:80]}”", "bad")

    # 2b. linked questions: the student wrote "as explained in Q3", or the question
    #     builds on an earlier one. The grader sees the other answer in its own
    #     block, outside <student_answer>, so our note is never mistaken for
    #     text the student wrote to the grader.
    linked_parts, quotable = [], []
    for rid, rtext, kind in seg.get("referenced") or []:
        rfind = guard.scan(rtext) if settings.guard_detector != "off" else []
        rtext = guard.strip(rtext, rfind) if rfind and strip_mode else rtext
        if kind == "refers":
            note = (f"The student refers to their answer to {rid} in this answer. Credit what that answer "
                    f"contains only where it answers {q['id']}; you may quote it as evidence.")
            info.append(f"Refers to the answer to {rid}; the grader saw it too")
            quotable.append(rtext.strip())
        else:
            note = (f"{q['id']} builds on {rid}. The answer to {rid} is here for context only; do not grade it "
                    f"and do not quote it as evidence. If a mistake from {rid} is carried into this answer but "
                    f"the method here is right, do not take marks off again for that mistake (error carried "
                    f"forward).")
            info.append(f"Builds on {rid}; the grader saw that answer (error carried forward)")
        linked_parts.append(f'{note}\n<answer question="{rid}">\n{rtext.strip()}\n</answer>')
    linked, linked_quotable = "\n\n".join(linked_parts), "\n".join(quotable)

    common = {"coverage": cov, "answer_text": answer_raw, "injection_findings": guard.as_dicts(findings),
              "language": language, "feedback_language": fb_language}

    # 3. blank check: no model call for an empty answer (a lone label or the copied question
    #    counts as empty). If the reader could not read what is written, the grader looks at the
    #    page image itself; only without an image does a human have to look.
    images, used_pages, note = _page_images(sheet, seg.get("pages") or [], th["max_images_per_question"])
    mode = _input_mode(settings.grader_input, bool(images))
    body = "" if echo else strip_label(answer, q["id"])
    unreadable = "[illegible]" in answer.lower()
    if settings.presence_check and is_blank(body) and not (unreadable and not echo and mode != "text"):
        crit = [{"id": c["id"], "description": c["description"], "max": c["marks"], "awarded": 0.0,
                 "verdict": "not_met", "evidence": "", "reason": "No answer written."} for c in q["criteria"]]
        if unreadable and not echo:  # something is written, nobody could read it, and there is no image
            review.append("Answer is there but could not be read at all; 0 given until checked")
        why = "only the question is written" if echo else "not attempted"
        trace.emit("grade", f"{q['id']}: {why}, 0 without a model call")
        return _result(q, criteria=crit, feedback="Not attempted.",
                       info=info + ["Not attempted"] + (["Only the question was written"] if echo else []),
                       review_reasons=review, graded_text=answer, pages=seg.get("pages") or [], **common)

    if unreadable and mode == "text":
        review.append("Part of the answer could not be read")
    elif unreadable:
        info.append("Part of the transcription was unreadable; the grader read the page image as well")

    # 4. what the grader sees
    if note:
        info.append(note)
    trace.emit("grade", f"{q['id']}: grading from {'image + text' if mode == 'both' else mode}"
               f" with {settings.n_samples} sample(s)")

    # 5. grade (samples, aggregate, critic)
    grade_images = images if mode != "text" else []
    core = _grade_core(q, answer, findings, grade_images, mode, settings, profile, prompt_language,
                       linked=linked, linked_quotable=linked_quotable)

    # 6. OCR re-check where marks were lost (only possible with page images)
    recheck_info = None
    if settings.ocr_recheck and images and core["final"]["awarded"] < q["max_marks"]:
        rc = recheck(q, answer, core["final"]["criteria"], images, profile)
        affecting = [d for d in rc["discrepancies"] if d.get("affects_deduction", True)]
        high = [d for d in affecting if str(d.get("confidence", "")).lower() == "high"]
        low = [d for d in affecting if d not in high]
        corrected = rc["corrected_answer"]
        before = core["final"]["awarded"]
        recheck_info = {"discrepancies": rc["discrepancies"], "applied": False, "awarded_before": before}
        similar = corrected and fuzz.ratio(normalize_ws(corrected), normalize_ws(answer)) >= th["recheck_min_similarity"]
        if high and similar:
            d0 = high[0]
            trace.emit("recheck", f"{q['id']}: misread “{d0.get('transcribed', '')}”, page says "
                       f"“{d0.get('actually_written', '')}”; re-grading", "warn")
            f2 = guard.scan(corrected) if settings.guard_detector != "off" else []
            answer = guard.strip(corrected, f2) if f2 and strip_mode else corrected
            findings = findings + f2
            core = _grade_core(q, answer, findings, grade_images, mode, settings, profile, prompt_language,
                               offset=500, linked=linked, linked_quotable=linked_quotable)
            recheck_info.update(applied=True, corrected_answer=answer)
            info.append(f"OCR re-check fixed a misreading and re-graded ({before:g} -> {core['final']['awarded']:g})")
        elif high:
            review.append("OCR re-check found a misreading but its correction was unusable; check the page")
        else:
            trace.emit("recheck", f"{q['id']}: transcription matches the handwriting")
        if low:
            review.append("OCR re-check unsure whether the handwriting was read correctly")

    first, final = core["first"], core["final"]
    if disagreed(q, core["samples"]):
        review.append(f"Grading samples disagreed (totals {first['sample_totals']})")
    if first["rejections"] or final["rejections"]:
        review.append(f"{first['rejections'] + final['rejections']} mark(s) removed: evidence quote not in the answer")
    if first["evidence_flags"] or final["evidence_flags"]:
        review.append("Evidence quotes not found in the transcription; compare with the page")
    addressed = final["addressed_text"] or first["addressed_text"]
    if addressed and (is_page_number(addressed)  # a page footer like "1 / total 3" is not a message
                      or (linked and normalize_ws(addressed[:60]) in normalize_ws(linked))):  # our own note
        addressed = ""
    elif first["addressed_to_grader"] or final["addressed_to_grader"]:
        review.append("Grader reports text addressed to it" + (f": “{addressed[:120]}”" if addressed else ""))
        trace.emit("guard", f"{q['id']}: grader reports text addressed to it", "bad")
    rubric_note = final.get("rubric_note") or first.get("rubric_note") or ""
    if rubric_note and rubric_note.strip().lower() not in ("none", "n/a", "-"):
        review.append(f"Grader questions the rubric here: {rubric_note[:200]}")
    if any(o.get("injection_in_output") for o in core["tool_obs"]):
        review.append("A tool result contained text aimed at the grader")
    if core["regraded"] is not None:
        info.append("Critic reviewed and re-graded")

    feedback = final["feedback"]
    if translate_after and feedback:
        try:
            feedback = bodhan.translate(feedback, "English", fb_language, profile)
            info.append(f"Feedback translated to {fb_language} by Bodhan")
        except bodhan.BodhanError as e:
            info.append(f"Feedback left in English ({e})")

    trace.emit("grade", f"{q['id']}: {final['awarded']:g}/{q['max_marks']:g}"
               + (" (needs a human check)" if review else ""), "warn" if review else "good")
    return _result(
        q, awarded=final["awarded"], first_pass_awarded=first["awarded"], confidence=first["confidence"],
        sample_totals=first["sample_totals"], review_reasons=review, info=info,
        criteria=final["criteria"], feedback=feedback, grader_input=mode, pages=used_pages or seg.get("pages") or [],
        addressed_text=addressed, tool_observations=core["tool_obs"], critique=core["critique"],
        recheck=recheck_info, graded_text=answer,
        samples=[{"raw": s["raw"], "total": s["total"]} for s in core["samples"]],
        regraded_raw=core["regraded"]["raw"] if core["regraded"] else None,
        reflected=core["regraded"] is not None, **common)


# ================================================================ one sheet
def _overall_feedback(questions: list[dict], profile: str, language: str) -> str:
    """Writes one feedback paragraph for the whole sheet.

    Args:
        questions (list[dict]): Graded question results.
        profile (str): Model profile name.
        language (str): Language to write the feedback in.

    Returns:
        str: The feedback paragraph.

    Raises:
        LLMError: If the model call fails.
    """
    lines = [f"{q['id']}: {q['awarded']:g}/{q['max_marks']:g}. {q['feedback']}" for q in questions]
    messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_USER.format(results="\n".join(lines), language=language)}]
    return chat("summary", messages, profile=profile, max_tokens=1500)


def grade_sheet(rubric: dict | list | str, sheet: Document, settings: RunSettings,
                progress: Callable[[float, str], None] | None = None, overall_feedback: bool = True) -> dict:
    """Grades one answer sheet with a fixed rubric.

    A question whose grading fails (e.g. an unreadable model reply) gets 0
    and is flagged for a human; the rest of the sheet is still graded.

    Args:
        rubric (dict | list | str): The rubric; validated again here.
        sheet (Document): The answer sheet, already read.
        settings (RunSettings): Run settings.
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback.
        overall_feedback (bool): Whether to write the overall feedback.

    Returns:
        dict: The report, with keys "meta", "total", "max_total",
            "needs_review", "sheet_review_reasons", "questions",
            "overall_feedback", "unassigned_text", "unassigned_pages", "warnings",
            "annotation_positions", "usage" and "rubric".

    Raises:
        ValueError: If the rubric is unusable.
        LLMError: If every question fails, or another model call fails.
    """
    progress = progress or _noop
    profile = resolve_profile_name(settings.profile)
    th = thresholds()
    rubric, warnings = validate_rubric(rubric)
    warnings = list(sheet.warnings) + warnings

    progress(0.05, "Matching answers to questions...")
    trace.emit("segment", "Matching parts of the sheet to questions")
    answers, unassigned, seg_warn = segment(rubric, sheet.full_text(), profile)
    warnings.extend(seg_warn)
    grid = grid_answers(sheet.full_text(with_markers=False))
    for q in rubric["questions"]:  # MCQ answers written as a grid that the segmenter missed
        a = answers[q["id"]]
        num = normalize_qid(q["id"])
        if not a["answer_text"].strip() and num in grid and _is_mcq(q):
            a.update(answer_text=grid[num], from_grid=True)
            trace.emit("segment", f"{q['id']}: answer read from the grid of answers (“{grid[num]}”)")
    found = sum(1 for a in answers.values() if a["answer_text"].strip())
    trace.emit("segment", f"Found answers to {found} of {len(answers)} questions", "good")
    plain = sheet.full_text(with_markers=False)

    questions = rubric["questions"]
    if settings.cross_references:
        # two kinds of links, both hand the grader the other answer as well:
        #  - the ANSWER says so ("as explained in Q3")
        #  - the QUESTION builds on an earlier one ("Hence ...", "using your answer to Q3"),
        #    from rubric["depends_on"]: then a mistake carried forward is not punished twice
        qids = [q["id"] for q in questions]
        for q in questions:
            a = answers[q["id"]]
            said = find_references(a["answer_text"], qids, q["id"])
            links = [(r, "refers") for r in said] + [(r, "builds_on") for r in q.get("depends_on", []) if r not in said]
            a["referenced"] = [(r, answers[r]["answer_text"], kind) for r, kind in links
                               if r in answers and answers[r]["answer_text"].strip()]
            if a["referenced"]:
                trace.emit("segment", f"{q['id']} is linked to {', '.join(r for r, _, _ in a['referenced'])}")
    results: dict[str, dict] = {}
    failed: list[LLMError] = []
    with ThreadPoolExecutor(max_workers=max_parallel(profile)) as pool:
        futures = [(q, submit(pool, grade_question, q, answers[q["id"]], sheet, plain, settings, profile))
                   for q in questions]
        for done, (q, fut) in enumerate(futures, start=1):
            try:
                results[q["id"]] = fut.result()
            except LLMError as e:  # one bad reply should not lose the whole sheet: a human grades this one
                failed.append(e)
                trace.emit("grade", f"{q['id']}: could not be graded ({str(e)[:80]})", "bad")
                a = answers[q["id"]]
                results[q["id"]] = _result(
                    q, review_reasons=["Could not be graded automatically (the model's reply was unusable); "
                                       "grade it by hand"],
                    answer_text=a["answer_text"], graded_text=a["answer_text"], pages=a.get("pages") or [],
                    criteria=[{"id": c["id"], "description": c["description"], "max": c["marks"], "awarded": 0.0,
                               "verdict": "not_met", "evidence": "", "reason": "Not graded (model error)."}
                              for c in q["criteria"]])
            progress(0.1 + 0.8 * done / len(questions), f"Graded {q['id']} ({done}/{len(questions)})")
    if failed and len(failed) == len(questions):  # nothing worked (e.g. no credit left): report the error
        raise failed[0]
    ordered = [results[q["id"]] for q in questions]

    sheet_review = [w for w in seg_warn if w == FALLBACK_WARNING]
    leftover = "\n".join(x for x in (unassigned or "").splitlines() if not is_page_number(x))
    words = len(re.findall(r"\w+", leftover))
    unmatched_pages = _pages_with(leftover, sheet) if words >= th["unassigned_flag_words"] else []
    if words >= th["unassigned_flag_words"]:
        where = (f"on page{'s' if len(unmatched_pages) > 1 else ''} {', '.join(map(str, unmatched_pages))}"
                 if unmatched_pages else "on the sheet")
        sheet_review.append(f"{words} words {where} were not matched to any question (possible unlabelled "
                            "answer); the text is shown under “Text not matched to any question”")
        trace.emit("segment", f"{words} words {where} not matched to any question", "warn")

    langs = Counter(q["feedback_language"] for q in ordered if "Not attempted" not in q["info"])
    sheet_language = langs.most_common(1)[0][0] if langs else "English"
    overall = ""
    if overall_feedback:
        progress(0.92, "Writing overall feedback...")
        trace.emit("summary", f"Writing overall feedback ({sheet_language})")
        overall = _overall_feedback(ordered, profile, sheet_language)

    positions = {}
    if settings.annotate:
        progress(0.96, "Placing marks on the pages...")
        positions = locate(sheet, ordered, profile, use_model=settings.annotate_locate)

    total = sum(q["awarded"] for q in ordered)
    max_total = sum(q["max_marks"] for q in ordered)
    trace.emit("done", f"Total {total:g} / {max_total:g}", "good")
    return {
        "meta": {
            "settings": settings.to_dict() | {"profile": profile},
            "models": {"vision": model_for("ocr", profile), "grader": model_for("grader", profile),
                       "grader_vision": model_for("grader_vision", profile)},
            "answer_sheet": sheet.name,
            "ocr_engines": sorted({p.ocr_engine for p in sheet.pages if p.ocr_engine}),
            "language": sheet_language,
        },
        "total": total,
        "max_total": max_total,
        "needs_review": bool(sheet_review) or any(q["needs_review"] for q in ordered),
        "sheet_review_reasons": sheet_review,
        "questions": ordered,
        "overall_feedback": overall,
        "unassigned_text": leftover if words >= th["unassigned_flag_words"] else unassigned,
        "unassigned_pages": unmatched_pages,
        "warnings": warnings,
        "annotation_positions": positions,
        "usage": current_usage().as_dict(),
        "rubric": rubric,
    }


# ================================================================ whole class
def grade_batch(rubric: dict | list | str, students: list[tuple[str, list[str]]], settings: RunSettings,
                progress: Callable[[float, str], None] | None = None) -> dict:
    """Grades every student's sheet and builds the class analytics.

    A sheet that fails with LLMError or ValueError is recorded with its error
    and the rest are still graded.

    Args:
        rubric (dict | list | str): The rubric.
        students (list[tuple[str, list[str]]]): (name, file paths) per student.
        settings (RunSettings): Run settings.
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback.

    Returns:
        dict: Keys "results" (per student: "student", "report", "sheet",
            "error"), "analytics", "usage", "rubric" and "settings".

    Raises:
        ValueError: If the rubric is unusable.
    """
    progress = progress or _noop
    profile = resolve_profile_name(settings.profile)
    rubric, _ = validate_rubric(rubric)
    results = []
    for i, (name, files) in enumerate(students):
        progress(i / max(1, len(students)), f"Student {i + 1}/{len(students)}: {name}")
        trace.emit("ingest", f"Student {i + 1}/{len(students)}: {name}")
        try:
            sheet = read_sheet(files, settings)
            rep = grade_sheet(rubric, sheet, settings)
            rep["meta"]["student"] = name
            results.append({"student": name, "report": rep, "sheet": sheet, "error": None})
        except (LLMError, ValueError) as e:
            trace.emit("error", f"{name}: {e}", "bad")
            results.append({"student": name, "report": None, "sheet": None, "error": str(e)})

    reports = [r["report"] for r in results if r["report"]]
    analytics = class_analytics(rubric, reports)
    if settings.class_summary and reports:
        progress(0.95, "Summarising common mistakes...")
        for qa in analytics["questions"]:
            q = next(x for x in rubric["questions"] if x["id"] == qa["id"])
            notes = [f"- {qr['awarded']:g}/{qr['max_marks']:g}: {qr['feedback']}"
                     for rep in reports for qr in rep["questions"]
                     if qr["id"] == q["id"] and qr["awarded"] < qr["max_marks"] and qr["feedback"]]
            if not notes:
                qa["common_mistakes"] = "No marks lost on this question."
                continue
            trace.emit("summary", f"Common mistakes in {q['id']}")
            messages = [{"role": "system", "content": CLASS_SUMMARY_SYSTEM},
                        {"role": "user", "content": CLASS_SUMMARY_USER.format(
                            qid=q["id"], question=q["text"], avg=qa["mean"], max_marks=q["max_marks"],
                            n=qa["n"], notes="\n".join(notes[:200]))}]
            try:
                qa["common_mistakes"] = chat("class_summary", messages, profile=profile, max_tokens=1500)
            except LLMError as e:
                qa["common_mistakes"] = f"(summary failed: {e})"
    progress(1.0, "Done")
    return {"results": results, "analytics": analytics, "usage": current_usage().as_dict(),
            "rubric": rubric, "settings": settings.to_dict() | {"profile": profile}}
