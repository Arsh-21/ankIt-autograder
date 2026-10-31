"""Locator: finds where each answer starts on each page.

Used only to place the red-ink marks on the annotated sheet, never for
grading.

  typed PDF pages - exact, from the PDF's own line positions (no model call)
  scanned pages   - one vision call per page (setting `annotate_locate`)
Anything not found is placed by the annotator in reading order instead.
"""

import re

from core import trace
from core.ingest import Document, Page
from core.lang import to_ascii_digits
from core.llm import LLMError, chat_json, image_part
from core.utils import QUESTION_WORDS, clamp, normalize_qid, to_float

from .prompts import LOCATOR_SYSTEM, LOCATOR_USER

_LABEL = re.compile(rf"^\s*(?:({QUESTION_WORDS})\s*\.?\s*)?([0-9]+\s*\(?[a-z]?\)?)\s*[.):]", re.I)


def _pdf_positions(page: Page, qids: list[str]) -> dict[str, float]:
    """Finds answer starts on a typed PDF page from its own line positions.

    Args:
        page (Page): A page with "lines" as (y, text) pairs, y from 0 to 1.
        qids (list[str]): Question ids that may start on this page.

    Returns:
        dict[str, float]: Question id -> y of the first line labelled with it.
    """
    wanted = {normalize_qid(q): q for q in qids}
    found = {}
    for y, text in page.lines:
        m = _LABEL.match(to_ascii_digits(text))
        if m:
            q = wanted.get(normalize_qid(m.group(2)))
            if q and q not in found:
                found[q] = y
    return found


def locate(sheet: Document, questions: list[dict], profile: str | None = None,
           use_model: bool = True) -> dict[str, dict[str, float]]:
    """Finds where each graded answer starts on each page of the sheet.

    Typed PDF pages use their line positions; scanned pages use one vision
    call each when `use_model` is set. A failed call on a page is logged and
    that page is skipped.

    Args:
        sheet (Document): The answer sheet.
        questions (list[dict]): Graded questions with "id" and "pages".
        profile (str | None): Model profile name from config.yaml.
        use_model (bool): Whether to ask a vision model for scanned pages.

    Returns:
        dict[str, dict[str, float]]: Page number (as str) -> {question id: y},
            with y from 0 (top) to 1 (bottom). Pages with nothing found are
            left out.
    """
    out: dict[str, dict[str, float]] = {}
    all_ids = [q["id"] for q in questions]
    for page in sheet.pages:
        if page.image is None:
            continue
        # every question is asked about on every page: the segmenter's page numbers can be wrong
        # (e.g. MCQs answered in a grid on the last page), and the model leaves out what isn't there
        on_page = all_ids
        if page.lines:  # typed PDF page
            pos = _pdf_positions(page, on_page)
        elif page.source == "ocr" and use_model:
            listing = "\n".join(f"- {qid}" for qid in on_page)
            messages = [{"role": "system", "content": LOCATOR_SYSTEM},
                        {"role": "user", "content": [
                            {"type": "text", "text": LOCATOR_USER.format(page_no=page.number, questions=listing)},
                            image_part(page.image)]}]
            try:
                reply = chat_json("locator", messages, profile=profile, max_tokens=1500)
            except LLMError as e:
                trace.emit("locate", f"Could not locate answers on page {page.number}: {e}", "warn")
                continue
            pos = {}
            for item in (reply.get("positions") or []) if isinstance(reply, dict) else []:
                qid = str(item.get("question_id", "")).strip()
                if qid in on_page:
                    pos[qid] = clamp(to_float(item.get("y")) / 1000.0, 0.0, 1.0)
            trace.emit("locate", f"Found {len(pos)} answer start(s) on page {page.number}")
        else:
            pos = {}
        if pos:
            out[str(page.number)] = pos
    return out
