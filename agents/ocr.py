"""OCR agent: reads every page that has no usable text (scans, phone photos,
images inside a .docx). Typed PDFs and .docx text are read directly by
core/ingest.py and never reach this agent.

Engines (setting `ocr_backend`):
  vlm    - a vision LLM with our transcription rules (markers for illegible,
           crossed-out, margin and diagram content)
  bodhan - Bodhan AI's Indic OCR (course-provided), for Indian-language sheets
  auto   - Bodhan when the language hint is an Indian language and Bodhan is
           set up, otherwise the vision LLM
If Bodhan fails on a page, that page falls back to the vision LLM.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from core import bodhan, trace
from core.config import load_config, max_parallel, model_for
from core.ingest import Document, Page, looks_blank
from core.lang import is_indic
from core.llm import chat, image_part, submit

from .prompts import OCR_LANGUAGE_LINE, OCR_SYSTEM, OCR_USER


def resolve_backend(backend: str, language: str) -> str:
    """Resolves the "auto" OCR backend to a concrete engine.

    Args:
        backend (str): "auto", "vlm" or "bodhan".
        language (str): Language hint for the document.

    Returns:
        str: "bodhan" or "vlm" for "auto"; otherwise `backend` unchanged.
    """
    if backend == "auto":
        return "bodhan" if is_indic(language) and bodhan.is_configured() else "vlm"
    return backend


def transcribe_page(page: Page, doc_label: str, profile: str | None = None, language: str = "auto") -> str:
    """Transcribes one page image with the vision model.

    Args:
        page (Page): Page with an image.
        doc_label (str): Name of the document, used in the prompt.
        profile (str | None): Model profile name from config.yaml.
        language (str): Expected language, or "auto" for no hint.

    Returns:
        str: The transcription, with the markers defined in the OCR prompt.

    Raises:
        LLMError: If the model call fails.
    """
    lang_line = OCR_LANGUAGE_LINE.format(language=language) if language != "auto" else ""
    messages = [
        {"role": "system", "content": OCR_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": OCR_USER.format(doc_label=doc_label, page_no=page.number,
                                                     language_line=lang_line)},
            image_part(page.image)]},
    ]
    return chat("ocr", messages, profile=profile, max_tokens=6000)


def run_ocr(doc: Document, doc_label: str, profile: str | None = None,
            progress: Callable[[float, str], None] | None = None,
            language: str = "auto", backend: str = "auto") -> Document:
    """Fills in the text of every page marked needs_ocr, in parallel.

    Changes `doc` in place. Pages with no ink are marked blank without a model
    call. Page images are kept, since the grader and the OCR re-check use them
    later.

    Args:
        doc (Document): The document to read.
        doc_label (str): Name of the document, used in prompts and messages.
        profile (str | None): Model profile name from config.yaml.
        progress (Callable[[float, str], None] | None): Optional
            progress(fraction, message) callback.
        language (str): Language hint, or "auto".
        backend (str): "auto", "vlm" or "bodhan".

    Returns:
        Document: The same document, with page text filled in.

    Raises:
        LLMError: If a vision model call fails.
    """
    todo = [p for p in doc.pages if p.needs_ocr and p.image is not None]
    if not todo:
        return doc
    engine = resolve_backend(backend, language)
    max_ink = float(load_config()["ingest"].get("blank_page_max_ink", 0) or 0)

    def work(p: Page) -> None:
        """Reads one page and stores the text, source and engine on it.

        Args:
            p (Page): The page to read.
        """
        text, used = None, engine
        if looks_blank(p.image, max_ink):  # no ink: skip the model call (see core/ingest.py)
            p.text, p.source, p.ocr_engine, p.needs_ocr = "", "blank-check", "none", False
            doc.warnings.append(f"{doc_label} page {p.number} looked empty and was not read. If it has very "
                                "faint writing, set ingest.blank_page_max_ink: 0 in config.yaml.")
            trace.emit("ocr", f"{doc_label}, page {p.number}: no ink found, treated as blank (not sent to OCR)")
            return
        if engine == "bodhan":
            try:
                text = bodhan.ocr_image(p.image, language if language != "auto" else "Hindi", profile)
            except bodhan.BodhanError as e:
                doc.warnings.append(f"{doc_label} page {p.number}: Bodhan OCR failed ({e}); used the vision model.")
                used = "vlm"
        if text is None:
            text = transcribe_page(p, doc_label, profile, language)
        p.text = "" if text.strip() == "[BLANK PAGE]" else text
        p.source = "ocr"
        p.ocr_engine = "bodhan" if used == "bodhan" else model_for("ocr", profile)["model"]
        p.needs_ocr = False
        trace.emit("ocr", f"Read {doc_label}, page {p.number} with {p.ocr_engine}"
                   + (" (blank page)" if not p.text else ""))

    with ThreadPoolExecutor(max_workers=max_parallel(profile)) as pool:
        futures = [submit(pool, work, p) for p in todo]
        for done, fut in enumerate(futures, start=1):
            fut.result()
            if progress:
                progress(done / len(todo), f"Reading {doc_label}: page {done}/{len(todo)}")
    return doc
