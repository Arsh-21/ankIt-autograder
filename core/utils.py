"""Small helpers with no AI: JSON extraction, quote checks, blank checks,
question-id handling and cross-references between questions."""

import json
import re
from typing import Any

from rapidfuzz import fuzz

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
STRUCK_RE = re.compile(r"\[STRUCK:[^\]]*\]", re.I)
_MARKERS_RE = re.compile(r"\[(illegible|BLANK PAGE)\]|\[STRUCK:[^\]]*\]|\[Page \d+\]|"
                         r"\[removed:[^\]]*\]|\(no answer written\)", re.I)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some local models print.

    Args:
        text (str): Raw model reply.

    Returns:
        str: The reply without reasoning blocks, stripped.
    """
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<think>", "").strip()


# A valid JSON escape (kept as is) or a lone backslash (doubled). The text is
# read left to right, so an already escaped "\\\\" pair is never split. An
# earlier one-step regex turned a correct "\\\\s" into "\\\\\\s" and broke the JSON.
_ESCAPE_TOKEN = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def _fix_escapes(text: str) -> str:
    """Double every backslash that does not start a valid JSON escape.

    Args:
        text (str): JSON text.

    Returns:
        str: The text with lone backslashes escaped.
    """
    return _ESCAPE_TOKEN.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", text)


# LaTeX such as \times, \theta, \frac, \beta, \right written with ONE backslash inside JSON.
# json would read \t, \f, \b, \r as tab, form feed, ... and corrupt the text.
_LATEX_ESCAPE = re.compile(r'(?<!\\)\\(?=[tbfr][a-zA-Z]{2,})')


def parse_json(text: str) -> Any:
    """Find and load the JSON in a model reply.

    Handles a ```json wrapper around the reply (even when the JSON itself
    contains ``` code blocks, e.g. a student's code), text before or after
    the JSON, a second copy of the JSON after the first (only the first is
    kept), trailing commas, raw newlines inside strings, and stray
    backslashes from LaTeX. As a last resort, a reply that ends like
    complete JSON is mended with json_repair (e.g. a quote inside a string
    that was not escaped, or a missing comma); a reply cut off half-way is
    never mended, so no part of it is silently lost.

    Args:
        text (str): Raw model reply.

    Returns:
        Any: The parsed JSON value (usually a dict or list).

    Raises:
        ValueError: If no JSON can be loaded (json.JSONDecodeError is a subclass).
    """
    t = strip_think(text).strip()
    if t.startswith("```"):  # drop the outer fence only; inner ``` blocks are part of the data
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON object found in model output")
    candidate = t[min(starts):max(t.rfind("}"), t.rfind("]")) + 1]
    candidate = _LATEX_ESCAPE.sub(r"\\\\", candidate)
    decoder = json.JSONDecoder(strict=False)
    last_error = None
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        for fixed in (attempt, _fix_escapes(attempt)):
            try:
                return decoder.raw_decode(fixed)[0]  # the first complete value; anything after it is ignored
            except json.JSONDecodeError as e:
                last_error = e
    if t.rstrip().endswith(("}", "]")):
        from json_repair import repair_json
        mended = repair_json(candidate, return_objects=True)
        if mended and isinstance(mended, (dict, list)):
            return mended
    raise last_error


def normalize_ws(text: str) -> str:
    """Collapse whitespace and lower-case a string for fuzzy comparison.

    Args:
        text (str): Any text; None is treated as empty.

    Returns:
        str: Lower-cased text with single spaces.
    """
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def evidence_supported(evidence: str, source_text: str, threshold: int = 80) -> bool:
    """Check that a quoted piece of evidence really appears in the source.

    partial_ratio lets a quote match part of the answer and tolerates small
    OCR and whitespace differences.

    Args:
        evidence (str): Quote given by the grader.
        source_text (str): The student's answer.
        threshold (int): Minimum rapidfuzz partial_ratio for a match.

    Returns:
        bool: True if the quote (or all its pieces) is found in the source.
    """
    ev, src = normalize_ws(evidence), normalize_ws(source_text)
    if not ev or not src:
        return False
    if len(ev) < 3:  # a one- or two-character quote (an MCQ letter like "B") must be a whole token
        return re.search(rf"(?<![^\W_]){re.escape(ev)}(?![^\W_])", src) is not None
    if ev in src or fuzz.partial_ratio(ev, src) >= threshold:
        return True
    # Graders often stitch one quote from several places: lines of a list,
    # cells of a table, text on both sides of a crossed-out word. Accept it
    # when every real piece (8+ characters) is on the sheet and those pieces
    # make up most of the quote.
    pieces = [normalize_ws(x).strip(" -•|.,;:'\"") for x in _QUOTE_SPLIT.split(evidence)]
    pieces = [x for x in pieces if len(x) >= 8]
    if len(pieces) < 2 or sum(map(len, pieces)) < 0.6 * len(ev):
        return False
    return all(x in src or fuzz.partial_ratio(x, src) >= threshold for x in pieces)


_QUOTE_SPLIT = re.compile(r"\n|\||\.\.\.|…|\s-\s|•|\[STRUCK:[^\]]*\]", re.I)


def remove_struck(text: str) -> str:
    """Remove crossed-out text ([STRUCK: ...]), which can never earn marks.

    Args:
        text (str): Answer text.

    Returns:
        str: The text with each struck block replaced by a space.
    """
    return STRUCK_RE.sub(" ", text or "")


def is_blank(text: str) -> bool:
    """Check whether nothing gradable is left after removing markers.

    A single letter counts as an answer (MCQ "D").

    Args:
        text (str): Answer text, possibly with [illegible], [STRUCK: ...] etc.

    Returns:
        bool: True if no letter or digit remains.
    """
    rest = _MARKERS_RE.sub(" ", text or "")
    return re.search(r"[^\W_]", rest) is None


# "1 / total 3", "Page: 2 of 6", "... 3 of 6", and OCR-garbled "Dano 5 of 6" / "Page 6 of r"
_PAGE_NO_RE = re.compile(r"^\W*(?:(?!marks?\b)[a-z]{1,5}[:.]?\s*)?\d{1,3}\s*(?:/|of|out of)\s*(?:total\s*)?"
                         r"(?:\d{1,3}|[a-z])\W*$|^\W*(?:page|pg)[:.]?\s*\d{1,3}\W*$", re.I)


def is_page_number(text: str) -> bool:
    """Check for page numbers and footers like "1 / total 3", "Page 2 of 6", "pg 4".

    Graders sometimes report these as text addressed to the grader.

    Args:
        text (str): A short piece of text.

    Returns:
        bool: True if the text is only a page number.
    """
    return bool(_PAGE_NO_RE.match(normalize_ws(text)))


def coverage(part: str, whole: str, threshold: int = 85) -> float:
    """Share of the lines of `part` that can be found in `whole`.

    Used to check that the segmenter copied an answer from the sheet and did
    not invent it. Lines shorter than 8 characters are ignored.

    Args:
        part (str): Text returned by the segmenter.
        whole (str): Full sheet text.
        threshold (int): Minimum rapidfuzz partial_ratio for a line to count as found.

    Returns:
        float: Share between 0 and 1; 1.0 if there are no lines to check.
    """
    lines = [normalize_ws(x) for x in (part or "").splitlines()]
    lines = [x for x in lines if len(x) >= 8]
    if not lines:
        return 1.0
    src = normalize_ws(whole)
    found = sum(1 for x in lines if x in src or fuzz.partial_ratio(x, src) >= threshold)
    return found / len(lines)


QUESTION_WORDS = (r"question|ques|qn|q|ans|answer|प्रश्न|प्र|उत्तर|प्रश्न\s*क्रमांक|கேள்வி|வினா|ప్రశ్న|প্রশ্ন|"
           r"ಪ್ರಶ್ನೆ|ചോദ്യം|પ્રશ્ન|ਪ੍ਰਸ਼ਨ|سوال")


_LEADING_LABEL = re.compile(rf"^\s*(?:(?:{QUESTION_WORDS})\s*\.?\s*(?:no\.?\s*)?)?\(?(\d{{1,3}}\s*\(?[a-z]?\)?)\s*[.):\-]?\s*",
                            re.I)


def strip_label(text: str, qid: str) -> str:
    """Remove the question's own label at the start of its answer ("Q23.", "23)", "Ans 23:", "प्रश्न २३.").

    Only a label with this question's number is removed, so an answer that is just a number
    ("28.") is kept as it is.

    Args:
        text (str): Answer text.
        qid (str): The question's id, e.g. "Q23".

    Returns:
        str: The text without its leading label.
    """
    from .lang import to_ascii_digits

    t = (text or "").strip()
    m = _LEADING_LABEL.match(to_ascii_digits(t))
    if m and normalize_qid(m.group(1)) == normalize_qid(qid):
        return t[m.end():].strip()
    return t


def only_question(answer: str, question: str) -> bool:
    """Whether an "answer" is nothing but the question itself (copied by the student, or handed back
    by the segmenter for a question that was not answered).

    True when the question text covers the whole answer, or when the answer is a large part of the
    question text (a short option such as "(b) ReLU" is never treated as a copy of an MCQ).

    Args:
        answer (str): Answer text without its label.
        question (str): The question text from the rubric.

    Returns:
        bool: True if nothing but the question is written.
    """
    a, q = normalize_ws(remove_struck(answer)), normalize_ws(question)
    if len(a) < 10 or len(q) < 10:
        return False
    hit = fuzz.partial_ratio_alignment(q, a) if len(q) <= len(a) else None
    if hit and hit.score >= 90:  # the question is inside the answer: is anything else written?
        rest = a[:hit.dest_start] + " " + a[hit.dest_end:]
        return is_blank(re.sub(r"^\W*\w{0,4}\s*\d{1,3}\s*[.):\-]", "", rest.strip()))  # at most a label is left
    hit = fuzz.partial_ratio_alignment(a, q)  # the answer is inside the question
    return hit.score >= 90 and len(a) >= 0.5 * len(q)


def normalize_qid(qid: str) -> str:
    """Reduce a question id to digits and letters.

    'Q1', 'q 1', 'Question 1', '1.', 'प्रश्न १' -> '1';  'Q2(a)' -> '2a'.

    Args:
        qid (str): Question id as written in a rubric or on a sheet.

    Returns:
        str: Normalised id.
    """
    from .lang import to_ascii_digits

    s = to_ascii_digits(str(qid)).strip().lower()
    s = re.sub(rf"^({QUESTION_WORDS})\s*\.?\s*", "", s)
    return re.sub(r"[^0-9a-z]", "", s)


def round_half(x: float) -> float:
    """Round to the nearest half mark.

    Args:
        x (float): Marks.

    Returns:
        float: x rounded to a multiple of 0.5.
    """
    return round(x * 2) / 2


def clamp(x: float, lo: float, hi: float) -> float:
    """Limit a value to a range.

    Args:
        x (float): Value.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: x limited to [lo, hi].
    """
    return max(lo, min(hi, x))


def to_float(x: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising.

    Args:
        x (Any): Value from a model reply, e.g. "2.5" or None.
        default (float): Value returned when conversion fails.

    Returns:
        float: The converted value, or default.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# "see Q3", "refer answer 2(b)", "as explained in question 5", "same as Q.4",
# "already answered in 3", "प्रश्न 3 देखें" ...
_REF_RE = re.compile(
    r"(?:\bsee|\brefer(?:\s+to)?|\breferred\s+(?:to\s+)?in|\bas\s+(?:explained|shown|discussed|mentioned|written|"
    r"done|given|stated|derived|answered)(?:\s+above)?\s+in|\bsame\s+as(?:\s+in)?|\balready\s+(?:explained|answered|"
    r"written|done)\s+in)\s+(?:the\s+)?(?:answer\s+(?:to|of)\s+|ans(?:wer)?\.?\s*|question\s*|ques\.?\s*|q\.?\s*)?"
    r"(?:no\.?\s*)?(\d{1,3}\s*(?:\(?[a-z]\)|[a-z]\b)?)"
    r"|प्रश्न\s*(?:संख्या\s*)?(\d{1,3})\s*(?:में|देखें|का\s+उत्तर|के\s+उत्तर)", re.I)


# One "question number + chosen option" pair, as students write MCQ answers in a grid:
# "7) c", "7-c", "7. (c)", "7 = C", "| 7 | c |", "७) c". The option must stand alone (not the start of a word).
_GRID_PAIR = re.compile(r"(?<![\w.])(\d{1,3})\s*(?:[-–—.):=|]\s*)*\(?([A-Da-d])\)?(?![\w])")
_OPTION = re.compile(r"^\(?([A-Da-d])\)?$")


def grid_answers(text: str) -> dict[str, str]:
    """Multiple-choice answers written as a grid or table, read without a model.

    Three layouts are understood: rows of pairs ("1) b   2) c   3) a"), table
    rows ("| 7 | c |"), and a row of question numbers with a row of options
    under it ("1 2 3 4" over "b c a d"). A line counts only if it holds two or
    more pairs, or nothing but a pair, so working such as "2) b is wrong" in a
    long answer is not mistaken for one.

    Args:
        text (str): Sheet transcription.

    Returns:
        dict[str, str]: Question number (ASCII digits) -> the pair as written
            (e.g. "7) c"), or "7: c" for the numbers-over-options layout.
    """
    from .lang import to_ascii_digits

    out: dict[str, str] = {}
    lines = [ln.strip() for ln in to_ascii_digits(text or "").splitlines()]
    for k, line in enumerate(lines):
        pairs = list(_GRID_PAIR.finditer(line))
        rest = _GRID_PAIR.sub("", line).strip(" |\t-–—,;")
        if len(pairs) >= 2 or (pairs and not rest):
            for m in pairs:
                out.setdefault(m.group(1), m.group(0).strip(" |"))
            continue
        cells = [c for c in re.split(r"[\s|]+", line) if c]
        if len(cells) >= 3 and all(c.isdigit() for c in cells):  # numbers over options
            nxt = next((ln for ln in lines[k + 1:] if ln), "")
            opts = [c for c in re.split(r"[\s|]+", nxt) if c]
            if len(opts) == len(cells) and all(_OPTION.match(o) for o in opts):
                for n, o in zip(cells, opts, strict=True):
                    out.setdefault(n, f"{n}: {_OPTION.match(o).group(1)}")
    return out


def looks_like_grid(text: str) -> bool:
    """Whether a sheet seems to have MCQ answers written as a grid (four or more pairs found)."""
    return len(grid_answers(text)) >= 4


def find_references(text: str, qids: list[str], own: str) -> list[str]:
    """Find the questions an answer explicitly points to ("as explained in Q3").

    Args:
        text (str): The student's answer.
        qids (list[str]): Question ids in the rubric.
        own (str): Id of the question this answer belongs to.

    Returns:
        list[str]: Referenced rubric ids in order of appearance, never `own`.
    """
    by_norm = {normalize_qid(q): q for q in qids}
    out = []
    for m in _REF_RE.finditer(text or ""):
        ref = normalize_qid(m.group(1) or m.group(2) or "")
        target = by_norm.get(ref) or by_norm.get(re.sub(r"[a-z]+$", "", ref))  # "3b" -> "3" if no part 3b
        if target and target != own and target not in out:
            out.append(target)
    return out


# Words in a QUESTION that say it builds on an earlier one.
_DEP_HINT = re.compile(
    r"\b(?:hence|using (?:this|that|these|the above|your|the (?:result|value|answer|expression|model))|"
    r"from (?:the )?(?:above|previous)|(?:in|of) (?:the )?previous (?:part|question)|"
    r"(?:obtained|found|derived|calculated|computed) (?:in|above)|above (?:result|answer|value|expression))\b", re.I)
_DEP_REF = re.compile(r"\b(?:question|ques\.?|q\.?|part)\s*\(?(\d{1,3}\s*\(?[a-z]?\)?|[a-z])\)?(?![\w])", re.I)


def infer_dependencies(questions: list[dict]) -> dict[str, list[str]]:
    """Work out which questions build on which, from the question text.

    Reads the QUESTION text, not the student's answer: "Using your answer
    to Question 3, ...", "Hence find ...", "From part (a), ...". A sub-part
    with "hence" or "using the above" depends on the part before it.

    Args:
        questions (list[dict]): Rubric questions with "id" and "text".

    Returns:
        dict[str, list[str]]: {qid: [ids it depends on]}, only for questions with dependencies.
    """
    ids = [q["id"] for q in questions]
    norm = [normalize_qid(i) for i in ids]
    by_norm = dict(zip(norm, ids, strict=False))
    out: dict[str, list[str]] = {}
    for k, q in enumerate(questions):
        own, text, deps = norm[k], q.get("text", ""), []
        parent = re.match(r"\d*", own).group()
        for m in _DEP_REF.finditer(text):
            ref = re.sub(r"[^0-9a-z]", "", m.group(1).lower())
            target = by_norm.get(ref if ref[:1].isdigit() else parent + ref)
            if target and target != q["id"] and target not in deps:
                deps.append(target)
        if not deps and _DEP_HINT.search(text) and k > 0 and parent and \
                re.match(r"\d*", norm[k - 1]).group() == parent and own != parent:
            deps.append(ids[k - 1])  # "hence ..." in part (b): builds on part (a)
        if deps:
            out[q["id"]] = deps
    return out
