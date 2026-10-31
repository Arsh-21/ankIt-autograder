"""Draws the grading on the student's pages like a teacher with a red pen.

Each answer gets a tick, cross or half-tick, a circled score, a short note
on what was missed, and a "?" where a human should check. The circled total
goes at the top of the first page.

No AI here. Positions come from agents/locator.py, checked against where the
answer is in the page transcription; failing both, answers are placed down
the margin in question order. The page is widened
with a margin strip so nothing covers the student's writing.
"""

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ingest import Document
from .lang import has_script
from .utils import normalize_ws

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
Color = tuple[int, int, int]
RED = (196, 38, 38)
ORANGE = (214, 120, 0)
MARGIN_SHARE = 0.34      # margin strip width, as a share of the page width


def _font(text: str, size: int) -> ImageFont.FreeTypeFont:
    """Pick a handwriting font that can draw the text.

    Kalam covers Latin and Devanagari, Kavivanar covers Tamil, Caveat is the
    default for Latin.

    Args:
        text (str): Text to be drawn.
        size (int): Font size in pixels.

    Returns:
        ImageFont.FreeTypeFont: The font, or Pillow's default if the file is missing.
    """
    name = "Kavivanar-Regular.ttf" if has_script(text, "Tamil") else \
           "Kalam-Regular.ttf" if has_script(text, "Devanagari") else "Caveat.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def _drawable(text: str) -> bool:
    """Check that the bundled fonts cover the text's script (others show as boxes).

    Args:
        text (str): Text to be drawn.

    Returns:
        bool: False if the text uses a script without a bundled font.
    """
    other = any(has_script(text, s) for s in ["Bengali", "Gurmukhi", "Gujarati", "Odia", "Telugu",
                                               "Kannada", "Malayalam", "Arabic"])
    return not other


def _wobbly(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], rng: random.Random,
            width: int, color: Color) -> None:
    """Draw a slightly shaky pen stroke through the given points.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        points (list[tuple[float, float]]): Stroke points in pixels.
        rng (random.Random): Source of the jitter.
        width (int): Line width.
        color (Color): RGB colour.
    """
    jit = [(x + rng.uniform(-1.5, 1.5), y + rng.uniform(-1.5, 1.5)) for x, y in points]
    draw.line(jit, fill=color, width=width, joint="curve")


def _tick(draw: ImageDraw.ImageDraw, x: float, y: float, s: float, rng: random.Random, w: int,
          color: Color = RED) -> None:
    """Draw a tick mark.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        x (float): Left edge in pixels.
        y (float): Vertical centre in pixels.
        s (float): Size in pixels.
        rng (random.Random): Source of the jitter.
        w (int): Line width.
        color (Color): RGB colour.
    """
    _wobbly(draw, [(x, y), (x + s * 0.35, y + s * 0.4), (x + s, y - s * 0.55)], rng, w, color)


def _cross(draw: ImageDraw.ImageDraw, x: float, y: float, s: float, rng: random.Random, w: int,
           color: Color = RED) -> None:
    """Draw a cross.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        x (float): Left edge in pixels.
        y (float): Vertical centre in pixels.
        s (float): Size in pixels.
        rng (random.Random): Source of the jitter.
        w (int): Line width.
        color (Color): RGB colour.
    """
    _wobbly(draw, [(x, y - s * 0.45), (x + s * 0.8, y + s * 0.4)], rng, w, color)
    _wobbly(draw, [(x + s * 0.8, y - s * 0.45), (x, y + s * 0.4)], rng, w, color)


def _half_tick(draw: ImageDraw.ImageDraw, x: float, y: float, s: float, rng: random.Random, w: int,
               color: Color = RED) -> None:
    """Draw a tick with a slash through it, for part marks.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        x (float): Left edge in pixels.
        y (float): Vertical centre in pixels.
        s (float): Size in pixels.
        rng (random.Random): Source of the jitter.
        w (int): Line width.
        color (Color): RGB colour.
    """
    _tick(draw, x, y, s, rng, w, color)
    _wobbly(draw, [(x + s * 0.15, y + s * 0.35), (x + s * 0.75, y - s * 0.2)], rng, max(2, w - 1), color)


def _circled(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.FreeTypeFont,
             rng: random.Random, w: int, color: Color = RED) -> float:
    """Write text and draw a hand-drawn ellipse around it.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        x (float): Left edge of the text in pixels.
        y (float): Vertical centre in pixels.
        text (str): Text to write, e.g. "3/5".
        font (ImageFont.FreeTypeFont): Font for the text.
        rng (random.Random): Source of the jitter.
        w (int): Line width of the ellipse.
        color (Color): RGB colour.

    Returns:
        float: x position of the right edge of the ellipse.
    """
    box = draw.textbbox((x, y), text, font=font, anchor="lm")
    pad = (box[3] - box[1]) * 0.35
    draw.text((x, y), text, font=font, fill=color, anchor="lm")
    draw.ellipse([box[0] - pad + rng.uniform(-2, 2), box[1] - pad, box[2] + pad + rng.uniform(-2, 2), box[3] + pad],
                 outline=color, width=w)
    return box[2] + pad


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Split text into lines that fit a width.

    Args:
        text (str): Text to wrap.
        font (ImageFont.FreeTypeFont): Font used to measure.
        max_w (int): Maximum line width in pixels.
        draw (ImageDraw.ImageDraw): Canvas used to measure text.

    Returns:
        list[str]: The lines.
    """
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = f"{cur} {wd}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _short_note(q: dict) -> str:
    """Write the margin note for a question.

    Args:
        q (dict): Question result with "awarded", "max_marks", "info" and "criteria".

    Returns:
        str: "Good", "Not attempted", "Missing: <first missed criterion>", or "".
    """
    if q["awarded"] >= q["max_marks"]:
        return "Good"
    if "Not attempted" in q.get("info", []):
        return "Not attempted"
    missed = [c for c in q["criteria"] if c["awarded"] < c["max"]]
    if not missed:
        return ""
    desc = missed[0]["description"].strip().rstrip(".")
    return ("Missing: " + desc) if len(desc) < 70 else ("Missing: " + desc[:67] + "...")


def _find_in_pages(answer: str, pages: list) -> tuple[int, float] | None:
    """Find where an answer is written by looking for its first line in each page's text.

    The transcription follows the page from top to bottom, so the line's
    position in the page text is a fair guess of its height on the page.

    Args:
        answer (str): The answer text as segmented.
        pages (list[Page]): Pages with "number" and "text".

    Returns:
        tuple[int, float] | None: (page number, y as share of page height), or None if not found.
    """
    first = next((ln.strip() for ln in (answer or "").splitlines() if len(ln.strip()) >= 3), "")
    key = normalize_ws(first)[:60]
    if not key:
        return None
    for p in pages:
        lines = [normalize_ws(ln) for ln in (p.text or "").splitlines()]
        lines = [ln for ln in lines if ln]
        for k, ln in enumerate(lines):
            if ln.startswith(key) or (len(key) >= 12 and key in ln):
                return p.number, 0.05 + 0.9 * (k + 0.5) / len(lines)
    return None


def _place(questions: list[dict], pages: list, positions: dict) -> dict[int, list[tuple[float, dict]]]:
    """Decide the page and height of every question's mark.

    In order of trust: the locator's position on a page that the segmenter or
    the transcription also points to; the answer's place in the page
    transcription; any page the locator found it on; the segmenter's first
    page; and last, an even spread over the pages in question order.

    Args:
        questions (list[dict]): Question results with "id", optional "pages" and "answer_text".
        pages (list[Page]): The sheet's pages that have images.
        positions (dict): {page number as str: {qid: y as share of page height}} from the locator.

    Returns:
        dict[int, list[tuple[float, dict]]]: {page number: [(y share, question), ...]} sorted by y.
    """
    page_nums = [p.number for p in pages]
    placed: dict[int, list[tuple[float | None, dict]]] = {n: [] for n in page_nums}
    unplaced = []
    for q in questions:
        found = [n for n in page_nums if q["id"] in positions.get(str(n), {})]
        seg = [n for n in (q.get("pages") or []) if n in placed]
        txt = _find_in_pages(q.get("answer_text", ""), pages) if "Not attempted" not in q.get("info", []) else None
        agree = [n for n in found if n in seg or (txt and n == txt[0])]
        if agree:
            placed[agree[0]].append((positions[str(agree[0])][q["id"]], q))
        elif txt:
            placed[txt[0]].append((txt[1], q))
        elif found:
            placed[found[0]].append((positions[str(found[0])][q["id"]], q))
        elif seg:
            placed[seg[0]].append((None, q))
        else:
            unplaced.append(q)
    for q in unplaced:  # no page info at all: spread over the pages in order
        page = page_nums[min(len(page_nums) - 1, questions.index(q) * len(page_nums) // max(1, len(questions)))]
        placed[page].append((None, q))
    out: dict[int, list[tuple[float, dict]]] = {}
    for n, items in placed.items():  # questions without a height share the free space in order
        k = len(items)
        filled = [(y if y is not None else 0.12 + 0.8 * i / max(1, k), q) for i, (y, q) in enumerate(items)]
        out[n] = sorted(filled, key=lambda t: t[0])
    return out


def _fit(d: ImageDraw.ImageDraw, items: list[tuple[float, dict]], H: int, top: float, big: int, small: int,
         mw: int) -> tuple[list[tuple[float, dict, list[str]]], float]:
    """Lay out one page's marks in the margin so that every one of them stays on the page.

    Each mark sits as close to its answer as it can, below the one before.
    If they don't all fit, the notes are left out first, and then everything
    is drawn smaller; a last pass from the bottom pulls marks back up onto
    the page.

    Args:
        d (ImageDraw.ImageDraw): Canvas, used to measure text.
        items (list[tuple[float, dict]]): (y as share of page height, question) sorted by y.
        H (int): Page height in pixels.
        top (float): Highest y (pixels) a mark may use.
        big (int): Font size for scores.
        small (int): Font size for labels and notes.
        mw (int): Margin width in pixels.

    Returns:
        tuple[list[tuple[float, dict, list[str]]], float]: (centre y in pixels, question, note lines) per
            mark, and the scale the sizes were multiplied by.
    """
    if not items:
        return [], 1.0
    bottom_limit = H - small * 0.8
    space = bottom_limit - top - big * 0.5

    def blocks(scale: float, notes: bool) -> list[tuple[float, dict, list[str], float]]:
        b, sm = big * scale, max(10, int(small * scale))
        out = []
        for y_frac, q in items:
            note = _short_note(q) if notes else ""
            lines = _wrap(note, _font(note, sm), int(mw * 0.86), d)[:2] if note and _drawable(note) else []
            out.append((y_frac * H, q, lines, b * 1.15 + len(lines) * sm * 1.1))
        return out

    scale, rows = 1.0, blocks(1.0, True)
    if sum(r[3] for r in rows) > space:
        rows = blocks(1.0, False)  # too crowded for notes: keep the marks only
        need = sum(r[3] for r in rows)
        if need > space:
            scale = max(0.35, space / need)
            rows = blocks(scale, False)
    half = big * scale * 0.5
    ys, prev_end = [], top + half
    for target, _q, _lines, h in rows:  # top to bottom: near the answer, below the previous mark
        y = max(target, prev_end)
        ys.append(y)
        prev_end = y + h
    limit = bottom_limit
    for k in range(len(rows) - 1, -1, -1):  # bottom to top: nothing below the page
        h = rows[k][3]
        if ys[k] + h - half > limit:
            ys[k] = limit - h + half
        limit = ys[k] - half
    return [(y, q, lines) for y, (_t, q, lines, _h) in zip(ys, rows, strict=True)], scale


def annotate(report: dict, sheet: Document, positions: dict | None = None) -> list[Image.Image]:
    """Draw the marks and notes on the student's pages.

    Args:
        report (dict): Sheet report with "questions", "total", "max_total", "needs_review".
        sheet (Document): The answer sheet.
        positions (dict | None): Locator positions; defaults to report["annotation_positions"].

    Returns:
        list[Image.Image]: Annotated page images, empty if the sheet has no page images.
    """
    pages = [p for p in sheet.pages if p.image is not None]
    if not pages:
        return []
    positions = positions or report.get("annotation_positions") or {}
    placed = _place(report["questions"], pages, positions)
    rng = random.Random(int(hashlib.md5(sheet.name.encode()).hexdigest()[:8], 16))
    out = []
    for i, page in enumerate(pages):
        img = page.image.convert("RGB")
        W, H = img.size
        mw = int(W * MARGIN_SHARE)
        bg = img.getpixel((min(5, W - 1), min(5, H - 1)))
        canvas = Image.new("RGB", (W + mw, H), bg)
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        stroke = max(3, W // 300)
        d.line([(W + 8, 0), (W + 8, H)], fill=(230, 160, 160), width=max(2, stroke - 1))
        big = max(22, W // 24)
        small = max(16, W // 40)
        x0 = W + 8 + mw * 0.06

        top = 0.0  # where the per-question marks may start
        if i == 0:  # circled total at the top of the first page
            ty = int(max(H * 0.04, big * 0.9))
            total = f"{report['total']:g} / {report['max_total']:g}"
            end = _circled(d, x0 + big * 0.2, ty, total, _font(total, int(big * 1.15)), rng, stroke)
            if report.get("needs_review"):
                d.text((end + big * 0.4, ty), "check", font=_font("check", small), fill=ORANGE, anchor="lm")
            top = ty + big * 1.1

        items = placed.get(page.number, [])
        rows, scale = _fit(d, items, H, top, big, small, mw)
        b, sm, s = int(big * scale), max(10, int(small * scale)), big * 0.9 * scale
        for (y, q, note_lines) in rows:
            label = q["id"]
            lf = _font(label, sm)
            d.text((x0, y), label, font=lf, fill=RED, anchor="lm")
            xs = x0 + d.textlength(label, font=lf) + sm * 0.5
            w = max(2, int(stroke * max(scale, 0.6)))
            if q["awarded"] >= q["max_marks"]:
                _tick(d, xs, y, s, rng, w)
            elif q["awarded"] <= 0:
                _cross(d, xs, y, s, rng, w)
            else:
                _half_tick(d, xs, y, s, rng, w)
            score = f"{q['awarded']:g}/{q['max_marks']:g}"
            end = _circled(d, xs + s * 1.4, y, score, _font(score, b), rng, w)
            if q.get("needs_review") and not q.get("human_reviewed"):
                d.text((end + sm * 0.5, y), "?", font=_font("?", b), fill=ORANGE, anchor="lm")
            for j, line in enumerate(note_lines):
                d.text((x0, y + b * 0.95 + j * sm * 1.1), line, font=_font(line, sm), fill=RED, anchor="lm")
        out.append(canvas)
    return out


def save_pdf(images: list[Image.Image], path: str | Path) -> str:
    """Save page images as one PDF.

    Args:
        images (list[Image.Image]): Pages in order; must not be empty.
        path (str | Path): Output file.

    Returns:
        str: The output path.
    """
    rgb = [im.convert("RGB") for im in images]
    rgb[0].save(path, save_all=True, append_images=rgb[1:], resolution=110)
    return str(path)
