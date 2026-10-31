"""Step 1 of the pipeline: turn any uploaded file into pages.

No AI here. Each file becomes a Document made of Pages. A page has text if
the text could be read directly (typed .docx, digital PDF, .txt). If it
could not (scanned PDF, phone photo, image inside a .docx), the page keeps
an image and is marked `needs_ocr`; the OCR agent fills in its text later.
"""

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from .config import load_config

try:  # optional: iPhone photos (.heic)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIC = True
except ImportError:
    _HEIC = False

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
TEXT_EXT = {".txt", ".md", ".py", ".java", ".c", ".cpp", ".js", ".csv", ".tex", ".json"}


@dataclass
class Page:
    """One page of an uploaded file.

    Attributes:
        number (int): Page number, starting at 1.
        text (str): Page text; empty until OCR fills it for image pages.
        image (Image.Image | None): Page image, if there is one.
        needs_ocr (bool): True if the text must come from OCR.
        source (str): How the text was read: "text", "pdf-text", "ocr" or "pending-ocr".
        ocr_engine (str): Which OCR read the page (VLM model id or "bodhan").
        lines (list[tuple[float, str]]): For typed PDF pages, (y position 0-1 from
            the top, line text). Used to place annotations without asking a model.
    """

    number: int
    text: str = ""
    image: Image.Image | None = None
    needs_ocr: bool = False
    source: str = "text"
    ocr_engine: str = ""
    lines: list[tuple[float, str]] = field(default_factory=list)


@dataclass
class Document:
    """An uploaded file (or several files joined) as a list of pages.

    Attributes:
        name (str): File name, or the joined names for several files.
        kind (str): "pdf", "docx", "image", "notebook", "text" or "multi".
        pages (list[Page]): Pages in reading order.
        warnings (list[str]): Problems found while reading, shown in the report.
    """

    name: str
    kind: str
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def full_text(self, with_markers: bool = True) -> str:
        """Join the text of all non-empty pages.

        Args:
            with_markers (bool): Put a "[Page n]" line before each page.

        Returns:
            str: Page texts separated by blank lines.
        """
        parts = []
        for p in self.pages:
            if not p.text.strip():
                continue
            parts.append(f"[Page {p.number}]\n{p.text}" if with_markers else p.text)
        return "\n\n".join(parts)

    def transcript_markdown(self) -> str:
        """Render the pages as Markdown, noting how each page was read.

        Returns:
            str: Markdown transcript with one section per page.
        """
        out = [f"### {self.name}"]
        for p in self.pages:
            out.append(f"**Page {p.number}** _(read via {p.source})_\n\n{p.text or '_(empty)_'}")
        return "\n\n".join(out)


def ink_share(img: Image.Image) -> float:
    """Share of pixels clearly darker (by 25+ grey levels) than the page background.

    Measured against the page's own background, not an absolute level, so a
    faint pencil page or a faded scan still counts as written. Big images are
    first shrunk by plain pixel sampling (NEAREST), which keeps the share
    unbiased and memory small (a 600 dpi page is ~35 megapixels).

    Args:
        img (Image.Image): Page image.

    Returns:
        float: Share of ink pixels, between 0 and 1.
    """
    img = img.convert("L")
    scale = 1600 / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.NEAREST)
    g = np.asarray(img, dtype=np.int16)
    return float((g < np.median(g) - 25).mean())


def looks_blank(img: Image.Image | None, max_ink: float) -> bool:
    """Check whether a scanned page has (almost) no marks.

    Covers an empty page and a page with only faint mirrored show-through
    from the back of the sheet. Such pages skip OCR, because vision models
    "read" show-through as gibberish.

    Design note: on the Mendeley set, ghost pages measure <= 0.0002 and a
    faded (contrast x0.35) copy of a written page >= 0.0036, so the default
    0.0005 sits in the gap. A first version used an absolute darkness level
    and called faded pages blank; the degraded-scan set (scripts/make_degraded.py)
    caught that.

    Args:
        img (Image.Image | None): Page image; None is never blank.
        max_ink (float): Largest ink share still called blank; 0 turns the check off.

    Returns:
        bool: True if the page should be treated as blank.
    """
    if img is None or max_ink <= 0:
        return False
    return ink_share(img) < max_ink


def _shrink(img: Image.Image, cfg: dict) -> Image.Image:
    """Shrink a page image in place to the size the models get anyway.

    Some scans are 600 dpi (~35 megapixels a page); holding those for every
    sheet ran a 16 GB laptop out of memory. This is the same resize as
    llm.image_part, so what the model sees (and the cache key) does not change.

    Args:
        img (Image.Image): Page image.
        cfg (dict): The `ingest` block of config.yaml (uses image_max_side).

    Returns:
        Image.Image: The same image, at most image_max_side on its longest side.
    """
    side = cfg.get("image_max_side", 1600)
    if max(img.size) > side:
        img.thumbnail((side, side))
    return img


def _fix_image(img: Image.Image) -> Image.Image:
    """Apply the EXIF rotation tag and convert to RGB.

    Phone photos are often stored sideways with a rotation tag.

    Args:
        img (Image.Image): Image as opened from disk.

    Returns:
        Image.Image: Upright RGB image.
    """
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _load_pdf(path: Path, force_ocr: bool, cfg: dict) -> Document:
    """Read a PDF. Pages with little text or mostly images are rendered for OCR.

    Args:
        path (Path): PDF file.
        force_ocr (bool): Send every page to OCR even if it has a text layer.
        cfg (dict): The `ingest` block of config.yaml.

    Returns:
        Document: One Page per PDF page, up to max_pages.
    """
    doc = Document(name=path.name, kind="pdf")
    pdf = pymupdf.open(path)
    for i, page in enumerate(pdf):
        if i >= cfg["max_pages"]:
            doc.warnings.append(f"{path.name}: only the first {cfg['max_pages']} pages were read.")
            break
        text = page.get_text("text").strip()
        # how much of the page is covered by pictures (scans are ~100%)
        page_area = page.rect.width * page.rect.height or 1
        img_area = 0.0
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            img_area += max(0, x1 - x0) * max(0, y1 - y0)
        mostly_image = img_area / page_area > 0.3
        needs = force_ocr or len(text) < cfg["min_text_chars"] or mostly_image
        p = Page(number=i + 1, text="" if needs else text, needs_ocr=needs,
                 source="pending-ocr" if needs else "pdf-text")
        if needs:
            pix = page.get_pixmap(dpi=cfg["pdf_dpi"])
            p.image = _shrink(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), cfg)
        else:
            # typed page: keep a small picture for the annotated view, and line positions
            pix = page.get_pixmap(dpi=110)
            p.image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            height = page.rect.height or 1
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    txt = "".join(span["text"] for span in line["spans"]).strip()
                    if txt:
                        p.lines.append((line["bbox"][1] / height, txt))
        doc.pages.append(p)
    return doc


def _load_docx(path: Path) -> Document:
    """Read a .docx: typed text and tables as page 1, embedded pictures as OCR pages.

    Args:
        path (Path): .docx file.

    Returns:
        Document: Text page first, then one page per embedded image.
    """
    import docx  # python-docx

    doc = Document(name=path.name, kind="docx")
    d = docx.Document(str(path))
    lines = [para.text for para in d.paragraphs]
    for t_i, table in enumerate(d.tables, start=1):
        lines.append(f"\n[Table {t_i}]")
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    text = "\n".join(lines).strip()
    doc.pages.append(Page(number=1, text=text, source="text"))

    # pictures pasted into the document (e.g. a photo of handwritten working)
    with zipfile.ZipFile(path) as z:
        media = sorted(n for n in z.namelist() if n.startswith("word/media/"))
        for n in media:
            if Path(n).suffix.lower() not in IMAGE_EXT:
                continue
            try:
                img = _fix_image(Image.open(io.BytesIO(z.read(n))))
            except Exception:
                continue
            if min(img.size) < 100:  # skip icons / logos
                continue
            doc.pages.append(Page(number=len(doc.pages) + 1, image=img,
                                  needs_ocr=True, source="pending-ocr"))
    if len(doc.pages) > 1:
        doc.warnings.append(f"{path.name}: images inside the .docx are read after the typed "
                            "text, so their position in the document is approximate.")
    return doc


def _load_notebook(path: Path) -> Document:
    """Read a Jupyter notebook as one text page; code cells become ```python blocks.

    Args:
        path (Path): .ipynb file.

    Returns:
        Document: A single-page document.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code":
            chunks.append(f"```python\n{src}\n```")
        else:
            chunks.append(src)
    return Document(name=path.name, kind="notebook",
                    pages=[Page(number=1, text="\n\n".join(chunks))])


def load_document(path: str | Path, force_ocr: bool = False) -> Document:
    """Read one uploaded file into a Document.

    Args:
        path (str | Path): The file.
        force_ocr (bool): Send PDF pages to OCR even if they have a text layer.

    Returns:
        Document: The file's pages.

    Raises:
        ValueError: If the file type is not supported.
    """
    cfg = load_config()["ingest"]
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path, force_ocr, cfg)
    if ext == ".docx":
        return _load_docx(path)
    if ext in IMAGE_EXT or (ext == ".heic" and _HEIC):
        img = _shrink(_fix_image(Image.open(path)), cfg)
        return Document(name=path.name, kind="image",
                        pages=[Page(number=1, image=img, needs_ocr=True, source="pending-ocr")])
    if ext == ".ipynb":
        return _load_notebook(path)
    if ext in TEXT_EXT:
        return Document(name=path.name, kind="text",
                        pages=[Page(number=1, text=path.read_text(encoding="utf-8", errors="replace"))])
    if ext == ".heic":
        raise ValueError("For .heic photos run: pip install pillow-heif")
    if ext == ".doc":
        raise ValueError("Old .doc files are not supported. Save it as .docx or PDF first.")
    raise ValueError(f"Unsupported file type: {ext}")


def load_many(paths: list[str | Path], force_ocr: bool = False) -> Document:
    """Read several files for one role into a single Document.

    For example, 5 photos of one answer sheet. Pages are renumbered in the
    order given.

    Args:
        paths (list[str | Path]): Files; empty entries are skipped.
        force_ocr (bool): Send PDF pages to OCR even if they have a text layer.

    Returns:
        Document: All pages joined, with the warnings of every file.

    Raises:
        ValueError: If a file type is not supported.
    """
    paths = [p for p in paths if p]
    if len(paths) == 1:
        return load_document(paths[0], force_ocr)
    merged = Document(name=" + ".join(Path(p).name for p in paths), kind="multi")
    for p in paths:
        d = load_document(p, force_ocr)
        merged.warnings.extend(d.warnings)
        for page in d.pages:
            page.number = len(merged.pages) + 1
            merged.pages.append(page)
    return merged
