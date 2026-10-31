"""Build harder-to-read copies of a scanned dataset, to test OCR robustness.

Each kind of damage copies the chosen sheets with every page damaged the
same way, and keeps the teacher's marks unchanged: the student wrote the same
thing, so any drop in agreement comes from the pipeline failing to read it.

  blur    - out of focus phone photo (Gaussian blur)
  lowres  - photo taken from far away (shrunk to 700 px, then enlarged)
  skew    - page photographed at an angle (rotated 4 degrees)
  noisy   - grainy, heavily compressed photo (noise + JPEG quality 20)
  faded   - faint pencil / weak scan (low contrast, brighter)

Usage:
    python scripts/make_degraded.py --dataset eval/datasets/mendeley --sheets Student_12,Student_13,Student_14
        --kinds blur,faded
    # writes eval/datasets/mendeley_blur and eval/datasets/mendeley_faded
"""

import argparse
import csv
import io
import shutil
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip an image through JPEG compression.

    Args:
        img (Image.Image): Input image.
        quality (int): JPEG quality (1-95; lower = more artefacts).

    Returns:
        Image.Image: The compressed image, in RGB.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def damage(img: Image.Image, kind: str, seed: int = 0) -> Image.Image:
    """Apply one kind of damage to a page image.

    Args:
        img (Image.Image): The page, in RGB.
        kind (str): One of KINDS ("blur", "lowres", "skew", "noisy", "faded").
        seed (int): Random seed for the noise ("noisy" only).

    Returns:
        Image.Image: The damaged page ("skew" may be larger than the input).

    Raises:
        ValueError: If kind is unknown.
    """
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=max(1.5, img.height / 900)))
    if kind == "lowres":
        scale = 700 / max(img.size)
        small = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        return small.resize(img.size, Image.BILINEAR)
    if kind == "skew":
        return img.rotate(4, resample=Image.BICUBIC, expand=True, fillcolor=(235, 235, 230))
    if kind == "noisy":
        rng = np.random.default_rng(seed)
        a = np.asarray(img, dtype=np.int16) + rng.normal(0, 22, (img.height, img.width, 1)).astype(np.int16)
        return _jpeg(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), 20)
    if kind == "faded":
        return ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(0.35)).enhance(1.25)
    raise ValueError(f"unknown kind {kind}")


KINDS = ["blur", "lowres", "skew", "noisy", "faded"]


def main() -> None:
    """Write one damaged copy of the dataset per requested kind."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sheets", required=True, help="comma-separated sheet names")
    ap.add_argument("--kinds", default=",".join(KINDS))
    args = ap.parse_args()
    src = Path(args.dataset)
    only = [x.strip() for x in args.sheets.split(",") if x.strip()]
    with open(src / "human_marks.csv", encoding="utf-8") as f:
        human = [r for r in csv.DictReader(f) if r["sheet"] in only]

    for kind in args.kinds.split(","):
        out = src.parent / f"{src.name}_{kind}"
        (out / "sheets").mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name not in ("human_marks.csv", "calibration_sheets.txt"):
                shutil.copy(f, out / f.name)
        for name in only:
            pdf = src / "sheets" / f"{name}.pdf"
            pages = []
            for i, page in enumerate(pymupdf.open(pdf)):
                dpi = max(30, round(72 * 2000 / max(page.rect.width, page.rect.height)))  # ~2000 px long side
                pix = page.get_pixmap(dpi=dpi)
                pages.append(damage(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), kind, seed=i))
            pages[0].save(out / "sheets" / f"{name}.pdf", save_all=True, append_images=pages[1:], resolution=150)
        with open(out / "human_marks.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sheet", "question_id", "marks"])
            w.writeheader()
            w.writerows(human)
        print(f"Wrote {len(only)} sheets to {out}")


if __name__ == "__main__":
    main()
