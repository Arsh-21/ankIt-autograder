"""Make samples/answer_sheet_scan.png: the sample answer sheet drawn as an image.

The image is slightly rotated and noisy, to exercise the OCR path without a
real scan. It is typed text, so it is a much easier test than a real
handwritten scan.

Usage:
    python scripts/make_sample_scan.py
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
text = (ROOT / "samples" / "answer_sheet.txt").read_text(encoding="utf-8")

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
except OSError:
    font = ImageFont.load_default()

lines = text.splitlines()
img = Image.new("RGB", (1240, 60 + 38 * len(lines)), (250, 248, 240))
d = ImageDraw.Draw(img)
for i, line in enumerate(lines):
    d.text((60, 30 + 38 * i), line, fill=(20, 20, 60), font=font)

random.seed(0)
px = img.load()
for _ in range(4000):  # specks of noise
    x, y = random.randrange(img.width), random.randrange(img.height)
    px[x, y] = (120, 120, 120)
img = img.rotate(1.2, expand=True, fillcolor=(250, 248, 240)).filter(ImageFilter.GaussianBlur(0.6))
out = ROOT / "samples" / "answer_sheet_scan.png"
img.save(out)
print("saved", out)
