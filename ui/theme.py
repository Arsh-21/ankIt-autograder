"""Look and feel: an examiner's desk. Sheets of white paper lie on graph
paper with a teacher's faint pencil working in one corner, under a header
with the red double margin rule of an answer booklet; text is in blue-black
ink, marks are in red pen (Kalam, an Indian handwriting typeface), and
"check" is an amber rubber stamp."""

import base64
import io
from functools import lru_cache
from pathlib import Path

import gradio as gr

INK = gr.themes.Color(
    name="ink", c50="#EEF2F9", c100="#DCE4F2", c200="#B9C8E4", c300="#8FA6D1", c400="#5F7DB8",
    c500="#3B5C9E", c600="#2A4884", c700="#1F3A6E", c800="#182E58", c900="#122244", c950="#0B1630")


def build_theme() -> gr.themes.Base:
    """Gradio theme: desk-and-ink colours, IBM Plex type, Indian-script fonts, dark mode.

    Returns:
        gr.themes.Base: The theme.
    """
    font = [gr.themes.GoogleFont("IBM Plex Sans"), gr.themes.GoogleFont("IBM Plex Sans Devanagari"),
            gr.themes.GoogleFont("Noto Sans Tamil"), gr.themes.GoogleFont("Kalam"), "system-ui", "sans-serif"]
    return gr.themes.Base(
        primary_hue=INK, secondary_hue=INK, neutral_hue=gr.themes.colors.slate,
        font=font, font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#E8EDF1", body_background_fill_dark="#0E1621",
        body_text_color="#1D2B53", body_text_color_dark="#E3E9F2",
        body_text_color_subdued="#5E6B7E", body_text_color_subdued_dark="#9AA9BE",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#152031",
        background_fill_secondary="#F7FAF9", background_fill_secondary_dark="#111B29",
        block_background_fill="#FFFFFF", block_background_fill_dark="#152031",
        block_border_color="#D5DEE7", block_border_color_dark="#26354B",
        block_border_width="1px", block_shadow="none", block_shadow_dark="none",
        block_label_text_color="#3D4E6A", block_label_text_color_dark="#B5C2D4",
        block_title_text_weight="600",
        input_background_fill="#FBFCFD", input_background_fill_dark="#0F1826",
        input_border_color="#C9D6E3", input_border_color_dark="#2A3A52",
        input_border_color_focus="#2A4884", input_border_color_focus_dark="#8FA6D1",
        button_primary_background_fill="#1D2B53", button_primary_background_fill_hover="#141F3E",
        button_primary_background_fill_dark="#3B5C9E", button_primary_background_fill_hover_dark="#2A4884",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark="#1A2840",
        button_secondary_border_color="#C9D6E3", button_secondary_border_color_dark="#2A3A52",
        button_secondary_text_color="#1D2B53", button_secondary_text_color_dark="#DCE4F2",
        checkbox_label_background_fill_selected="#EEF2F9",
        checkbox_label_background_fill_selected_dark="#1F3A6E",
        border_color_accent="#2A4884", color_accent_soft="#EEF2F9", color_accent_soft_dark="#1A2840",
    )


@lru_cache(maxsize=1)
def pencil_notes() -> str:
    """Faint pencil working for the page background, drawn once as a transparent PNG.

    Returns:
        str: A data: URL of the image ("" if Pillow or the font is missing).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Kalam-Regular.ttf"),
                                  34)
    except (ImportError, OSError):
        return ""
    lines = ["out = (W - K + 2P) / S + 1", "     = (64 - 5 + 4) / 2 + 1 = 32", "",
             "dL/dw = (y - t) . y(1 - y) . x", "", "प्रश्न 3.  32 x 32 x 16", "softmax(x - c) = softmax(x)"]
    img = Image.new("RGBA", (760, 520), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 62), line, font=font, fill=(40, 52, 80, 30))
    d.ellipse((540, 360, 720, 470), outline=(200, 32, 47, 34), width=4)  # a faint red-pen circle round a mark
    d.text((585, 385), "18/20", font=font, fill=(200, 32, 47, 40))
    img = img.rotate(-7, resample=Image.BICUBIC, expand=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _css() -> str:
    """The stylesheet, with the pencil-notes background filled in."""
    css = (Path(__file__).resolve().parent / "style.css").read_text(encoding="utf-8")
    notes = pencil_notes()
    return css.replace("__PENCIL_NOTES__", f'url("{notes}")' if notes else "none")


CSS = _css()
