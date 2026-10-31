"""Adapter for the Bodhan AI APIs given by the course (console.bodhan.ai/api-docs).

  ocr_image(img, language)          -> text      POST /v1/chat/completions, model "indic-ocr"
  translate(text, source, target)   -> text      POST /translate

Indic OCR takes one whole page as a PNG/JPEG (no PDFs), finds the script by
itself (there is no language field) and returns the page as Markdown in
choices[0].message.content. Unlike the vision-LLM OCR it takes no custom
instructions, so it does not mark crossed-out or illegible text.

Defaults below follow the API docs; the `bodhan:` block in config.yaml can
override them. The key goes in .env as BODHAN_API_KEY. With the `mock`
profile both functions return fake text, so the code path is testable
without a key.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import TYPE_CHECKING

from PIL import Image

from .config import load_config, model_for
from .lang import code_for

if TYPE_CHECKING:
    import httpx


class BodhanError(RuntimeError):
    """Raised when a Bodhan call fails or Bodhan is not set up."""


def _cfg() -> dict:
    """Get the `bodhan:` block of config.yaml.

    Returns:
        dict: Bodhan settings.

    Raises:
        BodhanError: If the block is missing or empty.
    """
    cfg = load_config().get("bodhan") or {}
    if not cfg:
        raise BodhanError("No `bodhan:` block in config.yaml.")
    return cfg


def is_configured() -> bool:
    """Check whether Bodhan can be used.

    Returns:
        bool: True if BODHAN_API_KEY is set and config.yaml does not mark Bodhan as a placeholder.
    """
    cfg = load_config().get("bodhan") or {}
    return bool(os.getenv("BODHAN_API_KEY")) and not cfg.get("placeholder", False)


def _is_mock(profile: str | None) -> bool:
    """Check whether the profile uses the mock model for OCR.

    Args:
        profile (str | None): Config profile; None uses the active one.

    Returns:
        bool: True for the mock profile.
    """
    return model_for("ocr", profile)["provider"] == "mock"


def _headers(cfg: dict) -> dict:
    """Build the auth header.

    Args:
        cfg (dict): Bodhan settings (auth_header, auth_prefix).

    Returns:
        dict: {header name: prefix + key}.

    Raises:
        BodhanError: If BODHAN_API_KEY is empty.
    """
    key = os.getenv("BODHAN_API_KEY")
    if not key:
        raise BodhanError("BODHAN_API_KEY is empty. Add it to .env.")
    return {cfg.get("auth_header", "Authorization"): f"{cfg.get('auth_prefix', 'Bearer ')}{key}"}


def _client() -> httpx.Client:
    """Create an HTTP client with the timeout from config.yaml.

    Returns:
        httpx.Client: A new client; the caller closes it.
    """
    import httpx  # installed with the openai package

    return httpx.Client(timeout=load_config()["llm"]["timeout_seconds"])


def _post(cfg: dict, path: str, body: dict) -> dict:
    """POST JSON to Bodhan, retrying twice on rate limits.

    Args:
        cfg (dict): Bodhan settings (base_url and auth).
        path (str): API path, e.g. "/translate".
        body (dict): JSON body.

    Returns:
        dict: Parsed JSON reply.

    Raises:
        BodhanError: On network errors, HTTP errors, or repeated rate limiting.
    """
    url = cfg.get("base_url", "https://api.bodhan.ai").rstrip("/") + path
    for attempt in range(3):
        headers = _headers(cfg)
        try:
            with _client() as c:
                r = c.post(url, headers=headers, json=body)
        except Exception as e:  # noqa: BLE001 - network trouble: the caller falls back to the vision LLM
            raise BodhanError(f"Could not reach Bodhan ({type(e).__name__})") from e
        if r.status_code == 429 and attempt < 2:  # rate limited: wait and try again
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise BodhanError(f"Bodhan error {r.status_code} on {path}: {r.text[:300]}")
        return r.json()
    raise BodhanError(f"Bodhan kept rate-limiting {path}")


def ocr_image(img: Image.Image, language: str, profile: str | None = None) -> str:
    """OCR one page with Bodhan Indic OCR.

    Args:
        img (Image.Image): Page image.
        language (str): Language name; only used by the mock, as Bodhan detects the script.
        profile (str | None): Config profile; None uses the active one.

    Returns:
        str: Page text as Markdown, or "[BLANK PAGE]" if empty.

    Raises:
        BodhanError: If Bodhan is switched off, the call fails or the reply has no text.
    """
    if _is_mock(profile):
        return f"(mock Bodhan OCR, {language})\nप्रश्न 1. ग्रेडिएंट छोटा हो जाता है।\nप्रश्न 2. आउटपुट = 30"
    cfg = _cfg()
    if cfg.get("placeholder"):
        raise BodhanError("Bodhan is switched off in config.yaml (bodhan: placeholder: true).")
    o = cfg.get("ocr") or {}
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {"model": o.get("model", "indic-ocr"),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            "table_format": o.get("table_format", "markdown"),
            "max_tokens": int(o.get("max_tokens", 4096))}  # per layout block, not per page
    data = _post(cfg, o.get("path", "/v1/chat/completions"), body)
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise BodhanError(f"Bodhan OCR reply had no text: {str(data)[:300]}") from e
    return text.strip() or "[BLANK PAGE]"


def translate(text: str, source: str, target: str, profile: str | None = None) -> str:
    """Translate feedback with Bodhan. The API needs only the target language.

    Args:
        text (str): Text to translate.
        source (str): Source language name; text is returned unchanged if it equals target.
        target (str): Target language name.
        profile (str | None): Config profile; None uses the active one.

    Returns:
        str: Translated text.

    Raises:
        BodhanError: If Bodhan is switched off, the call fails, or the reply is cut off or empty.
    """
    if not text.strip() or source == target:
        return text
    if _is_mock(profile):
        return f"(mock Bodhan translation to {target}) {text}"
    cfg = _cfg()
    if cfg.get("placeholder"):
        raise BodhanError("Bodhan is switched off in config.yaml (bodhan: placeholder: true).")
    t = cfg.get("translate") or {}
    data = _post(cfg, t.get("path", "/translate"), {"text": text, "target_language": code_for(target)})
    if data.get("truncated"):
        raise BodhanError("Bodhan translation was cut off (truncated: true)")
    try:
        return str(data["translation"])
    except KeyError as e:
        raise BodhanError(f"Bodhan translation reply had no 'translation': {str(data)[:300]}") from e
