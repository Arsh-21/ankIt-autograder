"""Check that the models in a profile respond, before running anything else.

Sends one tiny text prompt to the grader model and one tiny image to the OCR
model. Exits with status 1 if either call fails.

Usage:
    python scripts/check_setup.py --profile local
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from core.config import load_config, model_for, resolve_profile_name  # noqa: E402
from core.llm import LLMError, chat, image_part  # noqa: E402


def main() -> None:
    """Call the text and OCR models of the profile once each and report the replies."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    profile = resolve_profile_name(args.profile)
    load_config()["llm"]["cache"] = False  # force a real call, not a cached reply

    print(f"Profile: {profile}")
    ok = True

    print(f"\n1) Text model {model_for('grader', profile)}")
    t = time.time()
    try:
        reply = chat("grader", [{"role": "user", "content": "Reply with just the word OK."}],
                     profile=profile, max_tokens=1500)
        print(f"   reply: {reply[:80]!r}   ({time.time() - t:.1f} s)")
    except LLMError as e:
        ok = False
        print(f"   FAILED: {e}")

    print(f"\n2) Vision/OCR model {model_for('ocr', profile)}")
    img = Image.new("RGB", (420, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "HELLO 42", fill="black")
    img = img.resize((840, 240))
    t = time.time()
    try:
        reply = chat("ocr", [{"role": "user", "content": [
            {"type": "text", "text": "What text is written in this image? Reply with the text only."},
            image_part(img)]}], profile=profile, max_tokens=1500)
        print(f"   reply: {reply[:80]!r}   ({time.time() - t:.1f} s)  -- expected something like 'HELLO 42'")
    except LLMError as e:
        ok = False
        print(f"   FAILED: {e}")

    print("\nAll good." if ok else "\nFix the errors above, then run this again.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
