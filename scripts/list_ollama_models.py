"""List the models available to the Ollama API key in .env (for the ollama_api profile).

Names that look like vision models are marked.

Usage:
    python scripts/list_ollama_models.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
key = os.getenv("OLLAMA_API_KEY")
if not key:
    sys.exit("OLLAMA_API_KEY is empty in .env (create one at ollama.com/settings/keys)")
r = httpx.get("https://ollama.com/api/tags", headers={"Authorization": f"Bearer {key}"}, timeout=30)
r.raise_for_status()
names = sorted(m["name"] for m in r.json().get("models", []))
print("Models (put the exact names in config.yaml, profile ollama_api):")
for n in names:
    hint = "   <- vision (can read images)" if any(t in n for t in ("vl", "vision", "gemma3", "gemma4", "llava")) else ""
    print("  " + n + hint)
