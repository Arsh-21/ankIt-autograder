"""List OpenRouter models with their prices, or the API key's credit usage.

The ids printed here go into the openrouter profile in config.yaml.

Usage:
    python scripts/list_openrouter_models.py                 # vision models only
    python scripts/list_openrouter_models.py --all --search gemini
    python scripts/list_openrouter_models.py --credits       # credit used by the key in .env
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import ROOT  # noqa: E402,F401  (loads .env)


def get(url: str, key: str | None = None) -> dict:
    """GET a JSON endpoint of the OpenRouter API.

    Args:
        url (str): Endpoint URL.
        key (str | None): API key sent as a bearer token; None sends no auth header.

    Returns:
        dict: The parsed JSON response.

    Raises:
        urllib.error.URLError: If the request fails.
    """
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> None:
    """Print the key's credit info, or a table of models sorted by price."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include text-only models")
    ap.add_argument("--search", default="", help="filter by text in the id")
    ap.add_argument("--credits", action="store_true")
    args = ap.parse_args()
    key = os.getenv("OPENROUTER_API_KEY")

    if args.credits:
        if not key:
            sys.exit("Put OPENROUTER_API_KEY in .env first.")
        info = get("https://openrouter.ai/api/v1/key", key).get("data", {})
        print(json.dumps(info, indent=2))
        return

    models = get("https://openrouter.ai/api/v1/models")["data"]
    rows = []
    for m in models:
        mods = m.get("architecture", {}).get("input_modalities", [])
        if not args.all and "image" not in mods:
            continue
        if args.search.lower() not in m["id"].lower():
            continue
        p = m.get("pricing", {})
        pin = float(p.get("prompt", 0) or 0) * 1e6
        pout = float(p.get("completion", 0) or 0) * 1e6
        rows.append((m["id"], "image" in mods, pin, pout, m.get("context_length", 0)))
    rows.sort(key=lambda r: (r[2] + r[3], r[0]))
    print(f"{'model id':55} {'vision':6} {'$/M in':>8} {'$/M out':>8} {'context':>9}")
    for r in rows:
        print(f"{r[0]:55} {'yes' if r[1] else 'no':6} {r[2]:8.2f} {r[3]:8.2f} {r[4]:9}")
    print(f"\n{len(rows)} models. Models ending in ':free' cost nothing but are rate-limited.")


if __name__ == "__main__":
    main()
