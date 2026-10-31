"""Deploy the app to a Hugging Face Space, which gives it a public URL.

Creates the Space if needed, uploads the code (no .env, no outputs), writes
the Space's README header, and stores the API keys from the local .env as
Space secrets, so they are never part of the uploaded files. The Space runs
with AUTOGRADER_PUBLIC=1 (page and class-size limits, no code execution) and
the chosen profile (default: openrouter).

Needs a Hugging Face access token with "write" permission
(huggingface.co: Settings -> Access Tokens).

Usage:
    python scripts/deploy_space.py --space <hf-username>/autograder --token hf_...
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import ROOT  # noqa: E402  (also loads .env)

IGNORE = {".env", ".git", ".venv", "venv", "outputs", ".cache", "__pycache__", "eval/runs", ".github",
          ".pytest_cache", ".ruff_cache"}

SPACE_HEADER = """---
title: Ankit
emoji: 📝
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: Grades handwritten answer sheets with agents
---

"""


def _copy_tree(dst: Path) -> None:
    """Copy the project into dst, leaving out secrets, caches, outputs and dot-files.

    Args:
        dst (Path): Destination folder (must exist).
    """
    for item in ROOT.iterdir():
        rel = item.name
        if rel in IGNORE or rel.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / rel, ignore=shutil.ignore_patterns("__pycache__", "runs", "*.pyc"))
        else:
            shutil.copy2(item, dst / rel)


def main() -> None:
    """Create or update the Space, upload the code and set the secrets."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--profile", default="openrouter")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if not args.token:
        sys.exit("Give --token or set HF_TOKEN.")

    import gradio
    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _copy_tree(tmp)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (tmp / "README.md").write_text(SPACE_HEADER.format(gradio_version=gradio.__version__) + readme,
                                       encoding="utf-8")
        api.upload_folder(folder_path=str(tmp), repo_id=args.space, repo_type="space",
                          commit_message="Deploy Ankit")

    secrets = {"AUTOGRADER_PROFILE": args.profile, "AUTOGRADER_PUBLIC": "1"}
    for key in ["OPENROUTER_API_KEY", "BODHAN_API_KEY", "AUTOGRADER_USER", "AUTOGRADER_PASSWORD"]:
        if os.getenv(key):
            secrets[key] = os.getenv(key)
    for k, v in secrets.items():
        api.add_space_secret(args.space, k, v)
    print(f"Deployed. Secrets set: {sorted(secrets)}")
    print(f"Open https://huggingface.co/spaces/{args.space} (first build takes a few minutes).")


if __name__ == "__main__":
    main()
