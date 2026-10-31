"""Shared test setup: every test runs on the fake `mock` model (no network,
no keys), from the project root so relative sample paths work.

Usage:
    python -m pytest -q
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["AUTOGRADER_PROFILE"] = "mock"
os.environ.pop("BODHAN_API_KEY", None)


@pytest.fixture(autouse=True)
def _in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def rubric():
    return {"questions": [
        {"id": "Q1", "text": "Explain vanishing gradients.", "max_marks": 4, "model_answer": "...",
         "criteria": [{"id": "Q1.1", "description": "gradients shrink through layers", "marks": 2, "partial_allowed": True},
                      {"id": "Q1.2", "description": "one remedy", "marks": 2, "partial_allowed": True}]},
        {"id": "Q2", "text": "Output size?", "max_marks": 3, "model_answer": "30",
         "criteria": [{"id": "Q2.1", "description": "final answer 30", "marks": 3, "partial_allowed": False}]},
    ]}
