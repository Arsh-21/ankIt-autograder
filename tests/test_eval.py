"""Evaluation helpers: the --sample pick in eval/run_ablation.py must be the
same for the same size and seed, so every ablation grades the same students.

Usage:
    python -m pytest -q tests/test_eval.py
"""

import pytest

from eval.run_ablation import sample_students

STUDENTS = [(f"Student_{i}", [f"sheets/Student_{i}.pdf"]) for i in range(1, 21)]


def test_sample_is_deterministic_and_keeps_order():
    first = sample_students(STUDENTS, 5, seed=0)
    assert first == sample_students(STUDENTS, 5, seed=0)
    assert first == sample_students(list(reversed(STUDENTS)), 5, seed=0)[::-1]  # input order does not change the pick
    assert len(first) == 5 and len({n for n, _ in first}) == 5
    assert first == [s for s in STUDENTS if s in first]                           # original order kept
    assert {n for n, _ in sample_students(STUDENTS, 5, seed=1)} != {n for n, _ in first}


def test_sample_larger_than_dataset_fails():
    assert sample_students(STUDENTS, 20, seed=3) == STUDENTS
    with pytest.raises(SystemExit, match="more than the 20"):
        sample_students(STUDENTS, 21, seed=0)
