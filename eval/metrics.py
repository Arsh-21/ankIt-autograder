"""Agreement metrics between predicted and human marks.

Each item is a dict with pred, human, max and flagged (needs_review).
  MAE        mean |pred - human|                 (lower is better)
  bias       mean (pred - human); + = over-grading (the main failure reported in the literature)
  RMSE       root mean squared error
  QWK        quadratic weighted kappa on half-mark steps (1 = perfect)
  within_1   share of answers within 1 mark of the human
  nMAE       MAE as a share of max marks (comparable across questions)
  review     share of answers flagged for a human
  flag_recall     share of BIG errors that were flagged (big = >= max(1, 25% of max))
  flag_precision  share of flagged answers that really had a big error

Usage:
    from eval.metrics import compute, fmt
    m = compute([{"pred": 2, "human": 1.5, "max": 2, "flagged": False}, ...])
    print(fmt(m["MAE"]), fmt(m["QWK"]))
"""

from collections.abc import Sequence

import numpy as np

from core.config import thresholds


def qwk(a: Sequence[float], b: Sequence[float]) -> float:
    """Quadratic weighted kappa between two lists of marks, on half-mark steps.

    Args:
        a (Sequence[float]): First list of marks (e.g. predicted).
        b (Sequence[float]): Second list of marks (e.g. human), same length as a.

    Returns:
        float: Kappa (1 = perfect agreement), or nan when all marks are equal.

    Raises:
        ValueError: If a and b differ in length.
    """
    a = [int(round(2 * x)) for x in a]
    b = [int(round(2 * x)) for x in b]
    lo, hi = min(a + b), max(a + b)
    n = hi - lo + 1
    if n < 2:
        return float("nan")
    obs = np.zeros((n, n))
    for x, y in zip(a, b, strict=True):
        obs[x - lo, y - lo] += 1
    w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    exp = np.outer(obs.sum(1), obs.sum(0)) / max(obs.sum(), 1)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else float("nan")


def compute(items: list[dict]) -> dict:
    """Compute all agreement metrics for a list of graded answers.

    Args:
        items (list[dict]): Answers with keys "pred", "human", "max" and optional "flagged".

    Returns:
        dict: "n", "MAE", "bias", "RMSE", "nMAE", "QWK", "within_1", "review",
            "big_errors", "flag_recall", "flag_precision", and "pearson" when it is
            defined. Only {"n": 0} for an empty list.
    """
    if not items:
        return {"n": 0}
    big_frac = thresholds()["big_error_fraction"]
    p = np.array([i["pred"] for i in items], dtype=float)
    h = np.array([i["human"] for i in items], dtype=float)
    m = np.array([i["max"] for i in items], dtype=float)
    flagged = np.array([bool(i.get("flagged")) for i in items])
    err = p - h
    big = np.abs(err) >= np.maximum(1.0, big_frac * m)
    out = {
        "n": len(items),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "nMAE": float(np.mean(np.abs(err) / np.where(m > 0, m, 1))),
        "QWK": qwk(p, h),
        "within_1": float(np.mean(np.abs(err) <= 1)),
        "review": float(np.mean(flagged)),
        "big_errors": int(big.sum()),
        "flag_recall": float(flagged[big].mean()) if big.any() else float("nan"),
        "flag_precision": float(big[flagged].mean()) if flagged.any() else float("nan"),
    }
    if len(items) > 2 and np.std(p) > 0 and np.std(h) > 0:
        out["pearson"] = float(np.corrcoef(p, h)[0, 1])
    return out


def fmt(x: object) -> str:
    """Format a metric value for a table: floats to 3 decimals, nan as "nan".

    Args:
        x (object): The value to format.

    Returns:
        str: The formatted value.
    """
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.3f}"
    return str(x)
