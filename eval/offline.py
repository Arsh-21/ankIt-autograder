"""Ablations recomputed from a finished run, with no model calls.

Every grading sample's raw reply is saved, so the parts done by plain code
can be recomputed with other settings:
  * number of samples k  (uses the first k of the N saved samples)
  * aggregation          (mean / median / max / min / p25 / p75)
  * evidence check       (reject vs off)
One run of a variant with many samples (e.g. n_samples: 5) is enough to fill
the whole table. Only first-pass samples are used: the critic re-grade and
OCR re-check are left out, so the rows are comparable with each other but not
with the full-pipeline numbers.

Writes offline_<variant>.csv into the run folder.

Usage:
    python eval/offline.py --run eval/runs/20261028_120000 --variant n5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.aggregator import combine  # noqa: E402
from agents.grader import normalize  # noqa: E402
from core.guard import Finding  # noqa: E402
from core.utils import clamp, remove_struck, round_half  # noqa: E402
from eval.metrics import compute, fmt  # noqa: E402

# aggregation name -> (combine() method, strictness)
METHODS = {"mean": ("mean", None), "median": ("median", None), "max": ("max", None), "min": ("min", None),
           "p25": ("strictness", "strict"), "p75": ("strictness", "lenient")}


def main() -> None:
    """Recompute metrics for every evidence policy, k and aggregation, and save the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    lines = [json.loads(x) for x in open(Path(args.run) / f"questions_{args.variant}.jsonl", encoding="utf-8")]
    lines = [x for x in lines if x["result"]["samples"]]  # blank answers have no samples
    if not lines:
        print("No graded samples found.")
        return
    n_max = min(len(x["result"]["samples"]) for x in lines)

    rows = []
    for policy in ["reject", "off"]:
        normalized = []
        for x in lines:
            q, r = x["question"], x["result"]
            findings = [Finding(f["kind"], f["text"], 0, 0) for f in r["injection_findings"]]
            text = remove_struck(r["graded_text"])
            normalized.append([normalize(q, s["raw"], text, findings, policy) for s in r["samples"]])
        for k in range(1, n_max + 1):
            for mname, (method, strict) in METHODS.items():
                items = []
                for x, samples in zip(lines, normalized, strict=True):
                    q = x["question"]
                    total = 0.0
                    for c in q["criteria"]:
                        vals = [s["criteria"][c["id"]]["marks"] for s in samples[:k]]
                        total += clamp(round_half(combine(vals, method, strict or "standard")), 0, c["marks"])
                    items.append({"pred": total * x["scale"], "human": x["human"], "max": q["max_marks"] * x["scale"],
                                  "flagged": False})
                m = compute(items)
                rows.append({"evidence": policy, "k": k, "aggregation": mname, "n": m["n"], "MAE": m["MAE"],
                             "bias": m["bias"], "QWK": m["QWK"], "within_1": m["within_1"]})

    out = Path(args.run) / f"offline_{args.variant}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("| evidence | k | aggregation | n | MAE | bias | QWK | within_1 |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in r) + " |")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
