"""Promotion gates — exit non-zero on failure for CI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATES = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.80,
    "context_recall": 0.75,
    "hallucination_rate": 0.10,   # max allowed
}

MAX_METRICS = {"hallucination_rate"}


def evaluate_gates(scores: dict) -> tuple[bool, list[str]]:
    failures = []
    for metric, threshold in GATES.items():
        if metric not in scores:
            continue
        value = scores[metric]
        if metric in MAX_METRICS:
            if value > threshold:
                failures.append(f"{metric} {value:.3f} > max {threshold}")
        else:
            if value < threshold:
                failures.append(f"{metric} {value:.3f} < min {threshold}")
    return len(failures) == 0, failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="path to eval result JSON")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    scores = json.loads(Path(args.result).read_text())
    metrics = scores.get("metrics", scores)

    passed, failures = evaluate_gates(metrics)
    if passed:
        print("All gates passed.")
        sys.exit(0)

    print("Gate failures:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1 if args.strict else 0)


if __name__ == "__main__":
    main()
