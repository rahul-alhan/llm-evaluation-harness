"""RAGAS metric evaluator."""
from __future__ import annotations

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def run_ragas(records: list[dict]) -> dict:
    """records: [{question, answer, contexts: list[str], ground_truth}, ...]"""
    ds = Dataset.from_list(records)
    out = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return out.to_pandas().mean(numeric_only=True).to_dict()
