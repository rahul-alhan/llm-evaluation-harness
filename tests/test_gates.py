"""Gate logic smoke tests — no LLM, no API calls."""
from __future__ import annotations

from evaluators.gates import GATES, MAX_METRICS, evaluate_gates


def test_all_passing_metrics_pass_all_gates():
    metrics = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.90,
        "context_precision": 0.85,
        "context_recall": 0.80,
        "hallucination_rate": 0.05,
    }
    passed, failures = evaluate_gates(metrics)
    assert passed is True
    assert failures == []


def test_low_faithfulness_fails_gate():
    metrics = {"faithfulness": 0.50}
    passed, failures = evaluate_gates(metrics)
    assert passed is False
    assert any("faithfulness" in f for f in failures)


def test_max_metric_fails_when_above_threshold():
    """hallucination_rate is a MAX gate — higher is worse."""
    metrics = {"hallucination_rate": 0.30}
    passed, failures = evaluate_gates(metrics)
    assert passed is False
    assert any("hallucination_rate" in f and "> max" in f for f in failures)


def test_missing_metric_is_skipped_not_failed():
    """If a metric isn't in the result, the gate is silently skipped."""
    metrics = {"faithfulness": 0.95}  # missing the others
    passed, failures = evaluate_gates(metrics)
    assert passed is True


def test_hallucination_rate_is_a_max_gate():
    assert "hallucination_rate" in MAX_METRICS


def test_gates_dict_is_non_empty():
    assert GATES and len(GATES) >= 4
