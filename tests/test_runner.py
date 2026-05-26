"""End-to-end runner.run() test with mocked LLM + hallucination + RAGAS.

Verifies the production contract: provenance fields are recorded, the
result JSON has the right shape, gates are computed against the merged
metrics, and the output file is written.
"""
from __future__ import annotations

import importlib
import json
import sys

import pytest


@pytest.fixture()
def fresh_modules(tmp_path, monkeypatch):
    """Run inside an isolated cwd so the file-backed registry doesn't pollute."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    for mod in ("prompt_versioning.registry", "harness.runner"):
        sys.modules.pop(mod, None)
    return (
        importlib.import_module("prompt_versioning.registry"),
        importlib.import_module("harness.runner"),
    )


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_run_records_full_provenance_and_writes_output(tmp_path, fresh_modules):
    registry, runner = fresh_modules

    # 1. Register a prompt
    registry.register(name="moderation", file=_write(tmp_path, "p.txt", "Be strict."), author="r")

    # 2. Build a tiny eval set
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"question": "Q1", "contexts": ["C1"], "ground_truth": "T1"}) + "\n"
        + json.dumps({"question": "Q2", "contexts": ["C2"], "ground_truth": "T2"}) + "\n",
        encoding="utf-8",
    )

    # 3. Mocks — no network, deterministic
    def fake_generate(prompt, item):
        return f"answer-for-{item['question']}"

    def fake_hallucinate(ans, ctx):
        return {"hallucination_rate": 0.0}

    def fake_ragas(records):
        # mimic ragas output keys
        return {
            "faithfulness": 0.95,
            "answer_relevancy": 0.90,
            "context_precision": 0.85,
            "context_recall": 0.80,
        }

    out_path = tmp_path / "result.json"
    result = runner.run(
        prompt_name="moderation",
        eval_path=str(eval_path),
        out_path=str(out_path),
        version=None,
        generate_fn=fake_generate,
        hallucinate_fn=fake_hallucinate,
        ragas_fn=fake_ragas,
    )

    # Provenance contract
    assert result["prompt"] == "moderation"
    assert result["version"] == 1                  # resolved
    assert result["version_requested"] is None
    assert result["prompt_hash"] and len(result["prompt_hash"]) == 12
    assert result["n_items"] == 2

    # Metrics merged correctly
    assert result["metrics"]["faithfulness"] == 0.95
    assert result["metrics"]["hallucination_rate"] == 0.0

    # All gates pass on this synthetic input
    assert result["gates_passed"] is True
    assert result["gate_failures"] == []

    # Samples carry through
    assert [s["answer"] for s in result["samples"]] == ["answer-for-Q1", "answer-for-Q2"]

    # Output file written and parseable
    on_disk = json.loads(out_path.read_text())
    assert on_disk["version"] == 1
    assert on_disk["prompt_hash"] == result["prompt_hash"]


def test_run_fails_gates_when_metrics_low(tmp_path, fresh_modules):
    registry, runner = fresh_modules
    registry.register(name="p", file=_write(tmp_path, "p.txt", "x"), author="r")

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"question": "Q", "contexts": ["C"]}) + "\n", encoding="utf-8")

    result = runner.run(
        prompt_name="p",
        eval_path=str(eval_path),
        out_path=str(tmp_path / "out.json"),
        generate_fn=lambda *_: "A",
        hallucinate_fn=lambda *_: {"hallucination_rate": 0.5},  # too high
        ragas_fn=lambda *_: {"faithfulness": 0.3},               # too low
    )
    assert result["gates_passed"] is False
    assert any("faithfulness" in f for f in result["gate_failures"])
    assert any("hallucination_rate" in f for f in result["gate_failures"])
