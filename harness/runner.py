"""Run a registered prompt against an eval set and emit a result file.

`run()` takes optional dependency-injection seams for the LLM call, the
hallucination check, and the RAGAS aggregator. This lets tests exercise
the full output contract — provenance fields, samples shape, gate result —
without network calls. The defaults wire the real production impls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from evaluators.gates import evaluate_gates
from prompt_versioning.registry import load as load_prompt, resolve as resolve_prompt


GenerateFn = Callable[[str, dict], str]
HallucinateFn = Callable[[str, list[str]], dict]
RagasFn = Callable[[list[dict]], dict]


def _default_generate(prompt_template: str, item: dict) -> str:
    # Imports deferred so the module can be imported (and tested with mocks)
    # without langchain / openai installed.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    tpl = ChatPromptTemplate.from_messages(
        [("system", prompt_template), ("user", "Question: {question}\n\nContext:\n{context}")]
    )
    chain = tpl | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ctx = "\n\n".join(item.get("contexts", []))
    return chain.invoke({"question": item["question"], "context": ctx}).content


def _default_hallucinate(answer: str, contexts: list[str]) -> dict:
    from evaluators.hallucination import hallucination_rate
    return hallucination_rate(answer, contexts)


def _default_ragas(records: list[dict]) -> dict:
    from evaluators.ragas_eval import run_ragas
    return run_ragas(records)


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(
    prompt_name: str,
    eval_path: str,
    out_path: str,
    version: int | None = None,
    *,
    generate_fn: GenerateFn | None = None,
    hallucinate_fn: HallucinateFn | None = None,
    ragas_fn: RagasFn | None = None,
) -> dict:
    generate = generate_fn or _default_generate
    hallucinate = hallucinate_fn or _default_hallucinate
    aggregate = ragas_fn or _default_ragas

    entry = resolve_prompt(prompt_name, version)
    resolved_version = entry["version"]
    prompt_hash = entry["hash"]
    prompt = load_prompt(prompt_name, resolved_version)
    items = _read_jsonl(eval_path)

    enriched = []
    halluc_rates = []
    for item in items:
        ans = generate(prompt, item)
        enriched.append(
            {
                "question": item["question"],
                "answer": ans,
                "contexts": item.get("contexts", []),
                "ground_truth": item.get("ground_truth", ""),
            }
        )
        halluc_rates.append(hallucinate(ans, item.get("contexts", []))["hallucination_rate"])

    ragas_scores = aggregate(enriched)
    metrics = {
        **ragas_scores,
        "hallucination_rate": sum(halluc_rates) / len(halluc_rates) if halluc_rates else 0.0,
    }
    passed, failures = evaluate_gates(metrics)

    result = {
        "prompt": prompt_name,
        "version": resolved_version,
        "version_requested": version,
        "prompt_hash": prompt_hash,
        "n_items": len(items),
        "metrics": metrics,
        "gates_passed": passed,
        "gate_failures": failures,
        "samples": enriched,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    return result
