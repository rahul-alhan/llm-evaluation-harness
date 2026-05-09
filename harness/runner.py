"""Run a registered prompt against an eval set and emit a result file."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from evaluators.gates import evaluate_gates
from evaluators.hallucination import hallucination_rate
from evaluators.ragas_eval import run_ragas
from prompt_versioning.registry import load as load_prompt


def _llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _generate(prompt_template: str, item: dict) -> str:
    tpl = ChatPromptTemplate.from_messages(
        [("system", prompt_template), ("user", "Question: {question}\n\nContext:\n{context}")]
    )
    chain = tpl | _llm()
    ctx = "\n\n".join(item.get("contexts", []))
    return chain.invoke({"question": item["question"], "context": ctx}).content


def run(prompt_name: str, eval_path: str, out_path: str, version: int | None = None) -> dict:
    prompt = load_prompt(prompt_name, version)
    items = _read_jsonl(eval_path)

    enriched = []
    halluc_rates = []
    for item in items:
        ans = _generate(prompt, item)
        enriched.append(
            {
                "question": item["question"],
                "answer": ans,
                "contexts": item.get("contexts", []),
                "ground_truth": item.get("ground_truth", ""),
            }
        )
        h = hallucination_rate(ans, item.get("contexts", []))
        halluc_rates.append(h["hallucination_rate"])

    ragas_scores = run_ragas(enriched)
    metrics = {
        **ragas_scores,
        "hallucination_rate": sum(halluc_rates) / len(halluc_rates) if halluc_rates else 0.0,
    }
    passed, failures = evaluate_gates(metrics)

    result = {
        "prompt": prompt_name,
        "version": version,
        "n_items": len(items),
        "metrics": metrics,
        "gates_passed": passed,
        "gate_failures": failures,
        "samples": enriched,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    return result
