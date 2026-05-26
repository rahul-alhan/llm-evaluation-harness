# LLM Evaluation Harness

Automated evaluation toolkit for LLM and RAG outputs — combines **RAGAS** metrics, custom hallucination checks, prompt versioning with git-style diffs, and a **Streamlit** dashboard for stakeholders.

> Built as a reusable internal tool to enforce LLM quality gates across releases.

---

## What This Solves

Most teams ship LLM features without automated eval — every prompt change is a vibes-based rollout. This harness turns LLM quality into a CI gate.

| Problem | Solution |
|---|---|
| "Is the new prompt better than the old one?" | Side-by-side eval over a fixed set; promotion gated on metric deltas |
| "Are we hallucinating?" | Faithfulness + custom NLI-style claim-extraction check |
| "Which prompt version is in prod?" | File-backed prompt registry with hash, author, timestamp |
| "Stakeholders can't read JSON" | Streamlit dashboard with per-question drilldown |

---

## Quickstart

```bash
# Production install:
pip install -r requirements.txt

# Development install (adds pytest):
pip install -r requirements-dev.txt

export OPENAI_API_KEY=sk-...

# 1. Register a prompt
python -m harness.cli register \
  --name moderation_v1 \
  --file prompts/moderation_v1.txt \
  --author rahul

# 2. Run eval against an eval set
python -m harness.cli evaluate \
  --prompt moderation_v1 \
  --eval-set evals/sample_eval.jsonl \
  --out results/moderation_v1.json

# 3. Launch the dashboard
streamlit run dashboards/streamlit_app.py
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover the prompt registry contract, gate logic, and a mocked runner
flow — no OpenAI API key or network calls required.

---

## Evaluators

### `ragas_eval.py`
Wraps RAGAS metrics with promotion gates:
- **faithfulness** ≥ 0.85
- **answer_relevancy** ≥ 0.80
- **context_precision** ≥ 0.80
- **context_recall** ≥ 0.75

### `hallucination.py`
LLM-as-judge claim-extraction approach:
1. Extract atomic claims from the answer
2. For each claim, check if it is entailed by the context
3. Hallucination rate = unsupported_claims / total_claims

### Prompt Versioning (`prompt_versioning/`)
- File-backed registry (`prompts/registry.json`)
- Each prompt: `name`, `version`, `hash`, `author`, `created_at`, `body_path`
- `diff <name> <vA> <vB>` shows unified diff between versions
- `register` rejects an exact duplicate (same content hash)

---

## Repository Layout

```
llm-evaluation-harness/
├── README.md
├── requirements.txt
├── harness/
│   ├── __init__.py
│   ├── cli.py
│   └── runner.py
├── evaluators/
│   ├── __init__.py
│   ├── ragas_eval.py
│   ├── hallucination.py
│   └── gates.py
├── prompt_versioning/
│   ├── __init__.py
│   └── registry.py
├── prompts/
│   ├── moderation_v1.txt
│   └── registry.json
├── evals/
│   └── sample_eval.jsonl
├── dashboards/
│   └── streamlit_app.py
└── results/                     # eval outputs land here
```

---

## CI Integration

Drop this into a GitHub Action:

```yaml
- name: LLM eval gate
  run: |
    python -m harness.cli evaluate \
      --prompt moderation_v2_candidate \
      --eval-set evals/sample_eval.jsonl \
      --out results/ci.json
    python -m evaluators.gates --result results/ci.json --strict
```

The script exits non-zero if any gate fails — build is blocked.

---

## Why This Matters

> Public eval tooling is rare. Most candidates list "RAGAS" on their resume but have never wired it into a release process. This repo demonstrates I have.

---

## License

MIT
