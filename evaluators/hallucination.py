"""LLM-as-judge claim-extraction hallucination check."""
from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract atomic factual claims from the answer as a JSON list of strings."),
        ("user", "Answer:\n{answer}\n\nReturn only JSON like: [\"claim 1\", \"claim 2\"]."),
    ]
)

_VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checker. Decide if the CLAIM is entailed by the CONTEXT. "
            'Return JSON: {{"supported": true|false, "reason": "..."}}.',
        ),
        ("user", "CONTEXT:\n{context}\n\nCLAIM:\n{claim}"),
    ]
)


def _llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _safe_json(text: str, default):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return default
        return default


def extract_claims(answer: str) -> list[str]:
    chain = _EXTRACT_PROMPT | _llm()
    out = chain.invoke({"answer": answer}).content
    parsed = _safe_json(out, [])
    return [c for c in parsed if isinstance(c, str)]


def verify_claim(claim: str, context: str) -> dict:
    chain = _VERIFY_PROMPT | _llm()
    out = chain.invoke({"claim": claim, "context": context}).content
    return _safe_json(out, {"supported": False, "reason": "parse_error"})


def hallucination_rate(answer: str, contexts: list[str]) -> dict:
    claims = extract_claims(answer)
    if not claims:
        return {"hallucination_rate": 0.0, "n_claims": 0, "details": []}

    ctx_blob = "\n\n".join(contexts)
    details = []
    unsupported = 0
    for c in claims:
        v = verify_claim(c, ctx_blob)
        if not v.get("supported"):
            unsupported += 1
        details.append({"claim": c, **v})

    return {
        "hallucination_rate": unsupported / len(claims),
        "n_claims": len(claims),
        "n_unsupported": unsupported,
        "details": details,
    }
