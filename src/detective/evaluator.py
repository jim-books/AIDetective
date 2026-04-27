"""Groundedness evaluator: independent LLM pass over each accusation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agent import Accusation, AgentState
from .prompts import EVALUATOR_SYSTEM, evaluator_user_prompt

DEFAULT_EVALUATOR_MODEL = "gpt-4o-mini"


@dataclass
class EvaluatorVerdict:
    supported: bool
    unsupported_claims: list[str]
    missing_checks: list[str]
    rationale: str
    raw: dict


def evaluate_accusation(
    *,
    client: Any,
    accusation: Accusation,
    state: AgentState,
    model: str = DEFAULT_EVALUATOR_MODEL,
) -> EvaluatorVerdict:
    """Call the evaluator model with the accusation + the agent's evidence log."""
    payload = {
        "role": accusation.role,
        "person_id": accusation.person_id,
        "name": accusation.name,
        "evidence": accusation.evidence,
    }
    user_msg = evaluator_user_prompt(payload, state.evidence_log)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "supported": False,
            "unsupported_claims": ["evaluator returned non-JSON"],
            "missing_checks": ["retry the evaluator"],
            "rationale": raw,
        }

    # Two-link rule: enforce locally as well as via the prompt
    distinct_sources = {e.get("source") for e in accusation.evidence if isinstance(e, dict)}
    if len(distinct_sources) < 2 and parsed.get("supported"):
        parsed["supported"] = False
        parsed.setdefault("missing_checks", []).append(
            "Need ≥2 evidence items from distinct sources (different tools or RAG chunks)."
        )

    return EvaluatorVerdict(
        supported=bool(parsed.get("supported", False)),
        unsupported_claims=list(parsed.get("unsupported_claims", [])),
        missing_checks=list(parsed.get("missing_checks", [])),
        rationale=str(parsed.get("rationale", "")),
        raw=parsed,
    )
