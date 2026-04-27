"""Multi-agent detective system: Manager + Records + Transcript + Critic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import (
    DEFAULT_MODEL,
    AgentState,
    Accusation,
    TurnRecord,
    _execute_tool,
    _parse_cot,
    _trim_history,
)
from .data import Evidence
from .evaluator import EvaluatorVerdict, evaluate_accusation
from .prompts import (
    INITIAL_USER_MESSAGE,
    MANAGER_SYSTEM,
    RECORDS_SYSTEM,
    TRANSCRIPTS_SYSTEM,
)
from .rag import VectorIndex
from .tools import TOOL_SCHEMAS

# ── Manager delegation tool schemas ──────────────────────────────────────────

MANAGER_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_records",
            "description": (
                "Send a structured-data query to the Records Specialist. "
                "Use for: finding people by name/address, gym member/check-in queries, "
                "license plate or physical attribute searches, event attendance, income lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Plain-English instruction. Be specific: include known names, "
                            "IDs, dates, and filter criteria."
                        ),
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_transcripts",
            "description": (
                "Send an interview retrieval or free-text search task to the Transcript Specialist. "
                "Use for: fetching a specific person's interview by person_id, "
                "or semantic search over all interview transcripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Plain-English instruction. Include person_id if known, "
                            "or a free-text description for RAG search."
                        ),
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_accusation",
            "description": (
                "Submit an accusation to the independent Critic for groundedness evaluation. "
                "Must be called before finalizing any accusation. "
                "Returns supported (bool), missing_checks, unsupported_claims, and rationale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["murderer", "mastermind"],
                        "description": "Which role is being accused.",
                    },
                    "person_id": {
                        "type": "integer",
                        "description": "The person.id of the accused. Must come from a specialist tool result.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the accused as returned by a specialist.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "claim": {"type": "string"},
                            },
                            "required": ["source", "claim"],
                            "additionalProperties": False,
                        },
                        "minItems": 2,
                        "description": "At least two evidence items. source must be tool:<name> or rag:<chunk_id>.",
                    },
                },
                "required": ["role", "person_id", "name", "evidence"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Specialist result ─────────────────────────────────────────────────────────

@dataclass
class SpecialistResult:
    findings: str                     # plain-English summary returned to Manager
    evidence_log: list[dict] = field(default_factory=list)  # merged into Manager state
    raw_history: list[TurnRecord] = field(default_factory=list)  # for logging only


# ── Shared specialist loop ────────────────────────────────────────────────────

def _run_specialist_loop(
    client: Any,
    ev: Evidence,
    vector_index: VectorIndex | None,
    embed_fn: Callable | None,
    system_prompt: str,
    task: str,
    tool_schemas: list[dict],
    max_turns: int = 5,
    model: str = DEFAULT_MODEL,
) -> SpecialistResult:
    state = AgentState()
    state.append(TurnRecord(role="system", content=system_prompt))
    state.append(TurnRecord(role="user", content=task))

    last_content = ""

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model,
            messages=state.messages(),
            tools=tool_schemas,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        content = msg.content
        tool_calls = getattr(msg, "tool_calls", None) or []

        tc_list = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
        state.append(TurnRecord(role="assistant", content=content, tool_calls=tc_list or None))

        if content:
            last_content = content

        cot = _parse_cot(content)

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(
                tc.function.name, args,
                ev=ev, vector_index=vector_index, embed_fn=embed_fn,
            )
            state.evidence_log.append({
                "tool_call_id": tc.id,
                "tool": tc.function.name,
                "args": args,
                "result": result,
            })
            state.append(TurnRecord(
                role="tool",
                tool_call_id=tc.id,
                name=tc.function.name,
                content=result,
            ))

        # Exit on "done"
        if cot and isinstance(cot.get("next_step_or_conclusion"), dict):
            nxt = cot["next_step_or_conclusion"]
            if nxt.get("type") == "done":
                findings = nxt.get("summary", last_content)
                return SpecialistResult(
                    findings=str(findings),
                    evidence_log=state.evidence_log,
                    raw_history=state.history,
                )

        # Fallback exit only after at least one tool call has been made.
        # If no evidence collected yet, nudge the specialist to use a tool.
        if not tool_calls:
            if state.evidence_log:
                return SpecialistResult(
                    findings=last_content,
                    evidence_log=state.evidence_log,
                    raw_history=state.history,
                )
            # No tools called yet — force the specialist to query the database
            state.append(TurnRecord(
                role="user",
                content=(
                    "You MUST call a tool to retrieve data from the database. "
                    "Do not answer from memory or training data. "
                    "Issue the appropriate tool call now."
                ),
            ))

    return SpecialistResult(
        findings=f"[max_turns exceeded] {last_content}",
        evidence_log=state.evidence_log,
        raw_history=state.history,
    )


# ── Specialist agents ─────────────────────────────────────────────────────────

class RecordsAgent:
    """Handles structured database queries using 5 tools (no interview/RAG access)."""

    TOOL_SCHEMAS: list[dict] = [
        s for s in TOOL_SCHEMAS
        if s["function"]["name"] in {
            "lookup_person",
            "search_gym_members",
            "search_drivers_license",
            "search_event_attendance",
            "lookup_income",
        }
    ]

    def __init__(self, *, client: Any, ev: Evidence, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._ev = ev
        self._model = model

    def run(self, task: str) -> SpecialistResult:
        return _run_specialist_loop(
            self._client, self._ev, None, None,
            RECORDS_SYSTEM, task, self.TOOL_SCHEMAS,
            max_turns=8, model=self._model,
        )


class TranscriptAgent:
    """Handles interview retrieval and RAG search using 2 tools."""

    TOOL_SCHEMAS: list[dict] = [
        s for s in TOOL_SCHEMAS if s["function"]["name"] == "get_interview"
    ] + [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search interview transcripts by free-text query. Returns the top-k "
                    "most-similar interview chunks (each chunk = one interviewee's transcript) "
                    "with cosine similarity scores."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def __init__(
        self,
        *,
        client: Any,
        ev: Evidence,
        vector_index: VectorIndex | None,
        embed_fn: Callable | None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._ev = ev
        self._vector_index = vector_index
        self._embed_fn = embed_fn
        self._model = model

    def run(self, task: str) -> SpecialistResult:
        return _run_specialist_loop(
            self._client, self._ev, self._vector_index, self._embed_fn,
            TRANSCRIPTS_SYSTEM, task, self.TOOL_SCHEMAS,
            max_turns=8, model=self._model,
        )


class CriticAgent:
    """Thin wrapper around evaluate_accusation() — no LLM loop of its own."""

    def __init__(self, *, client: Any, model: str = "gpt-4o") -> None:
        self._client = client
        self._model = model

    def validate(self, accusation: Accusation, state: AgentState) -> EvaluatorVerdict:
        return evaluate_accusation(
            client=self._client,
            accusation=accusation,
            state=state,
            model=self._model,
        )


# ── Manager agent ─────────────────────────────────────────────────────────────

class ManagerAgent:
    """Orchestrates the investigation by delegating to specialist agents."""

    MAX_TURNS = 60

    def __init__(
        self,
        *,
        client: Any,
        ev: Evidence,
        vector_index: VectorIndex | None,
        embed_fn: Callable | None,
        model: str = DEFAULT_MODEL,
        eval_model: str = "gpt-4o",
    ) -> None:
        self._client = client
        self._ev = ev
        self._vector_index = vector_index
        self._embed_fn = embed_fn
        self._model = model
        self._eval_model = eval_model

    def run(self) -> AgentState:
        state = AgentState()
        state.append(TurnRecord(role="system", content=MANAGER_SYSTEM))
        state.append(TurnRecord(role="user", content=INITIAL_USER_MESSAGE))

        for _ in range(self.MAX_TURNS):
            if {"murderer", "mastermind"} <= {a.role for a in state.accusations}:
                break

            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=state.messages(),
                    tools=MANAGER_TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as exc:
                if "ContextWindowExceeded" in str(exc) or "context_length_exceeded" in str(exc):
                    state.history = _trim_history(state.history)
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=state.messages(),
                        tools=MANAGER_TOOL_SCHEMAS,
                        tool_choice="auto",
                    )
                else:
                    raise

            msg = resp.choices[0].message
            content = msg.content
            tool_calls = getattr(msg, "tool_calls", None) or []

            tc_list = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
            state.append(TurnRecord(role="assistant", content=content, tool_calls=tc_list or None))

            cot = _parse_cot(content)

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                name = tc.function.name

                if name == "delegate_to_records":
                    specialist = RecordsAgent(
                        client=self._client, ev=self._ev, model=self._model,
                    )
                    result = specialist.run(args.get("task", ""))
                    state.evidence_log.extend(result.evidence_log)
                    findings_str = result.findings

                elif name == "delegate_to_transcripts":
                    specialist = TranscriptAgent(
                        client=self._client, ev=self._ev,
                        vector_index=self._vector_index, embed_fn=self._embed_fn,
                        model=self._model,
                    )
                    result = specialist.run(args.get("task", ""))
                    state.evidence_log.extend(result.evidence_log)
                    findings_str = result.findings

                elif name == "validate_accusation":
                    acc = Accusation(
                        role=args.get("role", "unknown"),
                        person_id=int(args.get("person_id", -1)),
                        name=str(args.get("name", "")),
                        evidence=list(args.get("evidence", [])),
                        raw_cot=cot or {},
                    )
                    verdict = CriticAgent(
                        client=self._client, model=self._eval_model,
                    ).validate(acc, state)
                    findings_str = json.dumps({
                        "supported": verdict.supported,
                        "unsupported_claims": verdict.unsupported_claims,
                        "missing_checks": verdict.missing_checks,
                        "rationale": verdict.rationale,
                    })
                    if verdict.supported:
                        state.accusations.append(acc)

                else:
                    findings_str = json.dumps({"error": f"unknown delegation: {name}"})

                state.append(TurnRecord(
                    role="tool",
                    tool_call_id=tc.id,
                    name=name,
                    content=findings_str,
                ))

            if not tool_calls:
                state.append(TurnRecord(
                    role="user",
                    content="Continue. Delegate to a specialist or validate an accusation.",
                ))

        return state
