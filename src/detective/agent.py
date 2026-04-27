"""The detective CoT loop: chat completion → optional tool calls → repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .data import Evidence
from .prompts import (
    DETECTIVE_SYSTEM,
    INITIAL_USER_MESSAGE,
    reinvestigation_prompt,
)
from .rag import VectorIndex, retrieve
from .tools import TOOL_SCHEMAS, dispatch

DEFAULT_MODEL = "gpt-4o-mini"
MAX_TURNS = 40

# Adds a 7th synthetic tool that taps the RAG index. We attach it so the
# model can ask "rag_search(query=...)" via the standard tool-call channel.
RAG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search interview transcripts by free-text query. Returns the top-k "
            "most-similar interview chunks (each chunk = one interviewee's "
            "transcript) with cosine similarity scores."
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

ALL_TOOL_SCHEMAS = TOOL_SCHEMAS + [RAG_TOOL_SCHEMA]


@dataclass
class TurnRecord:
    """Captured per-turn payload (for logging + evaluator)."""
    role: str
    content: Any | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # for tool-result messages

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content if isinstance(self.content, str) else json.dumps(self.content)
        elif self.role == "assistant":
            # Azure litellm requires content to be a string, not absent/null,
            # even when the assistant turn contains only tool_calls.
            msg["content"] = ""
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class Accusation:
    role: str
    person_id: int
    name: str
    evidence: list[dict]
    raw_cot: dict


@dataclass
class AgentState:
    """Holds the running message history + the tool-result audit trail."""
    history: list[TurnRecord] = field(default_factory=list)
    evidence_log: list[dict] = field(default_factory=list)  # for the evaluator
    accusations: list[Accusation] = field(default_factory=list)

    def messages(self) -> list[dict]:
        return [t.to_message() for t in self.history]

    def append(self, turn: TurnRecord) -> None:
        self.history.append(turn)


def _execute_tool(
    name: str,
    args: dict,
    *,
    ev: Evidence,
    vector_index: VectorIndex | None,
    embed_fn: Callable | None,
) -> dict:
    if name == "rag_search":
        if vector_index is None or embed_fn is None:
            return {"error": "rag_search is not available in this session"}
        top_k = int(args.get("top_k", 5))
        return {"results": retrieve(vector_index, args["query"], embed_fn, top_k=top_k)}
    return dispatch(ev, name, args)


def _parse_cot(content: str | None) -> dict | None:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Salvage if the model wrapped JSON in fences
        stripped = content.strip().strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None


def run_investigation(
    *,
    client: Any,
    ev: Evidence,
    vector_index: VectorIndex | None,
    embed_fn: Callable | None,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    state: AgentState | None = None,
    extra_user_messages: list[str] | None = None,
) -> AgentState:
    """Drive the agent loop. Returns the final AgentState (history + accusations).

    The loop terminates when the model emits a CoT JSON whose
    `next_step_or_conclusion.type == "accuse"` AND we have collected
    accusations for both `murderer` and `mastermind`, or when `max_turns`
    is exceeded.
    """
    state = state or AgentState()

    if not state.history:
        state.append(TurnRecord(role="system", content=DETECTIVE_SYSTEM))
        state.append(TurnRecord(role="user", content=INITIAL_USER_MESSAGE))

    if extra_user_messages:
        for m in extra_user_messages:
            state.append(TurnRecord(role="user", content=m))

    needed_roles = {"murderer", "mastermind"}
    found_roles: set[str] = {a.role for a in state.accusations}

    for _ in range(max_turns):
        if found_roles >= needed_roles:
            break

        resp = client.chat.completions.create(
            model=model,
            messages=state.messages(),
            tools=ALL_TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        content = msg.content
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Convert to JSON-serialisable tool_calls list for the history
        tc_list = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]

        state.append(TurnRecord(
            role="assistant",
            content=content,
            tool_calls=tc_list or None,
        ))

        cot = _parse_cot(content)

        # Run any tool calls, append tool results
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

        # If the assistant accused, capture it.
        if cot and isinstance(cot.get("next_step_or_conclusion"), dict):
            nxt = cot["next_step_or_conclusion"]
            if nxt.get("type") == "accuse":
                acc = Accusation(
                    role=nxt.get("role", "unknown"),
                    person_id=int(nxt.get("person_id", -1)),
                    name=str(nxt.get("name", "")),
                    evidence=list(nxt.get("evidence", [])),
                    raw_cot=cot,
                )
                state.accusations.append(acc)
                found_roles.add(acc.role)

        # If the assistant produced neither tool calls nor an accusation, nudge it.
        if not tool_calls and not (cot and cot.get("next_step_or_conclusion", {}).get("type") == "accuse"):
            state.append(TurnRecord(
                role="user",
                content="Continue the investigation. Either issue a tool call or, if you have ≥2 independent evidence links, output an accusation JSON.",
            ))

    return state


def feed_reinvestigation(state: AgentState, missing_checks: list[str], unsupported_claims: list[str]) -> None:
    """Append the evaluator's feedback so the next `run_investigation` call resumes."""
    state.append(TurnRecord(
        role="user",
        content=reinvestigation_prompt(missing_checks, unsupported_claims),
    ))
    # Pop the rejected accusation so the loop re-tries the same role
    if state.accusations:
        rejected = state.accusations.pop()
