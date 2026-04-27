"""Offline integration test for the CoT loop. The OpenAI client is stubbed so
no network call ever happens — we only validate that the loop dispatches
tools correctly and exits when both accusations have been emitted."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from detective.agent import AgentState, run_investigation
from detective.data import load_evidence


# ---------- helpers ---------------------------------------------------------


def _msg(content=None, tool_calls=None):
    """Build an OpenAI-style chat message namespace."""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Replays a scripted list of `chat.completions.create` responses."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, *, model, messages, tools, tool_choice):
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        if not self._scripted:
            raise AssertionError("FakeClient ran out of scripted responses")
        return _completion(self._scripted.pop(0))


# ---------- the actual test -------------------------------------------------


def _cot(text):
    """Wrap a `next_step_or_conclusion` dict in a full CoT JSON."""
    return json.dumps({
        "current_clue": "scripted",
        "assumptions": [],
        "planned_action": "scripted",
        "tool_or_rag_query": "scripted",
        "result_interpretation": None,
        "next_step_or_conclusion": text,
    })


def test_loop_dispatches_tool_then_accuses_both_roles():
    ev = load_evidence()

    scripted = [
        # Turn 1: model issues lookup_person for the last house on Northwestern Dr.
        _msg(
            content=_cot({"type": "continue"}),
            tool_calls=[_tc("call-1", "lookup_person", {
                "address_street_name": "Northwestern Dr",
                "last_house_on_street": True,
            })],
        ),
        # Turn 2: model accuses the murderer (using person_id from the tool result).
        _msg(
            content=_cot({
                "type": "accuse",
                "role": "murderer",
                "person_id": 14887,
                "name": "Morty Schapiro",
                "evidence": [
                    {"source": "tool:lookup_person", "claim": "lives at 4919 Northwestern Dr"},
                    {"source": "tool:lookup_person", "claim": "id=14887"},
                ],
            }),
            tool_calls=None,
        ),
        # Turn 3: model accuses the mastermind.
        _msg(
            content=_cot({
                "type": "accuse",
                "role": "mastermind",
                "person_id": 16371,
                "name": "Annabel Miller",
                "evidence": [
                    {"source": "tool:lookup_person", "claim": "Annabel on Franklin Ave"},
                    {"source": "tool:get_interview", "claim": "transcript implicates her"},
                ],
            }),
            tool_calls=None,
        ),
    ]

    client = FakeClient(scripted)
    state = run_investigation(
        client=client, ev=ev, vector_index=None, embed_fn=None, max_turns=10,
    )

    # Both accusations captured
    roles = {a.role for a in state.accusations}
    assert roles == {"murderer", "mastermind"}
    assert state.accusations[0].person_id == 14887
    assert state.accusations[1].person_id == 16371

    # Tool was actually executed and recorded in evidence_log
    assert len(state.evidence_log) == 1
    entry = state.evidence_log[0]
    assert entry["tool"] == "lookup_person"
    assert entry["result"]["count"] == 1
    assert entry["result"]["results"][0]["id"] == 14887

    # Conversation has system + user + (assistant + tool) + assistant + assistant
    roles_seq = [t.role for t in state.history]
    assert roles_seq[0] == "system"
    assert roles_seq[1] == "user"
    assert "tool" in roles_seq

    # Loop made exactly 3 client calls (one per scripted turn) — early exit after both accusations.
    assert len(client.calls) == 3


def test_loop_falls_back_to_continue_prompt_when_neither_tool_nor_accuse():
    """Verify the nudge path: when the assistant emits a 'continue' CoT with
    no tool call, the loop appends a user message asking it to proceed."""
    ev = load_evidence()
    scripted = [
        _msg(content=_cot({"type": "continue"}), tool_calls=None),
        _msg(
            content=_cot({
                "type": "accuse", "role": "murderer", "person_id": 14887,
                "name": "Morty Schapiro",
                "evidence": [{"source": "tool:lookup_person", "claim": "x"}, {"source": "tool:lookup_person", "claim": "y"}],
            }),
            tool_calls=None,
        ),
        _msg(
            content=_cot({
                "type": "accuse", "role": "mastermind", "person_id": 16371,
                "name": "Annabel Miller",
                "evidence": [{"source": "tool:lookup_person", "claim": "x"}, {"source": "tool:get_interview", "claim": "y"}],
            }),
            tool_calls=None,
        ),
    ]
    client = FakeClient(scripted)
    state = run_investigation(client=client, ev=ev, vector_index=None, embed_fn=None, max_turns=10)
    # Find the nudge user message after the first assistant turn
    user_after_assistant = [
        t for t in state.history
        if t.role == "user" and "Continue the investigation" in (t.content or "")
    ]
    assert len(user_after_assistant) == 1
    assert {a.role for a in state.accusations} == {"murderer", "mastermind"}
