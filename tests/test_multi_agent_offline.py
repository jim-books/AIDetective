"""Offline tests for the M3 multi-agent system. No network calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from detective.agent import Accusation, AgentState
from detective.data import load_evidence
from detective.multi_agent import (
    MANAGER_TOOL_SCHEMAS,
    CriticAgent,
    ManagerAgent,
    RecordsAgent,
    TranscriptAgent,
)


# ---------- helpers (local FakeClient accepts optional tools/response_format) -


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Replays scripted responses. Accepts optional tools/tool_choice/response_format."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, *, model, messages, tools=None, tool_choice=None, response_format=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._scripted:
            raise AssertionError("FakeClient ran out of scripted responses")
        return _completion(self._scripted.pop(0))


def _cot(nxt):
    return json.dumps({
        "current_clue": "scripted",
        "assumptions": [],
        "planned_action": "scripted",
        "tool_or_rag_query": "scripted",
        "result_interpretation": None,
        "next_step_or_conclusion": nxt,
    })


# ---------- tests 1–3: schema structure (no LLM needed) ----------------------


def test_records_agent_tool_restriction():
    tool_names = {s["function"]["name"] for s in RecordsAgent.TOOL_SCHEMAS}
    assert "get_interview" not in tool_names
    assert "rag_search" not in tool_names
    assert tool_names == {
        "lookup_person",
        "search_gym_members",
        "search_drivers_license",
        "search_event_attendance",
        "lookup_income",
    }


def test_transcript_agent_tool_restriction():
    tool_names = {s["function"]["name"] for s in TranscriptAgent.TOOL_SCHEMAS}
    assert tool_names == {"get_interview", "rag_search"}
    assert "lookup_person" not in tool_names
    assert "search_gym_members" not in tool_names


def test_manager_tool_schema_names():
    names = {s["function"]["name"] for s in MANAGER_TOOL_SCHEMAS}
    assert names == {"delegate_to_records", "delegate_to_transcripts", "validate_accusation"}


# ---------- test 4: specialist evidence_log format ---------------------------


def test_specialist_result_evidence_log_merged():
    ev = load_evidence()

    specialist_tool_turn = _msg(
        content=_cot({"type": "continue"}),
        tool_calls=[_tc("s1", "lookup_person", {"name": "Morty Schapiro"})],
    )
    specialist_done_turn = _msg(
        content=_cot({"type": "done", "summary": "Found Morty Schapiro (id=14887)."}),
        tool_calls=None,
    )
    client = FakeClient([specialist_tool_turn, specialist_done_turn])
    agent = RecordsAgent(client=client, ev=ev, model="gpt-4o-mini")
    result = agent.run("Find Morty Schapiro")

    assert len(result.evidence_log) == 1
    entry = result.evidence_log[0]
    assert entry["tool"] == "lookup_person"
    assert "tool_call_id" in entry
    assert entry["result"]["count"] >= 1
    assert "Morty" in result.findings


# ---------- test 5: critic propagates unsupported verdict --------------------


def test_critic_rejects_unsupported_accusation():
    reject_payload = json.dumps({
        "supported": False,
        "unsupported_claims": ["Name not found in evidence log"],
        "missing_checks": ["Confirm person_id via lookup_person"],
        "rationale": "No tool result confirms this person.",
    })
    client = FakeClient([SimpleNamespace(content=reject_payload)])
    acc = Accusation(
        role="murderer",
        person_id=99999,
        name="Made Up Person",
        evidence=[
            {"source": "tool:lookup_person", "claim": "fake claim 1"},
            {"source": "tool:search_gym_members", "claim": "fake claim 2"},
        ],
        raw_cot={},
    )
    critic = CriticAgent(client=client, model="gpt-4o")
    verdict = critic.validate(acc, AgentState())

    assert verdict.supported is False
    assert len(verdict.missing_checks) > 0


# ---------- test 6: full Manager delegation round-trip -----------------------


def test_manager_delegation_round_trip():
    ev = load_evidence()

    # ── Queue (consumed in strict sequential order) ──────────────────────────
    # Manager turn 1: delegate_to_records
    manager_t1 = _msg(
        content=_cot({"type": "continue"}),
        tool_calls=[_tc("m1", "delegate_to_records", {
            "task": "Find the last house on Northwestern Dr and Annabel on Franklin Ave",
        })],
    )
    # RecordsAgent turn 1: tool call
    spec_t1 = _msg(
        content=_cot({"type": "continue"}),
        tool_calls=[_tc("s1", "lookup_person", {
            "address_street_name": "Northwestern Dr",
            "last_house_on_street": True,
        })],
    )
    # RecordsAgent turn 2: done
    spec_t2 = _msg(
        content=_cot({"type": "done", "summary": "Witness 1 is Morty Schapiro (id=14887)."}),
        tool_calls=None,
    )
    # Manager turn 2: validate murderer
    manager_t2 = _msg(
        content=_cot({"type": "continue"}),
        tool_calls=[_tc("m2", "validate_accusation", {
            "role": "murderer",
            "person_id": 14887,
            "name": "Morty Schapiro",
            "evidence": [
                {"source": "tool:lookup_person", "claim": "last house Northwestern Dr id=14887"},
                {"source": "tool:search_gym_members", "claim": "gold gym member"},
            ],
        })],
    )
    # Critic call 1: supported=True (no tools arg → response_format path)
    critic_r1 = SimpleNamespace(content=json.dumps({
        "supported": True,
        "unsupported_claims": [],
        "missing_checks": [],
        "rationale": "Evidence confirmed.",
    }))
    # Manager turn 3: validate mastermind
    manager_t3 = _msg(
        content=_cot({"type": "continue"}),
        tool_calls=[_tc("m3", "validate_accusation", {
            "role": "mastermind",
            "person_id": 16371,
            "name": "Annabel Miller",
            "evidence": [
                {"source": "tool:lookup_person", "claim": "Annabel on Franklin Ave id=16371"},
                {"source": "tool:get_interview", "claim": "transcript implicates her"},
            ],
        })],
    )
    # Critic call 2: supported=True
    critic_r2 = SimpleNamespace(content=json.dumps({
        "supported": True,
        "unsupported_claims": [],
        "missing_checks": [],
        "rationale": "Evidence confirmed.",
    }))

    client = FakeClient([
        manager_t1,   # Manager turn 1
        spec_t1,      # RecordsAgent turn 1
        spec_t2,      # RecordsAgent turn 2
        manager_t2,   # Manager turn 2
        critic_r1,    # CriticAgent call 1
        manager_t3,   # Manager turn 3
        critic_r2,    # CriticAgent call 2
    ])

    manager = ManagerAgent(
        client=client, ev=ev,
        vector_index=None, embed_fn=None,
        model="gpt-4o-mini", eval_model="gpt-4o",
    )
    state = manager.run()

    # Both roles accused
    assert {a.role for a in state.accusations} == {"murderer", "mastermind"}

    # Evidence log contains the specialist's lookup_person result
    assert any(e["tool"] == "lookup_person" for e in state.evidence_log)

    # Manager history contains tool-result turns for delegation
    delegation_turns = [
        t for t in state.history
        if t.role == "tool" and t.name == "delegate_to_records"
    ]
    assert len(delegation_turns) == 1
    assert "Morty Schapiro" in (delegation_turns[0].content or "")

    # Validate-accusation turns are present in history
    validation_turns = [
        t for t in state.history
        if t.role == "tool" and t.name == "validate_accusation"
    ]
    assert len(validation_turns) == 2
