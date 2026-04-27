# Detective Agent System — Report

## Case Result

| Role | Name | Person ID |
|------|------|-----------|
| **Murderer** | Jeremy Bowers | 67318 |
| **Mastermind** | Miranda Priestly | 99716 |

---

## R1 — Prompt LLMs Appropriately

**File:** `src/detective/prompts.py`

The system prompt (`DETECTIVE_SYSTEM`) instructs the model to:
- Act as an evidence-driven detective with strict no-hallucination and two-link evidence rules
- Output every reasoning turn as a structured JSON object (current clue, assumptions, planned action, tool query, result interpretation, next step or accusation)
- Cite only verbatim tool results or RAG chunks — never invented facts
- Follow a mandated investigation order: identify the murderer before the mastermind

The evaluator prompt (`EVALUATOR_SYSTEM`) uses a separate LLM call with independent instructions to verify accusations against the evidence log.

---

## R2 + R3 — Chain-of-Thought and Prompt Chaining

**File:** `src/detective/agent.py`, `src/detective/prompts.py`

Each assistant turn produces a JSON block with schema:

```json
{
  "current_clue": "...",
  "assumptions": ["..."],
  "planned_action": "...",
  "tool_or_rag_query": "...",
  "result_interpretation": "...",
  "next_step_or_conclusion": {"type": "continue"} | {"type": "accuse", ...}
}
```

The pipeline chains prompts across stages:
1. Scene report → `INITIAL_USER_MESSAGE`
2. Witness lookup → `lookup_person` + `get_interview`
3. Suspect descriptors extracted from witness transcripts → `search_gym_members` + `search_drivers_license`
4. Gym suspect interviews → `get_interview` per suspect (murderer identification)
5. Murderer interview → mastermind physical description extracted
6. License + event attendance search → mastermind identity confirmed

---

## R4 — Tool / Function Calling

**File:** `src/detective/tools.py`, `src/detective/agent.py`

Six tool functions are exposed via OpenAI function-calling schemas (`TOOL_SCHEMAS`):

| Tool | Purpose |
|------|---------|
| `lookup_person` | Find residents by name/street/address; supports `last_house_on_street` |
| `get_interview` | Fetch transcript by `person_id` |
| `search_gym_members` | Join members with check-ins; filter by prefix/status/date |
| `search_drivers_license` | Filter licenses by physical attributes; enriches with person row |
| `search_event_attendance` | Facebook event check-ins by person, event name, date range |
| `lookup_income` | Annual income by SSN or person_id |

A synthetic seventh tool `rag_search` routes free-text queries through the RAG index so the model uses a single function-calling channel for all retrieval.

All tools return verbatim JSON rows or an explicit `{"results": [], "count": 0, "note": "no matching records"}` — never paraphrased data. Results are capped at 100 rows to prevent context overflow.

A single `dispatch(ev, name, args)` helper executes tools by name and returns structured error payloads on unknown tools or bad arguments.

---

## R5 — RAG over Interviews

**File:** `src/detective/rag.py`

- All 4,991 interview transcripts are embedded using `text-embedding-3-small`
- Vectors are L2-normalised and stored in `.cache/embeddings.npz` keyed by a SHA-256 content hash; subsequent runs skip re-embedding
- Retrieval uses cosine similarity via NumPy batch matrix multiplication (`VectorIndex.search`)
- The `rag_search` tool exposes this to the agent: query → top-k chunk results with `chunk_id`, `person_id`, `text`, and `score`

---

## R6 + R7 — Evaluation and Logging

**Files:** `src/detective/evaluator.py`, `src/detective/logger.py`

### Evaluator (R6)

After each accusation the evaluator (`evaluate_accusation`) runs an independent `gpt-4o-mini` call with:
- The accusation (role, person_id, name, evidence array)
- The full evidence log (every tool call + RAG retrieval from the session)

It returns `{supported, unsupported_claims, missing_checks, rationale}`. A local post-check enforces the two-link rule (≥2 distinct sources) regardless of the LLM verdict.

If the evaluator rejects an accusation, a targeted re-investigation runs with a role-specific hint (e.g., for the mastermind: read Bowers' interview → search drivers license by description → verify SQL Symphony attendance in Dec 2017).

### Logger (R7)

`write_jsonl(state)` saves every system/user/assistant/tool message to `runs/<timestamp>.jsonl`. `export_markdown(jsonl_path)` renders the same trace as a readable markdown document (used for this running history PDF). The full conversation trace — all tool calls, results, reasoning JSON, and evaluator feedback — is preserved for review.

---

---

## M3 — Multi-Agent System (Optional)

**Files:** `src/detective/multi_agent.py`, `src/detective/multi_main.py`, `src/detective/prompts.py`

### Design

The multi-agent system replaces the single monolithic agent with four collaborating agents:

| Agent | Role | Tools |
|-------|------|-------|
| **ManagerAgent** | Orchestrator — delegates queries, collects findings, validates accusations | 3 delegation tools |
| **RecordsAgent** | Structured DB specialist | `lookup_person`, `search_gym_members`, `search_drivers_license`, `search_event_attendance`, `lookup_income` |
| **TranscriptAgent** | Interview / RAG specialist | `get_interview`, `rag_search` |
| **CriticAgent** | Groundedness validator | wraps `evaluate_accusation()` |

**Orchestrator-as-caller pattern**: The Manager's LLM sees only three tools — `delegate_to_records(task)`, `delegate_to_transcripts(task)`, and `validate_accusation(role, person_id, name, evidence)`. When it calls a delegation tool, Python synchronously creates the specialist, runs its mini CoT loop (max 5 turns) with its restricted tool subset, and returns the findings string as the tool result. The Manager's `evidence_log` accumulates every specialist tool result so the Critic sees a unified log.

**Accusation integrity**: `state.accusations.append` in `ManagerAgent.run()` happens *only* inside the `validate_accusation` handler when `CriticAgent.validate()` returns `supported=True`. A CoT `"type":"accuse"` output from the Manager's LLM is treated as the call arguments for `validate_accusation`, not as a direct commit — this prevents ungrounded final accusations.

### Implementation

- `MANAGER_TOOL_SCHEMAS` — 3-element list with full JSON Schema definitions for the delegation tools
- `SpecialistResult` — `(findings: str, evidence_log: list[dict], raw_history: list[TurnRecord])` — findings returned to Manager; evidence_log merged into Manager state; raw_history kept for logging but not fed to Manager's context window
- `_run_specialist_loop()` — shared private helper used by both `RecordsAgent.run()` and `TranscriptAgent.run()`; imports `_execute_tool`, `_parse_cot` from `agent.py` to avoid duplication
- `RecordsAgent.TOOL_SCHEMAS` / `TranscriptAgent.TOOL_SCHEMAS` — class-level attributes filtered from the existing `TOOL_SCHEMAS` in `tools.py`; tool restriction verified by tests without any LLM call

Entry point: `python -m detective.multi_main` (also registered as `detective-multi` CLI).

---

## Evidence Chain Summary

### Murderer: Jeremy Bowers (id=67318)

1. Witness Morty Schapiro (last house on Northwestern Dr): "gym bag with membership starting '48Z', gold member, plate includes 'H42W'"
2. Witness Annabel Miller (Franklin Ave): "I recognized the killer from my gym when I was working out on January 9th"
3. `search_gym_members(membership_id_prefix="48Z", status="gold", check_in_date=20180109)` → Jeremy Bowers (membership 48Z55, checked in Jan 9 2018)
4. `search_drivers_license(plate_contains="H42W")` → Jeremy Bowers' plate `0H42W2`

### Mastermind: Miranda Priestly (id=99716)

1. `get_interview(person_id=67318)` → Bowers: "hired by a woman, ~5'5–5'7, red hair, drives Tesla Model S, attended SQL Symphony 3 times in December 2017"
2. `search_drivers_license(hair_color="red", car_make="Tesla", car_model="Model S", gender="female", height_min=65, height_max=67)` → Miranda Priestly (license id=202298, **person.id=99716**)
3. `search_event_attendance(person_id=99716, event_name_contains="SQL Symphony", date_min=20171201, date_max=20171231)` → 3 attendance records in December 2017
