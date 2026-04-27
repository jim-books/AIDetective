# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a course assignment (HKUST AI for Design, Assignment 5, due Apr 27) to build a Python detective agent system that solves a murder case using LLMs. The repository currently contains only specifications and evidence data — **no implementation exists yet**.

The case starting point:
> A murder occurred. Two witnesses: (1) lives at the last house on "Northwestern Dr", (2) named Annabel, lives on "Franklin Ave". Follow the chain: witness interviews → suspect descriptors → structured record lookups → murderer → mastermind.

## Evidence Dataset (`/Evidence/`)

All seven JSON files are loaded at startup. **`person.id` is the linkage hub** connecting nearly all tables.

| File | Rows | Key fields |
|------|------|------------|
| `person.json` | 10,011 | `id`, `name`, `ssn`, `address_street`, `city`, `license_id` |
| `drivers_license.json` | 10,007 | `id` (= `person.license_id`), `plate_number`, `hair_color`, `car_make`, `gender`, `height` |
| `income.json` | 7,514 | `ssn`, `annual_income` |
| `get_fit_now_member.json` | 184 | `id` (membership), `person_id`, `membership_status`, `membership_start_date` |
| `get_fit_now_check_in.json` | 2,703 | `membership_id`, `check_in_date`, `check_in_time`, `check_out_time` |
| `facebook_event_checkin.json` | 20,011 | `person_id`, `event_id`, `event_name`, `date` |
| `interview.json` | 4,991 | `person_id`, `transcript` (3,738 non-empty) |

Linkage: `person.license_id → drivers_license.id`, `person.ssn → income.ssn`, `get_fit_now_member.id → get_fit_now_check_in.membership_id`. Date fields are integers in `YYYYMMDD` format. Use left joins — not every person has a license or income record.

## System Architecture to Implement

### Required Tools (R4) — exposed to the LLM via function-calling schemas
- `lookup_person(name=None, address_street=None, city=None)` → rows from `person.json`
- `get_interview(person_id)` → transcript from `interview.json`
- `search_gym_members(membership_id_prefix=None, status=None)` → member + check-in rows
- `search_drivers_license(plate_contains=None, hair_color=None, car_make=None, gender=None, height_range=None)`
- `search_event_attendance(person_id, date=None)`
- `lookup_income(ssn)`

### RAG (R5)
Chunk and embed non-empty `interview.json` transcripts using `text-embedding-3-small`. Store in an in-memory vector store (or MongoDB). Retrieve top-k chunks per investigative question; LLM answers only from retrieved context.

### CoT Loop (R2)
Each reasoning turn outputs a JSON block: `{current_clue, assumptions, planned_action, tool_or_rag_query, result_interpretation, next_step_or_conclusion}`. The outer Python loop parses this to decide whether to continue or finalize.

### Evaluation / Groundedness (R6)
After each accusation, a separate evaluator LLM call checks: `{supported: bool, unsupported_claims: [...], missing_checks: [...]}`. If `supported` is false, re-investigate. Before any final answer, verify both murderer and mastermind with at least two independent evidence links each.

### Prompt Chaining (R3)
Pipeline stages: scene report → extract witness criteria → tool call → witness person records → RAG/tool retrieve interviews → extract suspect descriptors → tool queries on licenses/gym/events → narrow suspects → suspect interview → extract mastermind lead → repeat.

## Grading Rubric (10 pts + 4 optional)

| Item | Points |
|------|--------|
| Prompt LLMs appropriately (R1) | 1 |
| Construct Chain-of-Thought (R2, R3) | 2 |
| Analyze LLM outputs / evaluator (R6, R7) | 2 |
| Tool / function calling (R4) | 2 |
| RAG over interviews (R5) | 2 |
| Find correct murderer + mastermind | 1 |
| Design multi-agent system (optional) | 2 |
| Implement multi-agent system (optional) | 2 |

## Deliverables

1. Python code (RAG, tool use, CoT)
2. Running history PDF showing murderer and mastermind names
3. Report PDF explaining how each rubric item was addressed

## Key Constraints

- **No hallucination:** Tool functions must return verbatim JSON rows. System prompt must instruct the LLM: "If a tool returns no matching records, state that explicitly. Do not invent records."
- **Grounded citations:** Every factual claim in an accusation must cite a specific tool result or RAG chunk.
- **Logging (R7):** Persist full conversation history (system prompts, user messages, assistant reasoning, tool calls/results) to an exportable log for the PDF deliverable.
