---
title: "From Crime Scene to Mastermind: Building a Grounded Detective Agent"
slug: "assignment-5-detective-agent-system"
summary: "A PRD for an evidence-grounded LLM detective pipeline that identifies both killer and mastermind."
description: "This PRD defines a Python investigation system that uses prompt chaining, tool calling, RAG, and evaluator checks to solve a two-stage murder case. It maps requirements to rubric items and milestones while enforcing grounded evidence at every step."
---

## Title

**Detective Agent System** — An LLM-powered investigation pipeline that identifies a murderer and the mastermind behind the crime using structured evidence data.

## Purpose & Scope

Build a Python-based detective agent that uses RAG, tool/function calling, Chain-of-Thought prompting, and output evaluation to follow a chain of clues across seven JSON evidence files. The system starts from a crime scene report, locates two witnesses, extracts leads from their interviews, cross-references structured town records, identifies the killer, and then follows a second chain to the mastermind. Scope excludes manual sleuthing — all reasoning must flow through LLM calls with grounded evidence.

## Target User Stories

* **US-1:** As a user, I can provide the initial crime scene description and the system automatically identifies the two witnesses, retrieves their interviews, and extracts actionable clues.
* **US-2:** As a user, I can observe the system querying structured records (gym check-ins, license plates, event attendance, income) via tool calls to corroborate or eliminate suspects.
* **US-3:** As a user, I can review a full reasoning trace (CoT log) that shows every hypothesis, evidence lookup, verification step, and final accusation with cited sources.
* **US-4 (optional):** As a user, I can run the system in fully automatic multi-agent mode where specialized agents collaborate to solve the case end-to-end.

## System Requirements

* **R1 — Prompt Design:** All LLM calls use delimiter-separated context, explicit role/system prompts, and request structured JSON output where downstream parsing is needed.
* **R2 — Chain-of-Thought:** The main investigation loop uses a multi-step CoT prompt: (1) state current clue, (2) list assumptions, (3) specify which tool/RAG query to run, (4) interpret result, (5) conclude or iterate.
* **R3 — Prompt Chaining:** The pipeline chains multiple LLM calls: extract entities → parse to structured args → retrieve records → synthesize answer → next clue.
* **R4 — Tool / Function Calling:** Expose at least four tools via OpenAI-style tool schemas: `lookup_person_by_address`, `lookup_person_by_name`, `get_interview`, `search_gym_checkins`, `search_drivers_license`, `search_event_attendance`, `lookup_income`. The model selects tools; Python executes them; results return as `tool` messages.
* **R5 — RAG over Interviews:** Chunk and embed interview transcripts (and optionally other text-heavy records) using `text-embedding-3-small`. Store embeddings in an in-memory vector store (or MongoDB). Retrieve top-k chunks for each investigative question; the LLM answers from retrieved context only.
* **R6 — Output Evaluation / Groundedness Check:** After each accusation or major conclusion, run an evaluator prompt that checks whether every factual claim is directly supported by a cited tool result or RAG chunk. Unsupported claims are flagged and re-investigated.
* **R7 — Logging:** Persist the full conversation history (system prompts, user messages, assistant reasoning, tool calls/results) to an exportable log.
* **R8 — Final Output:** The system produces the murderer's name and the mastermind's name, each with a supporting evidence summary.
* **R9 (optional) — Multi-Agent Design:** Define a manager agent, a records-lookup worker agent, a transcript-analyst worker agent, and a critic agent. Document the architecture.
* **R10 (optional) — Multi-Agent Implementation:** Implement the multi-agent system using an orchestration loop (AutoGen-style or custom) so the case solves fully automatically.

## Interfaces & Data

**Data inputs (all JSON, loaded at startup):**
`person.json`, `drivers_license.json`, `income.json`, `get_fit_now_member.json`, `get_fit_now_check_in.json`, `facebook_event_checkin.json`, `interview.json`. **Linkage hub is `person.id`.**

**Tools exposed to the LLM (R4):**

* `lookup_person(name=None, address_street=None, city=None)` → rows from `person.json`
* `get_interview(person_id)` → transcript text from `interview.json`
* `search_gym_members(membership_id_prefix=None, status=None)` → matching members + check-in logs
* `search_drivers_license(plate_contains=None, hair_color=None, car_make=None, gender=None, height_range=None)` → matching license rows
* `search_event_attendance(person_id, date=None)` → matching event check-ins
* `lookup_income(ssn)` → annual income

**RAG corpus (R5):** Non-empty rows of `interview.json`, chunked per person, embedded and indexed for cosine similarity search.

## Prompting / Reasoning Plan

* **System prompt** establishes the detective role, forbids fabricating evidence, requires every claim to cite a specific tool output or retrieved chunk, and mandates CoT format (observation → assumption → query plan → result interpretation → next step or conclusion).
* **CoT enforcement (R2):** Each reasoning turn must output a JSON block with fields: `current_clue`, `assumptions`, `planned_action`, `tool_or_rag_query`, `result_interpretation`, `next_step_or_conclusion`. The outer loop parses this to decide whether to continue or finalize.
* **Prompt chaining (R3):** The pipeline is: (a) scene report → extract witness criteria → tool call → get witness person records; (b) witness person\_ids → RAG/tool retrieve interviews → extract suspect descriptors; (c) descriptors → tool queries on licenses/gym/events → narrow suspects; (d) suspect interview → extract mastermind lead → repeat (c)–(d).
* **Verification-first prompting:** Before any accusation, the LLM is prompted: "List every piece of evidence for and against this suspect. State whether each piece comes from a tool result or RAG chunk. Only accuse if net evidence is sufficient."

## Groundedness & Evaluation

* **Evaluator pass (R6):** After the agent names a suspect, a separate LLM call receives the accusation plus all cited tool/RAG outputs and answers: `{"supported": true/false, "unsupported_claims": [...], "missing_checks": [...]}`. If not fully supported, the main loop re-investigates.
* **Hallucination guard:** Tool functions return verbatim JSON rows. The system prompt instructs: "If a tool returns no matching records, state that explicitly. Do not invent records."
* **Final cross-check:** Before producing the final answer, the system verifies both the murderer and the mastermind by confirming at least two independent evidence links (e.g., gym log + interview mention, or license plate match + event attendance).

## Traceability Matrix

| PRD Requirement                       | Rubric Item                                 |
| ------------------------------------- | ------------------------------------------- |
| R1 (Prompt Design)                    | 1 — Prompt LLMs appropriately               |
| R2 (CoT), R3 (Prompt Chaining)        | 2 — Construct Chain-of-Thoughts             |
| R6 (Evaluator pass), R7 (Logging)     | 3 — Analyze the output of LLMs              |
| R4 (Tool schemas + execution loop)    | 4 — Utilize tools / function calling        |
| R5 (Embed + vector search interviews) | 5 — Construct and utilize RAG               |
| R9 (Multi-agent design doc)           | 6 — Design multi-agent system (optional)    |
| R10 (Multi-agent implementation)      | 7 — Implement multi-agent system (optional) |
| R7 (Log export)                       | Deliverable 2 — Running history PDF         |
| R1–R6 documented in report            | Deliverable 3 — Report PDF                  |

## Milestones

* **M1 — Core pipeline (days 1–2):** Data loading, tool functions (R4), RAG index over interviews (R5), basic CoT prompt (R2), and prompt chaining (R3) sufficient to solve the case interactively.
* **M2 — Evaluation & polish (day 3):** Evaluator/critic pass (R6), hallucination guards, full logging (R7), end-to-end run producing murderer + mastermind names (R8). Export running history PDF and write report PDF.
* **M3 — Optional multi-agent (day 4, if time permits):** Design doc (R9) and implementation (R10) of manager/worker/critic architecture for fully automatic solving.
