# CHANGELOG

Append-only progress log per the workflow. Newest entry on top.

## 2026-04-27 — M1 complete, M2 blocked on HKUST credit

### M2 status — live run attempts
- Previous run (before credit exhaustion): agent completed 40 turns, found mastermind = **Miranda Priestly (id=99716)** ✓ (correct). Murderer = **Joe Germuska (id=28819)** ✗ (evaluator rejected — both Joe Germuska and Jeremy Bowers match the gym criteria; Jeremy Bowers is the canonical murderer identified by whose interview leads to the mastermind).
- Prompt fix added: system prompt now explicitly requires interviewing ALL gym suspects and using the mastermind-lead as the tiebreaker.
- Evaluator switched from gpt-4o to gpt-4o-mini to stay within credit budget.
- Null-content bug fixed in `TurnRecord.to_message()`: Azure litellm requires `content=""` for assistant tool-call messages, not omitted field.
- Re-investigation loop simplified: one fresh-state retry per failed role (avoids 120-msg context bloat that caused the previous crash).
- HKUST Azure credit exhausted. Run resumes after top-up at https://pmt2.ust.hk/openai/.
- **Embedding cache saved at `.cache/embeddings.npz`** — next run skips re-embedding.

## 2026-04-27

### Step 9 — main.py wiring + editable install
- Added [src/detective/main.py](src/detective/main.py): loads `.env`, instantiates `OpenAI()`, builds index, runs `run_investigation`, then loops over the evaluator (up to 2 re-investigation rounds per failed accusation), writes JSONL + markdown logs, and prints the final answer. Exit codes: 0 = both roles found and verified, 1 = did not converge, 2 = missing API key.
- Ran `pip install -e .` in `.venv` so `python -m detective.main` resolves the package outside of pytest. (Pytest works without the install because of `pythonpath=["src"]`.)

### Step 8 — logger.py
- Added [src/detective/logger.py](src/detective/logger.py): `write_jsonl(state)` dumps every history turn + a final summary block to `runs/<timestamp>.jsonl`. `export_markdown(jsonl_path)` renders the same trace as a readable markdown document for the running-history PDF.

### Step 7 — evaluator.py
- Added [src/detective/evaluator.py](src/detective/evaluator.py): `evaluate_accusation(...)` calls `gpt-4o` with the accusation + the agent's full `evidence_log` and parses a strict JSON verdict. Locally enforces the two-link rule (≥2 distinct sources) on top of whatever the LLM returned, so a credulous evaluator cannot bypass the constraint.

### Step 6 — agent.py + offline integration test
- Added [src/detective/agent.py](src/detective/agent.py): `run_investigation(client, ev, vector_index, embed_fn, ...)` drives the CoT loop. Adds a synthetic `rag_search` tool to `ALL_TOOL_SCHEMAS` so the LLM uses one channel for everything. Maintains an `AgentState` (history + `evidence_log` for the evaluator + accusation list). Loop exits early once both `murderer` and `mastermind` accusations are emitted, or after `max_turns=40`.
- Robust to a model emitting JSON wrapped in ```` ```json ```` fences (the `_parse_cot` salvages it).
- When the model emits a "continue" CoT with no tool call, the loop appends a nudge user message rather than burning a turn silently.
- Added [tests/test_agent_offline.py](tests/test_agent_offline.py): 2 tests using a hand-rolled `FakeClient` that replays scripted `chat.completions.create` responses. Covers the happy path (tool-call → tool-result → murderer accusation → mastermind accusation), the nudge fallback path, and verifies the evidence_log captures the executed tool. **All 24 tests pass.**

### Step 5 — prompts.py
- Added [src/detective/prompts.py](src/detective/prompts.py): `DETECTIVE_SYSTEM`, `INITIAL_USER_MESSAGE`, `EVALUATOR_SYSTEM`, `evaluator_user_prompt`, `reinvestigation_prompt`. System prompt encodes the hard rules (no fabrication, two-link rule, schema correction re: `address_street_name`/no city), the CoT JSON schema, the tool inventory, and the starting clue.

### Step 4 — rag.py + tests
- Added [src/detective/rag.py](src/detective/rag.py): `build_chunks` (one chunk per non-empty interview), `openai_embedder` (batched 500-input calls to `text-embedding-3-small`), `build_index` (with content-hash-keyed cache at `.cache/embeddings.npz`), `VectorIndex.search` (cosine top-k via numpy matmul on L2-normalised vectors), and `retrieve(index, query, embed_fn)`.
- Added [tests/test_rag.py](tests/test_rag.py): five tests using a deterministic fake embedder — empty-transcript skip, build/search shape + ranking, cache hit avoidance via mocker spy, cache invalidation on corpus change, and a real-corpus assertion that 4991 chunks load from disk. **All 22 tests pass.**
- Embedder is injected (`EmbedFn` callable) so the test suite never touches the network.

### Step 3 — tools.py + tests
- Added [src/detective/tools.py](src/detective/tools.py): six tool functions (`lookup_person`, `get_interview`, `search_gym_members`, `search_drivers_license`, `search_event_attendance`, `lookup_income`) + `TOOL_SCHEMAS` (OpenAI function-calling JSON Schemas) + `dispatch(ev, name, args)`. All tools return `{results, count, note?}` on misses, never raise on bad args (returns `{"error": ...}`), and pass verbatim rows on hits.
- Added convenience flag `last_house_on_street=True` to `lookup_person` so the agent can request "last house on Northwestern Dr" in a single call without doing the max-aggregation itself.
- `search_drivers_license` enriches each license with the linked `person` (via `person_by_license_id`) so the agent doesn't need a follow-up call for every match.
- Added [tests/test_tools.py](tests/test_tools.py): 13 tests covering happy path + miss path on every tool, dispatch error paths, and schema shape. **All 17 tests pass (4 data + 13 tools).**

### Step 2 — data.py + tests
- Added [src/detective/data.py](src/detective/data.py): `load_evidence()` returns an `Evidence` dataclass with raw lists + indices (`person_by_id`, `person_by_license_id`, `license_by_id`, `income_by_ssn`, `members_by_person_id`, `member_by_id`, `checkins_by_membership_id`, `events_by_person_id`, `interview_by_person_id`).
- Added [tests/test_data.py](tests/test_data.py): row counts, witnesses present (Morty 14887, Annabel 16371), license linkage, and "last house on Northwestern Dr" derivation. **4/4 tests pass.**
- **Note on CLAUDE.md drift:** CLAUDE.md says 3,738 non-empty interviews, actual is 4,991 (every row populated). Coding to actuals.

### Step 1 — Scaffold
- Added [pyproject.toml](pyproject.toml): openai, numpy, python-dotenv runtime deps; pytest + pytest-mock dev extras; pytest configured to use `tests/` and `src/` on path.
- Added [.env.example](.env.example) with `OPENAI_API_KEY=` placeholder.
- Updated [.gitignore](.gitignore) to ignore `runs/`, `.cache/`, `.pytest_cache/`, `.venv/`, root-level `running_history.pdf` / `report.pdf`.
- Created [CHANGELOG.md](CHANGELOG.md) (this file).
- Created `src/detective/__init__.py` package marker.
- Created `.venv/` (Python 3.13) and installed deps: openai 2.32.0, numpy 2.4.4, pytest 9.0.3, pytest-mock, python-dotenv. System Python is PEP 668 protected; `.venv/bin/python` is the canonical interpreter for this project. Documented for future sessions.

### Failed attempts
- (none yet)
