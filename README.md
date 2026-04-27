# 🕵️ Detective Agent System

Grounded LLM detective for HKUST AI for Design Assignment 5.  
The project includes both a **single-agent pipeline** (R1-R7) and an **optional multi-agent pipeline** (M3 bonus: design + implementation), with tool calling + RAG over interviews to identify both the **murderer** and the **mastermind**.

## ✨ Features

- Evidence-first investigation with no-hallucination constraints
- Function-calling tools for person, interview, gym, license, event, and income lookup
- RAG retrieval over interview transcripts (`text-embedding-3-small`)
- Evaluator pass to verify accusation support before final output
- Run logging to JSONL and Markdown in `runs/`
- Optional multi-agent architecture: `ManagerAgent`, `RecordsAgent`, `TranscriptAgent`, `CriticAgent`

## 📂 Project Structure

- `src/detective/agent.py` — single-agent CoT loop and tool execution
- `src/detective/multi_agent.py` — multi-agent orchestration and specialist loops
- `src/detective/main.py` — single-agent entry point
- `src/detective/multi_main.py` — multi-agent entry point
- `src/detective/tools.py` — tool implementations + schemas
- `src/detective/rag.py` — embedding index + retrieval
- `src/detective/evaluator.py` — independent groundedness checks
- `src/detective/logger.py` — JSONL/Markdown run export
- `Evidence/` — JSON evidence tables
- `tests/` — offline/unit tests (single-agent, RAG, tools, multi-agent)
- `.env.example` — environment variable template

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set one provider in `.env`:

- Azure OpenAI: `AZURE_OPENAI_API_KEY` (and deployment vars), or
- OpenAI: `OPENAI_API_KEY`

## 🚀 Run

Single-agent mode:

```bash
python -m detective.main
```

Multi-agent mode (optional M3):

```bash
python -m detective.multi_main
```

Expected final roles:

- Murderer: `Jeremy Bowers` (`person_id=67318`)
- Mastermind: `Miranda Priestly` (`person_id=99716`)

## ✅ Rubric Coverage (R1-R7)

- **R1 Prompting**: strict detective and evaluator prompts in `src/detective/prompts.py`
- **R2/R3 CoT + Chaining**: structured JSON reasoning loop in `src/detective/agent.py`
- **R4 Tool Calling**: six core tools + `rag_search` bridge in `src/detective/tools.py`
- **R5 RAG**: interview embedding and cosine retrieval in `src/detective/rag.py`
- **R6 Evaluation**: independent accusation validation in `src/detective/evaluator.py`
- **R7 Logging**: full trace export to JSONL/Markdown in `src/detective/logger.py`

## ⭐ Optional Bonus (M3)

- **Design**: orchestrator-as-caller multi-agent system
- **Implementation**: `ManagerAgent` delegates via tools to:
  - `RecordsAgent` (structured records tools)
  - `TranscriptAgent` (interview + `rag_search`)
  - `CriticAgent` (accusation validation wrapper around evaluator)
- **Safety rule**: accusations are committed only after critic validation returns `supported=true`

## 🔎 Evidence Chain (Short)

- **Murderer path**: witness clues (`Northwestern Dr`, `Annabel`) -> gym prefix `48Z` + gold + Jan 9 check-in -> plate fragment `H42W` -> `Jeremy Bowers`
- **Mastermind path**: Bowers interview description (female, red hair, Tesla Model S, 5'5"-5'7", SQL Symphony Dec 2017) -> license filter -> event attendance verification -> `Miranda Priestly`

## 📄 Full Report

Detailed write-up is in `REPORT.md` (R1-R7 mapping, M3 design/implementation, and complete evidence justification).

## 🧪 Test

```bash
pytest
```

Current status (from latest changelog): all tests passing, including new offline multi-agent tests.

## 📦 Output Artifacts

- Run traces: `runs/<timestamp>.jsonl` and `runs/<timestamp>.md`
- PDF deliverables can be generated from markdown exports (e.g., running history + report)

## 📝 Deliverables

- Source code implementing tool use, CoT loop, and RAG
- Running history PDF with final murderer + mastermind
- Brief report mapping implementation to rubric requirements
