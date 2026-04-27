# 🕵️ Detective Agent System

Grounded LLM detective for HKUST AI for Design Assignment 5.  
The agent uses tool calling + RAG over interview transcripts to identify both the **murderer** and the **mastermind**.

## ✨ Features

- Evidence-first investigation with no-hallucination constraints
- Function-calling tools for person, interview, gym, license, event, and income lookup
- RAG retrieval over interview transcripts (`text-embedding-3-small`)
- Evaluator pass to verify accusation support before final output
- Run logging to JSONL and Markdown in `runs/`

## 📂 Project Structure

- `src/detective/` — core agent, tools, RAG, evaluator, logger
- `Evidence/` — JSON evidence tables
- `tests/` — unit tests (including RAG tests)
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

```bash
python -m detective.main
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

## 🔎 Evidence Chain (Short)

- **Murderer path**: witness clues (`Northwestern Dr`, `Annabel`) -> gym prefix `48Z` + gold + Jan 9 check-in -> plate fragment `H42W` -> `Jeremy Bowers`
- **Mastermind path**: Bowers interview description (female, red hair, Tesla Model S, 5'5"-5'7", SQL Symphony Dec 2017) -> license filter -> event attendance verification -> `Miranda Priestly`

## 📄 Full Report

Detailed write-up is in `REPORT.md` (prompt design, tooling, RAG, evaluation, and complete evidence justification).

## 🧪 Test

```bash
pytest
```

## 📝 Deliverables

- Source code implementing tool use, CoT loop, and RAG
- Running history PDF with final murderer + mastermind
- Brief report mapping implementation to rubric requirements
