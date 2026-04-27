"""Persist the agent conversation to JSONL and export a markdown trace."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import AgentState

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


def write_jsonl(state: AgentState, path: Path | None = None) -> Path:
    """Write each history turn + the final accusations as a JSONL file."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RUNS_DIR / f"{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for turn in state.history:
            f.write(json.dumps(turn.to_message(), ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "role": "system",
            "name": "summary",
            "accusations": [
                {"role": a.role, "person_id": a.person_id, "name": a.name, "evidence": a.evidence}
                for a in state.accusations
            ],
        }, ensure_ascii=False) + "\n")
    return path


def export_markdown(jsonl_path: Path, md_path: Path | None = None) -> Path:
    """Render the JSONL trace as a markdown document for the PDF deliverable."""
    md_path = md_path or jsonl_path.with_suffix(".md")
    lines: list[str] = ["# Detective Investigation — Running History\n"]
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            obj = json.loads(raw)
            role = obj.get("role", "?")

            if role == "system" and obj.get("name") == "summary":
                lines.append("\n---\n## Final accusations\n")
                for a in obj.get("accusations", []):
                    lines.append(f"- **{a['role']}**: {a['name']} (id={a['person_id']})")
                    for ev in a.get("evidence", []):
                        lines.append(f"    - `{ev.get('source')}` — {ev.get('claim')}")
                continue

            lines.append(f"\n### {role}")
            content = obj.get("content")
            if content:
                # Pretty-print embedded JSON if possible
                try:
                    parsed = json.loads(content) if isinstance(content, str) else content
                    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                    lines.append(f"```json\n{pretty}\n```")
                except (json.JSONDecodeError, TypeError):
                    lines.append(str(content))
            if obj.get("tool_calls"):
                lines.append(f"```json\n{json.dumps(obj['tool_calls'], indent=2)}\n```")
            if obj.get("tool_call_id"):
                lines.append(f"_tool_call_id_: `{obj['tool_call_id']}`  _name_: `{obj.get('name')}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
