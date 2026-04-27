"""Entry point for the multi-agent investigation mode."""

from __future__ import annotations

import sys

from .data import load_evidence
from .logger import export_markdown, write_jsonl
from .main import _make_client
from .multi_agent import ManagerAgent
from .rag import build_index, openai_embedder


def main() -> int:
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")

    client = _make_client()
    if client is None:
        print(
            "ERROR: Set AZURE_OPENAI_API_KEY (Azure) or OPENAI_API_KEY (standard OpenAI) in .env",
            file=sys.stderr,
        )
        return 2

    is_azure = os.environ.get("AZURE_OPENAI_API_KEY") is not None
    chat_model = (
        os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        if is_azure else "gpt-4o-mini"
    )
    eval_model = (
        os.environ.get("AZURE_OPENAI_EVAL_DEPLOYMENT", "gpt-4o")
        if is_azure else "gpt-4o"
    )

    print("Loading evidence …", flush=True)
    ev = load_evidence()
    print(f"  persons={len(ev.persons)} licenses={len(ev.licenses)} interviews={len(ev.interviews)}")

    print("Building / loading RAG index …", flush=True)
    embed_fn = openai_embedder(client)
    index = build_index(ev, embed_fn)
    print(f"  index size={len(index.chunks)}")

    print(f"Running multi-agent investigation (chat={chat_model}) …", flush=True)
    manager = ManagerAgent(
        client=client,
        ev=ev,
        vector_index=index,
        embed_fn=embed_fn,
        model=chat_model,
        eval_model=eval_model,
    )
    state = manager.run()

    jsonl_path = write_jsonl(state)
    md_path = export_markdown(jsonl_path)
    print(f"\nWrote run log:  {jsonl_path}")
    print(f"Wrote markdown: {md_path}")

    print("\n=== Final accusations ===")
    for a in state.accusations:
        print(f"  {a.role}: {a.name} (id={a.person_id})")
        for item in a.evidence:
            print(f"    - {item.get('source')}: {item.get('claim')}")

    roles = {a.role for a in state.accusations}
    if roles >= {"murderer", "mastermind"}:
        return 0
    print("\nWARNING: did not converge on both roles.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
