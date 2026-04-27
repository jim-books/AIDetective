"""Entry point: wire data → RAG → agent → evaluator → logger → run the case."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import AgentState, feed_reinvestigation, run_investigation
from .data import load_evidence
from .evaluator import evaluate_accusation
from .logger import export_markdown, write_jsonl
from .rag import build_index, openai_embedder

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_client():
    """Build an OpenAI or AzureOpenAI client from environment variables.

    Azure is used when AZURE_OPENAI_API_KEY is set.
    Falls back to standard OpenAI when OPENAI_API_KEY is set.
    """
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        from openai import AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://hkust.azure-api.net")
        # The SDK appends /openai automatically; strip it if the user included it.
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/openai"):
            endpoint = endpoint[: -len("/openai")]
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
        return AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI
        return OpenAI(api_key=openai_key)
    return None


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

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

    print(f"Running detective loop (chat={chat_model}) …", flush=True)
    state = run_investigation(
        client=client,
        ev=ev,
        vector_index=index,
        embed_fn=embed_fn,
        model=chat_model,
    )

    # Evaluator pass — one clean retry per failed role, fresh state to avoid context bloat.
    verified: list = []
    failed_roles: set[str] = set()

    def _evaluate_all(accusations):
        nonlocal verified
        passed, failed = [], []
        for acc in accusations:
            try:
                verdict = evaluate_accusation(
                    client=client, accusation=acc, state=state, model=eval_model,
                )
            except Exception as e:
                print(f"Evaluator error for {acc.role}: {e}")
                verdict = None
            supported = verdict.supported if verdict else False
            print(f"Evaluator on {acc.role}={acc.name} (id={acc.person_id}): supported={supported}")
            if supported:
                verified.append(acc)
                passed.append(acc)
            else:
                failed.append((acc, verdict))
        return passed, failed

    _, failures = _evaluate_all(state.accusations)

    # One re-investigation per failed role (fresh state avoids 120-msg context)
    for failed_acc, verdict in failures:
        if failed_acc.role in failed_roles:
            continue  # already retried this role
        failed_roles.add(failed_acc.role)
        missing = verdict.missing_checks if verdict else []
        unsupported = verdict.unsupported_claims if verdict else []
        hint = (
            f"Re-investigate the {failed_acc.role}. Previous suspect {failed_acc.name} "
            f"(id={failed_acc.person_id}) was rejected. Missing: {missing}. "
            f"Unsupported: {unsupported}. Find stronger evidence."
        )
        fresh = AgentState()
        fresh.accusations.extend(verified)  # seed with already-approved roles
        run_investigation(
            client=client, ev=ev, vector_index=index, embed_fn=embed_fn,
            model=chat_model, state=fresh, extra_user_messages=[hint],
        )
        new_accs = [a for a in fresh.accusations if a not in verified and a not in state.accusations]
        state.accusations.extend(new_accs)
        state.evidence_log.extend(fresh.evidence_log)
        state.history.extend(fresh.history)
        _evaluate_all(new_accs)

    jsonl_path = write_jsonl(state)
    md_path = export_markdown(jsonl_path)
    print(f"\nWrote run log:    {jsonl_path}")
    print(f"Wrote markdown:   {md_path}")

    print("\n=== Final accusations ===")
    for a in state.accusations:
        print(f"  {a.role}: {a.name} (id={a.person_id})")
        for ev_item in a.evidence:
            print(f"    - {ev_item.get('source')}: {ev_item.get('claim')}")

    roles = {a.role for a in state.accusations}
    if roles >= {"murderer", "mastermind"}:
        return 0
    print("\nWARNING: did not converge on both roles.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
