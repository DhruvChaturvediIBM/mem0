"""Token usage comparison: naive (no mem0) vs mem0-style (Db2 vector store).

Shows exactly how many tokens an LLM would consume:
  - WITHOUT mem0: entire conversation history stuffed into every prompt
  - WITH mem0:    only the top-k retrieved facts injected into the prompt,
                  retrieved via your mem0.vector_stores.db2.Db2VectorStore
                  configured through mem0.configs.vector_stores.db2.Db2Config

Classes used from mem0:
  mem0/configs/vector_stores/db2.py  → Db2Config       (Pydantic config model)
  mem0/vector_stores/db2.py          → Db2VectorStore  (insert / search / reset)

Token counting uses tiktoken (cl100k_base — GPT-4 encoding).

Run:
    DB2_PASSWORD=pass python3.11 examples/misc/db2_token_comparison.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

import tiktoken

# ── mem0 imports — YOUR actual classes ───────────────────────────────────────
# mem0/configs/vector_stores/db2.py  →  Db2Config       (Pydantic config model)
# mem0/vector_stores/db2.py          →  Db2VectorStore  (insert/search/reset)
from mem0.configs.vector_stores.db2 import Db2Config      # noqa: E402
from mem0.vector_stores.db2 import Db2VectorStore          # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("token_comparison")
logging.getLogger("mem0.vector_stores.db2").setLevel(logging.WARNING)
logging.getLogger("ibm_db_dbi").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config — Db2 connection (same env vars as all other examples)
# ---------------------------------------------------------------------------
CONNECTION_PARAMS = {
    "database": os.environ.get("DB2_DATABASE", "TESTDB"),
    "host":     os.environ.get("DB2_HOST", "127.0.0.1"),
    "port":     int(os.environ.get("DB2_PORT", "50000")),
    "username": os.environ.get("DB2_USERNAME", ""),
    "password": os.environ.get("DB2_PASSWORD", ""),
}

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text"
DATA_FILE    = Path(__file__).parent / "db2_vector_data.json"
TABLE_NAME   = "MEM0_TOKEN_CMP"
EMBEDDING_DIM = 768
TOP_K        = 3
ENCODING     = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

QUERIES = [
    "What pets do people own?",
    "Who is working with machine learning or AI?",
    "What programming languages are people using?",
    "Who does outdoor activities or fitness?",
    "What food-related things are people into?",
]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to user memories. "
    "Answer questions using the provided context about users."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def embed(text: str) -> list[float]:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["embedding"]


def _sep(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print("═" * 65)


def _subsep(title: str) -> None:
    print(f"\n  {'─' * 55}")
    print(f"  {title}")
    print(f"  {'─' * 55}")


# ---------------------------------------------------------------------------
# Scenario A — WITHOUT mem0
# Full conversation history in every LLM prompt
# ---------------------------------------------------------------------------

def scenario_without_mem0(records: list[dict], queries: list[str]) -> dict:
    _sep("SCENARIO A — Without mem0  (full history in every prompt)")
    print(f"\n  ℹ  Does NOT use Db2VectorStore — history is raw text, never stored")

    conversation_history = "\n".join(
        f"[Turn {i+1}] {r['user_id'].capitalize()}: {r['text']}"
        for i, r in enumerate(records)
    )

    system_tokens  = count_tokens(SYSTEM_PROMPT)
    history_tokens = count_tokens(conversation_history)

    print(f"\n  System prompt      : {system_tokens:>6} tokens")
    print(f"  Full history       : {history_tokens:>6} tokens  (pasted into EVERY call)")

    total_retrieval_tokens = 0
    query_breakdown = []

    for q in queries:
        query_tokens  = count_tokens(q)
        prompt_tokens = system_tokens + history_tokens + query_tokens
        total_retrieval_tokens += prompt_tokens
        query_breakdown.append({
            "query": q,
            "query_tokens": query_tokens,
            "context_tokens": history_tokens,
            "total_prompt_tokens": prompt_tokens,
        })
        print(f"\n  Query: '{q}'")
        print(f"    system={system_tokens} + history={history_tokens} + query={query_tokens} = {prompt_tokens} tokens")

    print(f"\n  ── Totals ──────────────────────────────────────────────")
    print(f"  Ingestion tokens (one-time store cost):      0  [no store]")
    print(f"  Retrieval tokens ({len(queries)} queries combined): {total_retrieval_tokens:>6}")
    print(f"  GRAND TOTAL                              : {total_retrieval_tokens:>6}")

    return {
        "ingestion_tokens": 0,
        "retrieval_tokens": total_retrieval_tokens,
        "grand_total": total_retrieval_tokens,
        "per_query": query_breakdown,
    }


# ---------------------------------------------------------------------------
# Scenario B — WITH mem0  (Db2VectorStore via Db2Config)
# Ingestion: build Db2Config → Db2VectorStore → embed + insert each fact
# Retrieval: embed query → Db2VectorStore.search() → inject top-k only
# ---------------------------------------------------------------------------

def scenario_with_mem0(records: list[dict], queries: list[str]) -> dict:
    _sep("SCENARIO B — With mem0  (Db2VectorStore + Db2Config)")

    # ── Show exactly which mem0 classes are being used ────────────────────
    print(f"\n  ✅  Db2Config       → mem0/configs/vector_stores/db2.py")
    print(f"  ✅  Db2VectorStore  → mem0/vector_stores/db2.py")
    print(f"\n  Db2Config fields being set:")
    print(f"    collection_name    = {TABLE_NAME}")
    print(f"    embedding_model_dims = {EMBEDDING_DIM}")
    print(f"    distance_strategy  = COSINE")
    print(f"    connection_params  = {{host={CONNECTION_PARAMS['host']}, port={CONNECTION_PARAMS['port']}, db={CONNECTION_PARAMS['database']}}}")

    # ── Instantiate via Db2Config (as mem0 does internally) ───────────────
    # Db2VectorStore.__init__ calls Db2Config(**kwargs) at line 133 of db2.py
    # We pass the same kwargs — this exercises the full config validation path
    log.info("Building Db2Config and Db2VectorStore …")
    store = Db2VectorStore(
        connection_params=CONNECTION_PARAMS,
        collection_name=TABLE_NAME,
        embedding_model_dims=EMBEDDING_DIM,
        distance_strategy="COSINE",
    )
    # Confirm the config object was created
    print(f"\n  Db2Config object   : {store.config.__class__.__module__}.{store.config.__class__.__name__}")
    print(f"    .collection_name   = {store.config.collection_name}")
    print(f"    .embedding_model_dims = {store.config.embedding_model_dims}")
    print(f"    .distance_strategy = {store.config.distance_strategy}")
    log.info("store.config = %s", store.config)

    # Reset so re-runs never hit duplicate-key errors
    log.info("Resetting table %s …", TABLE_NAME)
    store.reset()

    # ── Ingestion phase ───────────────────────────────────────────────────
    _subsep("Ingestion — tokenise + embed + Db2VectorStore.insert()")

    total_ingestion_tokens = 0
    vectors  = []
    payloads = []
    ids      = []

    for i, rec in enumerate(records):
        tok = count_tokens(rec["text"])
        total_ingestion_tokens += tok
        vec = embed(rec["text"])
        vectors.append(vec)
        payloads.append({
            "data":     rec["text"],
            "user_id":  rec["user_id"],
            "category": rec["category"],
        })
        ids.append(rec["id"])
        print(f"  [{i:02d}] tokens={tok:>3}  id={rec['id']}  '{rec['text'][:55]}...'")

    # Insert via YOUR Db2VectorStore.insert()
    stored_ids = store.insert(vectors=vectors, payloads=payloads, ids=ids)
    log.info("Inserted %d records via Db2VectorStore.insert()", len(stored_ids))
    print(f"\n  Db2VectorStore.insert() → {len(stored_ids)} records written to Db2")
    print(f"  Total ingestion tokens (facts tokenised for storage): {total_ingestion_tokens}")

    # ── Retrieval phase ───────────────────────────────────────────────────
    _subsep(f"Retrieval — embed query → Db2VectorStore.search(top_k={TOP_K}) → inject")

    system_tokens = count_tokens(SYSTEM_PROMPT)
    total_retrieval_tokens = 0
    query_breakdown = []

    for q in queries:
        query_tokens = count_tokens(q)
        query_vec    = embed(q)

        # ← This is YOUR Db2VectorStore.search() doing cosine similarity in Db2
        results = store.search(query=q, vectors=[query_vec], top_k=TOP_K)

        context_text   = "\n".join(
            f"- [{r.payload.get('user_id')}] {r.payload.get('data', '')}"
            for r in results
        )
        context_tokens = count_tokens(context_text)
        prompt_tokens  = system_tokens + context_tokens + query_tokens
        total_retrieval_tokens += prompt_tokens

        query_breakdown.append({
            "query": q,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "total_prompt_tokens": prompt_tokens,
        })

        print(f"\n  Query: '{q}'")
        print(f"    Db2VectorStore.search() top-{TOP_K}:")
        for r in results:
            print(f"      score={r.score:.3f}  user={r.payload.get('user_id'):<8}  '{r.payload.get('data','')[:55]}...'")
        print(f"    system={system_tokens} + context={context_tokens} + query={query_tokens} = {prompt_tokens} tokens")

    print(f"\n  ── Totals ──────────────────────────────────────────────")
    print(f"  Ingestion tokens (one-time store cost):   {total_ingestion_tokens:>6}")
    print(f"  Retrieval tokens ({len(queries)} queries combined): {total_retrieval_tokens:>6}")
    print(f"  GRAND TOTAL                            : {total_ingestion_tokens + total_retrieval_tokens:>6}")

    return {
        "ingestion_tokens": total_ingestion_tokens,
        "retrieval_tokens": total_retrieval_tokens,
        "grand_total": total_ingestion_tokens + total_retrieval_tokens,
        "per_query": query_breakdown,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(without: dict, with_mem0: dict) -> None:
    _sep("COMPARISON SUMMARY")

    print(f"\n  {'Metric':<45} {'Without mem0':>14} {'With mem0':>12}")
    print(f"  {'─'*45} {'─'*14} {'─'*12}")
    print(f"  {'Ingestion tokens (one-time store cost)':<45} {without['ingestion_tokens']:>14,} {with_mem0['ingestion_tokens']:>12,}")
    print(f"  {'Retrieval tokens (total over all queries)':<45} {without['retrieval_tokens']:>14,} {with_mem0['retrieval_tokens']:>12,}")
    print(f"  {'GRAND TOTAL':<45} {without['grand_total']:>14,} {with_mem0['grand_total']:>12,}")

    saved = without['grand_total'] - with_mem0['grand_total']
    pct   = (saved / without['grand_total'] * 100) if without['grand_total'] else 0
    print(f"\n  Tokens saved by mem0: {saved:,}  ({pct:.1f}% reduction)")

    print(f"\n  {'─' * 63}")
    print(f"  Per-query retrieval tokens (one LLM call each):")
    print(f"  {'─' * 63}")
    print(f"  {'Query':<42} {'Without':>9} {'With':>9} {'Saved':>8}")
    print(f"  {'─'*42} {'─'*9} {'─'*9} {'─'*8}")
    for wout, wmem in zip(without["per_query"], with_mem0["per_query"]):
        q     = wout["query"][:40]
        diff  = wout["total_prompt_tokens"] - wmem["total_prompt_tokens"]
        print(f"  {q:<42} {wout['total_prompt_tokens']:>9,} {wmem['total_prompt_tokens']:>9,} {diff:>8,}")

    print(f"\n  Note: mem0 injects top-{TOP_K} relevant facts per query via Db2VectorStore.search().")
    print(f"  Ingestion is paid ONCE. Savings compound with every additional query.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CONNECTION_PARAMS["password"]:
        log.error("DB2_PASSWORD is not set. Run: export DB2_PASSWORD=<password>")
        sys.exit(1)

    with open(DATA_FILE) as f:
        records = json.load(f)
    log.info("Loaded %d records from %s", len(records), DATA_FILE.name)

    # Verify Ollama
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            model_names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if not any(OLLAMA_MODEL in n for n in model_names):
            log.error("Model '%s' not found. Run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)
            sys.exit(1)
        log.info("Ollama OK — '%s' available", OLLAMA_MODEL)
    except Exception as exc:
        log.error("Cannot reach Ollama: %s", exc)
        sys.exit(1)

    print(f"\n  Tokenizer  : tiktoken cl100k_base (GPT-4 encoding)")
    print(f"  Records    : {len(records)} facts  |  Queries: {len(QUERIES)}  |  Top-K: {TOP_K}")
    print(f"  Vector store: Db2VectorStore (mem0/vector_stores/db2.py)")
    print(f"  Config class: Db2Config      (mem0/configs/vector_stores/db2.py)")

    without  = scenario_without_mem0(records, QUERIES)
    with_mem0 = scenario_with_mem0(records, QUERIES)
    print_comparison(without, with_mem0)


if __name__ == "__main__":
    main()
