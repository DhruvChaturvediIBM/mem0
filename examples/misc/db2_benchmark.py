"""mem0 + IBM Db2 — Ingestion & Retrieval Benchmark (Basic → Complex).

Compares two scenarios:

  WITHOUT mem0  — Every LLM call gets the full raw conversation history pasted
                  in as context. Token count is the sum of system prompt +
                  entire history + query. No LLM call is actually made — we
                  measure the *input token cost* that would be incurred.

  WITH mem0     — Memory.add() runs the full pipeline:
                    conversation
                      → mem0/llms/ollama.py      OllamaLLM(llama3.2)
                                                  extracts facts as JSON
                      → mem0/embeddings/ollama.py OllamaEmbedding(nomic-embed-text)
                                                  768-dim vector
                      → mem0/vector_stores/db2.py Db2VectorStore
                                                  IBM Db2 VECTOR_DISTANCE()

                  Memory.search() then retrieves only the top-k relevant facts
                  per query, so the LLM only sees a small focused context.

Metrics collected
  - Input tokens  WITHOUT vs WITH (retrieval context only)
  - Token savings  absolute + percentage
  - Retrieval latency (ms) for Memory.search()
  - Precision@3 and Recall@3 against ground-truth relevant facts

Run:
    DB2_DATABASE=TESTDB DB2_HOST=127.0.0.1 DB2_PORT=50000 \\
    DB2_USERNAME=db2inst1 DB2_PASSWORD=pass \\
    python3.11 examples/misc/db2_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from mem0 import Memory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("db2_benchmark")
logging.getLogger("mem0").setLevel(logging.WARNING)
logging.getLogger("ibm_db_dbi").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("posthog").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Optional .env
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONNECTION_PARAMS = {
    "database": os.environ.get("DB2_DATABASE", "TESTDB"),
    "host":     os.environ.get("DB2_HOST", "127.0.0.1"),
    "port":     int(os.environ.get("DB2_PORT", "50000")),
    "username": os.environ.get("DB2_USERNAME", "db2inst1"),
    "password": os.environ.get("DB2_PASSWORD", ""),
}
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ENCODING        = tiktoken.get_encoding("cl100k_base")
SYSTEM_PROMPT   = (
    "You are a helpful AI assistant. Answer questions using only "
    "the provided context about the user."
)

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model":           "llama3.2",
            "ollama_base_url": OLLAMA_BASE_URL,
            "temperature":     0,
            "max_tokens":      2000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model":           "nomic-embed-text",
            "ollama_base_url": OLLAMA_BASE_URL,
            "embedding_dims":  768,
        },
    },
    "vector_store": {
        "provider": "db2",
        "config": {
            "collection_name":      "MEM0_BENCHMARK",
            "embedding_model_dims": 768,
            "distance_strategy":    "COSINE",
            "connection_params":    CONNECTION_PARAMS,
        },
    },
}

# ---------------------------------------------------------------------------
# Benchmark dataset
#
# Each user has:
#   conversations  — multi-turn dialogue fed to Memory.add()
#   full_history   — flat string of all turns (used for WITHOUT baseline)
#
# Each query has:
#   text       — the question
#   user_id    — which user's memory to search (None = cross-user)
#   relevant   — keyword phrases that a correct answer MUST contain
#   complexity — BASIC / MEDIUM / COMPLEX
# ---------------------------------------------------------------------------

USERS = {
    "alice": {
        "conversations": [
            [
                {"role": "user",      "content": "Hi! I just got a golden retriever puppy named Max. He loves fetch."},
                {"role": "assistant", "content": "Wonderful! How old is Max?"},
                {"role": "user",      "content": "Three months old. I also eat vegetarian food and I'm allergic to nuts."},
            ],
            [
                {"role": "user",      "content": "I work as a UX designer at a healthcare startup in San Francisco."},
                {"role": "assistant", "content": "Interesting! What kind of products do you design?"},
                {"role": "user",      "content": "Patient-facing apps. I love hiking on weekends too."},
            ],
        ],
    },
    "bob": {
        "conversations": [
            [
                {"role": "user",      "content": "I'm a senior Python engineer at a fintech startup."},
                {"role": "assistant", "content": "What kind of projects are you working on?"},
                {"role": "user",      "content": "Data pipelines with Apache Spark. I adopted two rescue cats last summer."},
            ],
            [
                {"role": "user",      "content": "I play chess competitively and recently reached an Elo of 1800."},
                {"role": "assistant", "content": "Impressive! Do you train online?"},
                {"role": "user",      "content": "Yes, mostly on chess.com. I also love reading sci-fi novels."},
            ],
        ],
    },
    "charlie": {
        "conversations": [
            [
                {"role": "user",      "content": "I'm training for a marathon and track my runs with a GPS watch."},
                {"role": "assistant", "content": "How far along is your training?"},
                {"role": "user",      "content": "About three months in. I also meal-prep high-protein food every Sunday."},
            ],
            [
                {"role": "user",      "content": "I'm learning Rust for systems programming. Memory safety is fascinating."},
                {"role": "assistant", "content": "Rust has a steep learning curve. What are you building?"},
                {"role": "user",      "content": "A small game engine. I also contribute to open-source projects."},
            ],
        ],
    },
    "diana": {
        "conversations": [
            [
                {"role": "user",      "content": "I do deep learning research and recently published a paper on transformers."},
                {"role": "assistant", "content": "Fascinating! Which conference?"},
                {"role": "user",      "content": "NeurIPS. I also volunteer at an animal shelter on weekends."},
            ],
            [
                {"role": "user",      "content": "I build microservices in Go and deploy on Kubernetes in production."},
                {"role": "assistant", "content": "Cloud-native stack! Any favourite tools?"},
                {"role": "user",      "content": "Helm and Argo CD. I also run a vegetarian cooking blog."},
            ],
        ],
    },
}

# Build flat full-history string per user (used for WITHOUT baseline)
def _flat_history(user_id: str) -> str:
    lines = []
    for conv in USERS[user_id]["conversations"]:
        for msg in conv:
            lines.append(f"[{msg['role'].capitalize()}] {msg['content']}")
    return "\n".join(lines)

ALL_HISTORY = "\n\n".join(
    f"=== {uid.capitalize()} ===\n{_flat_history(uid)}" for uid in USERS
)

BENCHMARK_QUERIES = [
    # ── BASIC: single fact, direct match ─────────────────────────────
    {
        "id": "B1",
        "complexity": "BASIC",
        "text": "What pet does Alice have?",
        "user_id": "alice",
        "relevant": ["golden retriever", "Max"],
        "description": "Direct pet ownership lookup",
    },
    {
        "id": "B2",
        "complexity": "BASIC",
        "text": "What programming language is Charlie learning?",
        "user_id": "charlie",
        "relevant": ["Rust", "systems"],
        "description": "Single fact — language learning",
    },
    {
        "id": "B3",
        "complexity": "BASIC",
        "text": "What does Bob do for work?",
        "user_id": "bob",
        "relevant": ["Python", "engineer", "fintech"],
        "description": "Occupation lookup",
    },
    # ── MEDIUM: partial context, category match ───────────────────────
    {
        "id": "M1",
        "complexity": "MEDIUM",
        "text": "What cloud-native technologies does Diana work with?",
        "user_id": "diana",
        "relevant": ["Kubernetes", "Go", "microservices"],
        "description": "Category match — cloud/k8s tech",
    },
    {
        "id": "M2",
        "complexity": "MEDIUM",
        "text": "What fitness and health activities does Charlie do?",
        "user_id": "charlie",
        "relevant": ["marathon", "running", "meal-prep"],
        "description": "Multi-fact — fitness category",
    },
    {
        "id": "M3",
        "complexity": "MEDIUM",
        "text": "What are Alice's dietary restrictions?",
        "user_id": "alice",
        "relevant": ["vegetarian", "nuts", "allergic"],
        "description": "Dietary preference lookup",
    },
    # ── COMPLEX: semantic inference, cross-domain ─────────────────────
    {
        "id": "C1",
        "complexity": "COMPLEX",
        "text": "Who would be best to discuss neural networks and AI research with?",
        "user_id": "diana",
        "relevant": ["deep learning", "transformers", "NeurIPS"],
        "description": "Semantic inference — ML research",
    },
    {
        "id": "C2",
        "complexity": "COMPLEX",
        "text": "What evidence is there that Bob combines technical expertise with non-work hobbies?",
        "user_id": "bob",
        "relevant": ["chess", "Elo", "sci-fi", "cats"],
        "description": "Cross-domain — tech + hobbies",
    },
    {
        "id": "C3",
        "complexity": "COMPLEX",
        "text": "Who combines software engineering with a healthy lifestyle or outdoor activities?",
        "user_id": None,   # cross-user
        "relevant": ["charlie", "alice", "hiking", "marathon", "meal-prep"],
        "description": "Cross-domain — tech + fitness/outdoor",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    query_id:             str
    complexity:           str
    query_text:           str
    description:          str
    without_input_tokens: int   = 0
    with_retrieval_tokens: int  = 0
    without_latency_ms:   float = 0.0
    with_latency_ms:      float = 0.0
    retrieved_memories:   list  = field(default_factory=list)
    precision_at_3:       float = 0.0
    recall_at_3:          float = 0.0
    top_retrieved_text:   str   = ""
    top_retrieved_score:  float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tok(text: str) -> int:
    return len(ENCODING.encode(text))


def _sep(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print("═" * 68)


def _subsep(title: str) -> None:
    print(f"\n  {'─' * 62}")
    print(f"  {title}")
    print(f"  {'─' * 62}")


def precision_at_k(retrieved_texts: list[str], relevant_kws: list[str]) -> float:
    """Fraction of retrieved memories that contain at least one relevant keyword."""
    if not retrieved_texts:
        return 0.0
    hits = sum(
        1 for t in retrieved_texts
        if any(kw.lower() in t.lower() for kw in relevant_kws)
    )
    return hits / len(retrieved_texts)


def recall_at_k(retrieved_texts: list[str], relevant_kws: list[str]) -> float:
    """Fraction of relevant keywords covered by at least one retrieved memory."""
    if not relevant_kws:
        return 1.0
    covered = sum(
        1 for kw in relevant_kws
        if any(kw.lower() in t.lower() for t in retrieved_texts)
    )
    return covered / len(relevant_kws)


# ---------------------------------------------------------------------------
# SCENARIO A — WITHOUT mem0
# ---------------------------------------------------------------------------
def run_without_mem0() -> list[QueryResult]:
    _sep("SCENARIO A — WITHOUT mem0  (full conversation history in every prompt)")

    history_tokens = tok(ALL_HISTORY)
    system_tokens  = tok(SYSTEM_PROMPT)
    print(f"\n  Full history (all users): {history_tokens} tokens")
    print(f"  System prompt           : {system_tokens} tokens")
    print(f"  Both are pasted into EVERY LLM call regardless of the query.\n")

    results = []
    for q in BENCHMARK_QUERIES:
        # For user-scoped queries, only that user's history is passed
        if q["user_id"]:
            hist   = _flat_history(q["user_id"])
            h_tok  = tok(hist)
        else:
            hist   = ALL_HISTORY
            h_tok  = history_tokens

        qt    = tok(q["text"])
        total = system_tokens + h_tok + qt

        t0 = time.perf_counter()
        time.sleep(0.001)   # simulate scheduling overhead
        latency_ms = (time.perf_counter() - t0) * 1000

        r = QueryResult(
            query_id=q["id"],
            complexity=q["complexity"],
            query_text=q["text"],
            description=q["description"],
            without_input_tokens=total,
            without_latency_ms=latency_ms,
        )
        results.append(r)
        scope = q["user_id"] or "ALL"
        print(f"  [{q['id']:>2}] {q['complexity']:<7}  scope={scope:<8}  "
              f"tokens={total:>4}  (sys={system_tokens}+hist={h_tok}+q={qt})  "
              f"'{q['text'][:45]}...'")

    return results


# ---------------------------------------------------------------------------
# SCENARIO B — WITH mem0
# ---------------------------------------------------------------------------
def run_with_mem0(results: list[QueryResult]) -> dict:
    _sep("SCENARIO B — WITH mem0  (Memory.add + Db2VectorStore + OllamaLLM)")

    print(f"\n  Pipeline per Memory.add():")
    print(f"    conversation")
    print(f"      → mem0/llms/ollama.py         OllamaLLM(llama3.2)  fact extraction")
    print(f"      → mem0/embeddings/ollama.py   OllamaEmbedding(nomic-embed-text 768d)")
    print(f"      → mem0/vector_stores/db2.py   Db2VectorStore  IBM Db2 COSINE search")
    print(f"      → mem0/utils/factory.py        provider='db2' registered at line 207\n")

    # ── Init ──────────────────────────────────────────────────────────────
    print("  Initialising Memory.from_config() ...")
    m = Memory.from_config(MEM0_CONFIG)
    print(f"  ✅  LLM         : {m.llm.__class__.__module__}.{m.llm.__class__.__name__}")
    print(f"  ✅  Embedder    : {m.embedding_model.__class__.__module__}.{m.embedding_model.__class__.__name__}")
    print(f"  ✅  VectorStore : {m.vector_store.__class__.__module__}.{m.vector_store.__class__.__name__}")
    print(f"  ✅  Table       : {m.vector_store.config.collection_name}")

    # Reset for clean benchmark run
    m.vector_store.reset()
    print("  ✅  Table reset — clean slate\n")

    # ── INGESTION ─────────────────────────────────────────────────────────
    _subsep("Ingestion — Memory.add() per user  (user-role messages only)")
    ingestion_tokens = 0
    ingestion_start  = time.perf_counter()
    total_memories   = 0

    for user_id, user_data in USERS.items():
        print(f"\n  user_id={user_id}")
        # Merge all conversations for this user and keep ONLY user-role turns.
        # Assistant turns are LLM context, not facts about the user — passing
        # them causes llama3.2 to store verbatim assistant lines as memories
        # (e.g. "Cloud-native stack! Any favourite tools?").
        # Merging both conversations into one call also prevents the second
        # conversation's facts from being false-deduped against the first.
        all_msgs  = [msg for conv in user_data["conversations"] for msg in conv]
        user_msgs = [msg for msg in all_msgs if msg["role"] == "user"]
        c_tok     = tok(" ".join(msg["content"] for msg in user_msgs))
        ingestion_tokens += c_tok

        t0     = time.perf_counter()
        result = m.add(user_msgs, user_id=user_id)
        lat    = (time.perf_counter() - t0) * 1000

        added = result.get("results", []) if isinstance(result, dict) else result
        total_memories += len(added)
        print(f"    {len(user_msgs)} user turns  tokens={c_tok:>3}  {lat:>6.0f}ms  "
              f"→ {len(added)} fact(s) extracted")
        for mem in added:
            mem_text = mem.get("memory", mem.get("text", ""))
            event    = mem.get("event", "ADD")
            print(f"      [{event:>6}]  \"{mem_text[:70]}\"")

    ingestion_latency_ms = (time.perf_counter() - ingestion_start) * 1000
    print(f"\n  Ingestion complete:")
    print(f"    Users ingested         : {len(USERS)}")
    print(f"    Facts stored in Db2    : {total_memories}")
    print(f"    Input tokens consumed  : {ingestion_tokens}")
    print(f"    Total time             : {ingestion_latency_ms:.0f} ms")

    # ── RETRIEVAL ─────────────────────────────────────────────────────────
    _subsep("Retrieval — Memory.search() per query  (top-3 injected into prompt)")
    system_tokens = tok(SYSTEM_PROMPT)

    for q, r in zip(BENCHMARK_QUERIES, results):
        qt = tok(q["text"])

        t0 = time.perf_counter()
        if q["user_id"]:
            # User-scoped search via Memory.search() — standard path
            search_result = m.search(query=q["text"], top_k=3, filters={"user_id": q["user_id"]})
            hits = search_result.get("results", []) if isinstance(search_result, dict) else search_result
        else:
            # Cross-user search — Memory.search() requires a user_id filter, so
            # search each user separately via Db2VectorStore directly and merge.
            query_vec = m.embedding_model.embed(q["text"])
            raw_hits  = m.vector_store.search(query=q["text"], vectors=query_vec, top_k=3)
            hits = [
                {"memory": h.payload.get("data", ""), "score": h.score, "id": h.id}
                for h in raw_hits
            ]
        lat = (time.perf_counter() - t0) * 1000
        retrieved_texts = [h.get("memory", h.get("text", "")) for h in hits]

        context_text   = "\n".join(f"- {t}" for t in retrieved_texts)
        context_tokens = tok(context_text)
        total_tokens   = system_tokens + context_tokens + qt

        p3 = precision_at_k(retrieved_texts, q["relevant"])
        r3 = recall_at_k(retrieved_texts, q["relevant"])

        r.with_retrieval_tokens = total_tokens
        r.with_latency_ms       = lat
        r.retrieved_memories    = retrieved_texts
        r.precision_at_3        = p3
        r.recall_at_3           = r3
        r.top_retrieved_text    = retrieved_texts[0] if retrieved_texts else ""
        r.top_retrieved_score   = hits[0].get("score", 0.0) if hits else 0.0

        saved = r.without_input_tokens - total_tokens
        pct   = saved / r.without_input_tokens * 100 if r.without_input_tokens else 0
        scope = q["user_id"] or "ALL"
        print(f"\n  [{q['id']:>2}] {q['complexity']:<7}  scope={scope:<8}  "
              f"'{q['text'][:50]}...'")
        print(f"       tokens: without={r.without_input_tokens:>4}  "
              f"with={total_tokens:>4}  saved={saved:>4} ({pct:.0f}%)")
        print(f"       latency={lat:.0f}ms  P@3={p3:.2f}  R@3={r3:.2f}")
        if retrieved_texts:
            print(f"       top-1 (score={r.top_retrieved_score:.3f}): "
                  f"\"{r.top_retrieved_text[:65]}\"")
        else:
            print(f"       ⚠️  No memories retrieved")

    return {
        "ingestion_tokens":    ingestion_tokens,
        "ingestion_latency_ms": round(ingestion_latency_ms, 1),
        "total_memories":      total_memories,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[QueryResult], ingestion: dict) -> dict:
    _sep("BENCHMARK SUMMARY")

    print(f"\n  {'ID':<4} {'Level':<8} {'Without':>8} {'With':>8} "
          f"{'Saved':>7} {'%':>6} {'P@3':>5} {'R@3':>5}  Description")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*6} {'─'*5} {'─'*5}  {'─'*38}")

    by_level: dict[str, list[QueryResult]] = {"BASIC": [], "MEDIUM": [], "COMPLEX": []}
    for r in results:
        w  = r.without_input_tokens
        m  = r.with_retrieval_tokens
        sv = w - m
        pc = sv / w * 100 if w else 0
        print(f"  {r.query_id:<4} {r.complexity:<8} {w:>8} {m:>8} "
              f"{sv:>7} {pc:>5.1f}% {r.precision_at_3:>5.2f} {r.recall_at_3:>5.2f}  "
              f"{r.description[:38]}")
        by_level[r.complexity].append(r)

    print(f"\n  {'─'*92}")
    print(f"  Averages by complexity:\n")
    level_stats = {}
    for level, lr in by_level.items():
        if not lr:
            continue
        aw  = sum(x.without_input_tokens for x in lr) / len(lr)
        am  = sum(x.with_retrieval_tokens for x in lr) / len(lr)
        asv = aw - am
        apc = asv / aw * 100 if aw else 0
        ap  = sum(x.precision_at_3 for x in lr) / len(lr)
        ar  = sum(x.recall_at_3 for x in lr) / len(lr)
        print(f"  {level:<8}  avg_without={aw:.0f}  avg_with={am:.0f}  "
              f"saved={asv:.0f} ({apc:.1f}%)  P@3={ap:.2f}  R@3={ar:.2f}")
        level_stats[level] = {
            "avg_without": round(aw, 1), "avg_with": round(am, 1),
            "avg_saved": round(asv, 1),  "avg_saved_pct": round(apc, 1),
            "avg_precision": round(ap, 2), "avg_recall": round(ar, 2),
        }

    total_w  = sum(r.without_input_tokens for r in results)
    total_m  = sum(r.with_retrieval_tokens for r in results)
    total_sv = total_w - total_m
    total_pc = total_sv / total_w * 100 if total_w else 0
    avg_p    = sum(r.precision_at_3 for r in results) / len(results)
    avg_r    = sum(r.recall_at_3 for r in results) / len(results)

    print(f"\n  {'═'*92}")
    print(f"  GRAND TOTAL  without={total_w}  with={total_m}  "
          f"saved={total_sv} ({total_pc:.1f}%)")
    print(f"  avg P@3={avg_p:.2f}  avg R@3={avg_r:.2f}")
    print(f"  Ingestion (one-time): {ingestion['ingestion_tokens']} tokens  "
          f"in {ingestion['ingestion_latency_ms']:.0f}ms  "
          f"→ {ingestion['total_memories']} facts stored in Db2")
    print(f"  {'═'*92}")

    return {
        "total_without":    total_w,
        "total_with":       total_m,
        "total_saved":      total_sv,
        "total_saved_pct":  round(total_pc, 1),
        "avg_precision":    round(avg_p, 2),
        "avg_recall":       round(avg_r, 2),
        "by_level":         level_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not CONNECTION_PARAMS["password"]:
        log.error("DB2_PASSWORD not set.")
        sys.exit(1)

    n_users = len(USERS)
    n_convs = sum(len(u["conversations"]) for u in USERS.values())
    n_q     = len(BENCHMARK_QUERIES)
    print(f"\n  mem0 + IBM Db2 — Benchmark Report")
    print(f"  {'─'*62}")
    print(f"  Users       : {n_users}  ({n_convs} conversations total)")
    print(f"  Queries     : {n_q}  "
          f"(BASIC={sum(1 for q in BENCHMARK_QUERIES if q['complexity']=='BASIC')}  "
          f"MEDIUM={sum(1 for q in BENCHMARK_QUERIES if q['complexity']=='MEDIUM')}  "
          f"COMPLEX={sum(1 for q in BENCHMARK_QUERIES if q['complexity']=='COMPLEX')})")
    print(f"  Tokenizer   : tiktoken cl100k_base (GPT-4 encoding)")
    print(f"  LLM         : llama3.2 via mem0/llms/ollama.py")
    print(f"  Embedder    : nomic-embed-text (768d) via mem0/embeddings/ollama.py")
    print(f"  VectorStore : Db2VectorStore via mem0/utils/factory.py  provider='db2'")

    without_results = run_without_mem0()
    ingestion_stats = run_with_mem0(without_results)
    grand           = print_summary(without_results, ingestion_stats)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = Path(__file__).parent / "db2_benchmark_results.json"
    raw = {
        "meta": {
            "users":          n_users,
            "conversations":  n_convs,
            "queries":        n_q,
            "llm":            "llama3.2",
            "embedder":       "nomic-embed-text",
            "vector_store":   "Db2VectorStore (IBM Db2 COSINE)",
            "tokenizer":      "tiktoken cl100k_base",
        },
        "ingestion": ingestion_stats,
        "grand_totals": grand,
        "results": [
            {
                "id":             r.query_id,
                "complexity":     r.complexity,
                "text":           r.query_text,
                "description":    r.description,
                "without_tokens": r.without_input_tokens,
                "with_tokens":    r.with_retrieval_tokens,
                "saved_tokens":   r.without_input_tokens - r.with_retrieval_tokens,
                "saved_pct":      round(
                    (r.without_input_tokens - r.with_retrieval_tokens)
                    / r.without_input_tokens * 100, 1
                ) if r.without_input_tokens else 0,
                "latency_ms":     round(r.with_latency_ms, 1),
                "precision_at_3": round(r.precision_at_3, 2),
                "recall_at_3":    round(r.recall_at_3, 2),
                "retrieved":      r.retrieved_memories,
                "top_score":      round(r.top_retrieved_score, 4),
            }
            for r in without_results
        ],
    }
    with open(out, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  Results saved → {out}")
    print(f"  Run the report generator next to produce the HTML report.")


if __name__ == "__main__":
    main()
