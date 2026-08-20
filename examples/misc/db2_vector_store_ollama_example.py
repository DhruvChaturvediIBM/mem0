"""IBM Db2 vector store example — real embeddings via Ollama nomic-embed-text.

Loads records from db2_vector_data.json, generates 768-dim embeddings using
the local Ollama nomic-embed-text model, stores them in Db2, then runs
similarity search, filtered search, list, get, and update.

Prerequisites:
    1. Ollama running locally:       ollama serve
    2. nomic-embed-text pulled:      ollama pull nomic-embed-text
    3. Db2 instance reachable (container or cloud)

Set Db2 credentials via env vars or examples/misc/.env:

    export DB2_DATABASE=TESTDB
    export DB2_HOST=127.0.0.1
    export DB2_PORT=50000
    export DB2_USERNAME=db2inst1
    export DB2_PASSWORD=<your-password>

Run:
    DB2_PASSWORD=pass python3.11 examples/misc/db2_vector_store_ollama_example.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("db2_ollama")

logging.getLogger("ibm_db_dbi").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Optional .env loader
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)
        log.info("Loaded env from %s", _ENV_FILE)
    except ImportError:
        log.warning("python-dotenv not installed — skipping .env load")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONNECTION_PARAMS = {
    "database": os.environ.get("DB2_DATABASE", "TESTDB"),
    "host":     os.environ.get("DB2_HOST", "127.0.0.1"),
    "port":     int(os.environ.get("DB2_PORT", "50000")),
    "username": os.environ.get("DB2_USERNAME", ""),
    "password": os.environ.get("DB2_PASSWORD", ""),
}

TABLE_NAME     = "MEM0_NOMIC_EXAMPLE"
EMBEDDING_DIM  = 768
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = "nomic-embed-text"
DATA_FILE      = Path(__file__).parent / "db2_vector_data.json"


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """Call Ollama /api/embeddings and return the 768-dim vector."""
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    vector = result["embedding"]
    log.debug("embed('%s...') → dim=%d  first3=%s", text[:40], len(vector), vector[:3])
    return vector


# ---------------------------------------------------------------------------
# Separator helper
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Starting Db2 + Ollama nomic-embed-text example")
    log.debug(
        "Connection → host=%s  port=%s  database=%s  username=%s",
        CONNECTION_PARAMS["host"],
        CONNECTION_PARAMS["port"],
        CONNECTION_PARAMS["database"],
        CONNECTION_PARAMS["username"],
    )

    if not CONNECTION_PARAMS["password"]:
        log.error("DB2_PASSWORD is not set. Run: export DB2_PASSWORD=<password>")
        sys.exit(1)

    # ── 0. Verify Ollama is reachable ─────────────────────────────────────
    _sep("0. Verifying Ollama connection")
    log.info("Pinging Ollama at %s …", OLLAMA_URL)
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        model_names = [m["name"] for m in tags.get("models", [])]
        log.info("Ollama models available: %s", model_names)
        if not any(OLLAMA_MODEL in n for n in model_names):
            log.error("Model '%s' not found. Run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)
            sys.exit(1)
        print(f"Ollama OK.  Model '{OLLAMA_MODEL}' is available.")
    except Exception as exc:
        log.error("Cannot reach Ollama at %s: %s", OLLAMA_URL, exc)
        sys.exit(1)

    # ── 1. Load data from JSON ─────────────────────────────────────────────
    _sep("1. Loading data from JSON file")
    log.info("Reading %s …", DATA_FILE)
    with open(DATA_FILE) as f:
        records = json.load(f)
    log.info("Loaded %d records from %s", len(records), DATA_FILE.name)
    print(f"Loaded {len(records)} records from {DATA_FILE.name}")
    for i, r in enumerate(records):
        print(f"  [{i:02d}] id={r['id']}  user_id={r['user_id']}  category={r['category']}  text='{r['text'][:60]}...'")
        log.debug("  record[%d]: %s", i, r)

    # ── 2. Connect to Db2 ─────────────────────────────────────────────────
    _sep("2. Connecting to Db2")
    from mem0.vector_stores.db2 import Db2VectorStore
    log.info("Connecting to Db2 at %s:%s …", CONNECTION_PARAMS["host"], CONNECTION_PARAMS["port"])
    store = Db2VectorStore(
        connection_params=CONNECTION_PARAMS,
        collection_name=TABLE_NAME,
        embedding_model_dims=EMBEDDING_DIM,
        distance_strategy="COSINE",
    )
    log.info("Connected.  Table=%s  Dim=%d  Strategy=COSINE", TABLE_NAME, EMBEDDING_DIM)
    print(f"Connected.  Table={TABLE_NAME}  Dim={EMBEDDING_DIM}  Strategy=COSINE")

    # ── 3. Generate embeddings ─────────────────────────────────────────────
    _sep("3. Generating embeddings via nomic-embed-text (768-dim)")
    log.info("Embedding %d texts via Ollama …", len(records))
    vectors  = []
    payloads = []
    ids      = []
    for i, rec in enumerate(records):
        log.info("  [%d/%d] Embedding: '%s'", i + 1, len(records), rec["text"][:60])
        vec = embed(rec["text"])
        vectors.append(vec)
        payloads.append({
            "data":     rec["text"],
            "user_id":  rec["user_id"],
            "category": rec["category"],
            "tags":     ", ".join(rec.get("tags", [])),
        })
        ids.append(rec["id"])
        print(f"  [{i:02d}] ✓  id={rec['id']}  dim={len(vec)}  first3={[round(v, 4) for v in vec[:3]]}")
    log.info("All %d embeddings generated", len(vectors))

    # ── 3b. Reset table so re-runs don't hit duplicate-key errors ─────────
    _sep("3b. Resetting table (drop + recreate for a clean run)")
    log.info("Resetting table %s to avoid duplicate-key errors on re-run …", TABLE_NAME)
    store.reset()
    log.info("Table reset complete")
    print(f"Table {TABLE_NAME} reset — ready for fresh insert.")

    # ── 4. Insert into Db2 ────────────────────────────────────────────────
    _sep("4. Inserting vectors into Db2")
    log.info("Inserting %d vectors into table %s …", len(vectors), TABLE_NAME)
    stored_ids = store.insert(vectors=vectors, payloads=payloads, ids=ids)
    log.info("Insert complete — %d records stored", len(stored_ids))
    print(f"Inserted {len(stored_ids)} records:")
    for hid, p in zip(stored_ids, payloads):
        print(f"  hashed_id={hid}  user_id={p['user_id']}  text='{p['data'][:55]}...'")
        log.debug("  stored: hashed_id=%s  payload=%s", hid, p)

    # ── 5. Similarity search ──────────────────────────────────────────────
    _sep("5. Similarity search — query: 'animals and pets'")
    query = "animals and pets"
    log.info("Embedding query: '%s' …", query)
    query_vec = embed(query)
    log.info("Searching top-5 nearest …")
    results = store.search(query=query, vectors=[query_vec], top_k=5)
    log.info("Search returned %d result(s)", len(results))
    print(f"Query: '{query}'  →  Top-{len(results)} results:")
    for r in results:
        print(
            f"  score={r.score:.4f}"
            f"  user_id={r.payload.get('user_id'):<8}"
            f"  category={r.payload.get('category'):<10}"
            f"  text='{r.payload.get('data', '')[:55]}...'"
        )
        log.debug("  result: id=%s  score=%.4f  payload=%s", r.id, r.score, r.payload)

    # ── 6. Similarity search ──────────────────────────────────────────────
    _sep("6. Similarity search — query: 'machine learning and AI'")
    query2 = "machine learning and AI"
    log.info("Embedding query: '%s' …", query2)
    query_vec2 = embed(query2)
    results2 = store.search(query=query2, vectors=[query_vec2], top_k=5)
    log.info("Search returned %d result(s)", len(results2))
    print(f"Query: '{query2}'  →  Top-{len(results2)} results:")
    for r in results2:
        print(
            f"  score={r.score:.4f}"
            f"  user_id={r.payload.get('user_id'):<8}"
            f"  category={r.payload.get('category'):<10}"
            f"  text='{r.payload.get('data', '')[:55]}...'"
        )
        log.debug("  result: id=%s  score=%.4f  payload=%s", r.id, r.score, r.payload)

    # ── 7. Filtered search ────────────────────────────────────────────────
    _sep("7. Filtered search — category=tech, query: 'programming languages'")
    query3 = "programming languages"
    log.info("Embedding query: '%s' …", query3)
    query_vec3 = embed(query3)
    results3 = store.search(query=query3, vectors=[query_vec3], top_k=5, filters={"category": "tech"})
    log.info("Filtered search returned %d result(s)", len(results3))
    print(f"Query: '{query3}'  filter: category=tech  →  {len(results3)} results:")
    for r in results3:
        print(
            f"  score={r.score:.4f}"
            f"  user_id={r.payload.get('user_id'):<8}"
            f"  text='{r.payload.get('data', '')[:55]}...'"
        )
        log.debug("  result: id=%s  score=%.4f  payload=%s", r.id, r.score, r.payload)

    # ── 8. List by category ───────────────────────────────────────────────
    _sep("8. List all records — category=food")
    log.info("Listing all food records …")
    food_records = store.list(filters={"category": "food"}, top_k=10)[0]
    log.info("List returned %d food record(s)", len(food_records))
    print(f"Food records ({len(food_records)}):")
    for r in food_records:
        print(f"  id={r.id}  user_id={r.payload.get('user_id')}  text='{r.payload.get('data', '')[:60]}...'")
        log.debug("  food record: id=%s  payload=%s", r.id, r.payload)

    # ── 9. Get by ID ──────────────────────────────────────────────────────
    _sep("9. Get record by ID — doc_001 (Alice)")
    first_hid = stored_ids[0]
    log.info("Fetching record hashed_id=%s …", first_hid)
    record = store.get(first_hid)
    if record:
        print(f"Found:  hashed_id={record.id}")
        print(f"        user_id  = {record.payload.get('user_id')}")
        print(f"        category = {record.payload.get('category')}")
        print(f"        tags     = {record.payload.get('tags')}")
        print(f"        text     = '{record.payload.get('data', '')}'")
        log.info("Record found: %s", record.payload)
    else:
        print("Record not found.")
        log.warning("Record hashed_id=%s not found", first_hid)

    # ── 10. Update ────────────────────────────────────────────────────────
    _sep("10. Update record — Alice gets a second dog")
    updated_text = "Alice has two golden retrievers, Max and Bella, who both love fetch."
    log.info("Re-embedding updated text: '%s' …", updated_text)
    updated_vec = embed(updated_text)
    new_payload = {
        "data":     updated_text,
        "user_id":  "alice",
        "category": "pets",
        "tags":     "dog, golden retriever, outdoor, updated",
    }
    log.info("Updating hashed_id=%s …", first_hid)
    store.update(vector_id=first_hid, vector=updated_vec, payload=new_payload)
    updated_rec = store.get(first_hid)
    if updated_rec:
        print(f"Updated record:")
        print(f"  hashed_id = {updated_rec.id}")
        print(f"  text      = '{updated_rec.payload.get('data', '')}'")
        print(f"  tags      = {updated_rec.payload.get('tags')}")
        log.info("Update confirmed: %s", updated_rec.payload)

    # ── 11. Collection info ───────────────────────────────────────────────
    _sep("11. Collection info")
    log.info("Fetching collection info …")
    info = store.col_info()
    print(f"Table info: {info}")
    log.info("col_info: %s", info)
    all_tables = store.list_cols()
    mem0_tables = [t for t in all_tables if "MEM0" in t.upper()]
    print(f"All MEM0 tables in schema: {mem0_tables}")
    log.debug("All tables in schema: %s", all_tables)

    _sep("Done")
    print("Example completed successfully.")
    log.info("Example finished successfully")


if __name__ == "__main__":
    main()
