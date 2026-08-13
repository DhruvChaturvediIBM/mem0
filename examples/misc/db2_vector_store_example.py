"""IBM Db2 vector store example.

Runs insert, search, filter, update, delete, and reset against a live Db2
instance using hand-crafted vectors — no LLM required.

Set connection details in examples/misc/.env (copied from .env.example) or
export the variables manually before running:

    export DB2_DATABASE=TESTDB
    export DB2_HOST=127.0.0.1
    export DB2_PORT=50000
    export DB2_USERNAME=db2inst1
    export DB2_PASSWORD=<your-password>

Db2 Community Edition (Podman/Docker) — quickstart:

    podman run -itd --name db2server \\
      -e DB2INST1_PASSWORD=pass -e DBNAME=TESTDB -e LICENSE=accept \\
      -p 50000:50000 icr.io/db2_community/db2
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — shows timestamps + level for every log line
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("db2_example")

# Silence noisy third-party loggers
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

CONNECTION_PARAMS = {
    "database": os.environ.get("DB2_DATABASE", "TESTDB"),
    "host": os.environ.get("DB2_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB2_PORT", "50000")),
    "username": os.environ.get("DB2_USERNAME", ""),
    "password": os.environ.get("DB2_PASSWORD", ""),
}

TABLE_NAME = "MEM0_EXAMPLE"
EMBEDDING_DIM = 8


def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def main() -> None:
    log.info("Starting Db2 vector store example")
    log.debug(
        "Connection → host=%s  port=%s  database=%s  username=%s",
        CONNECTION_PARAMS["host"],
        CONNECTION_PARAMS["port"],
        CONNECTION_PARAMS["database"],
        CONNECTION_PARAMS["username"],
    )

    if not CONNECTION_PARAMS["password"]:
        log.error("DB2_PASSWORD is not set. Run: export DB2_PASSWORD=<your-password>")
        sys.exit(1)

    from mem0.vector_stores.db2 import Db2VectorStore

    # ── 1. Connect ────────────────────────────────────────────────────────
    _sep("1. Connecting to Db2 and creating the vector table")
    log.info("Connecting to Db2 at %s:%s …", CONNECTION_PARAMS["host"], CONNECTION_PARAMS["port"])
    store = Db2VectorStore(
        connection_params=CONNECTION_PARAMS,
        collection_name=TABLE_NAME,
        embedding_model_dims=EMBEDDING_DIM,
        distance_strategy="EUCLIDEAN",
    )
    log.info("Connected successfully.  Table=%s  Dim=%s", TABLE_NAME, EMBEDDING_DIM)
    print(f"Connected.  Table: {TABLE_NAME}  Dim: {EMBEDDING_DIM}")

    # ── 2. Insert ─────────────────────────────────────────────────────────
    _sep("2. Inserting vectors")
    vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    payloads = [
        {"data": "Alice loves cats", "user_id": "alice", "category": "pets"},
        {"data": "Bob has two dogs", "user_id": "bob", "category": "pets"},
        {"data": "Charlie writes Python", "user_id": "charlie", "category": "tech"},
        {"data": "Diana prefers Java", "user_id": "diana", "category": "tech"},
    ]
    log.info("Inserting %d vectors into table %s …", len(vectors), TABLE_NAME)
    ids = store.insert(vectors=vectors, payloads=payloads)
    log.info("Insert complete — %d records stored", len(ids))
    print(f"Inserted {len(ids)} records:")
    for i, (hid, vec, p) in enumerate(zip(ids, vectors, payloads)):
        print(
            f"  [{i}] id={hid}"
            f"  data='{p['data']}'"
            f"  user_id={p['user_id']}"
            f"  category={p['category']}"
            f"  vector={vec}"
        )
        log.debug("  record[%d]: id=%s  payload=%s  vector=%s", i, hid, p, vec)

    # ── 3. Similarity search ──────────────────────────────────────────────
    _sep("3. Similarity search — closest to first vector")
    query_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    log.info("Searching top-3 nearest to query_vector=%s", query_vector)
    results = store.search(query="", vectors=[query_vector], top_k=3)
    log.info("Search returned %d result(s)", len(results))
    print(f"Top-{len(results)} results:")
    for r in results:
        print(
            f"  id={r.id}"
            f"  score={r.score:.4f}"
            f"  data='{r.payload.get('data', '')}'"
            f"  user_id={r.payload.get('user_id')}"
        )
        log.debug("  result: id=%s  score=%.4f  payload=%s", r.id, r.score, r.payload)

    # ── 4. Filtered search ────────────────────────────────────────────────
    _sep("4. Search with filter  user_id=alice")
    log.info("Searching with filter user_id=alice …")
    results_filtered = store.search(
        query="",
        vectors=[query_vector],
        top_k=5,
        filters={"user_id": "alice"},
    )
    log.info("Filtered search returned %d result(s)", len(results_filtered))
    print(f"Filtered results ({len(results_filtered)} found):")
    for r in results_filtered:
        print(
            f"  id={r.id}"
            f"  score={r.score:.4f}"
            f"  user_id={r.payload.get('user_id')}"
            f"  data='{r.payload.get('data', '')}'"
        )
        log.debug("  filtered result: id=%s  payload=%s", r.id, r.payload)

    # ── 5. List with filter ───────────────────────────────────────────────
    _sep("5. List all  category=tech")
    log.info("Listing all records with category=tech …")
    tech_records = store.list(filters={"category": "tech"}, top_k=10)[0]
    log.info("List returned %d tech record(s)", len(tech_records))
    print(f"Tech records ({len(tech_records)}):")
    for r in tech_records:
        print(f"  id={r.id}  data='{r.payload.get('data', '')}' user_id={r.payload.get('user_id')}")
        log.debug("  tech record: id=%s  payload=%s", r.id, r.payload)

    # ── 6. Get by ID ──────────────────────────────────────────────────────
    _sep("6. Get record by ID")
    first_id = ids[0]
    log.info("Fetching record id=%s …", first_id)
    record = store.get(first_id)
    if record:
        print(f"Found:  id={record.id}")
        print(f"        payload={record.payload}")
        log.info("Record found: %s", record.payload)
    else:
        print("Record not found.")
        log.warning("Record id=%s not found", first_id)

    # ── 7. Update ─────────────────────────────────────────────────────────
    _sep("7. Update record")
    new_payload = {"data": "Alice loves cats AND dogs now", "user_id": "alice", "category": "pets", "updated": True}
    log.info("Updating id=%s with new payload=%s …", first_id, new_payload)
    store.update(vector_id=first_id, payload=new_payload)
    updated = store.get(first_id)
    if updated:
        print(f"Updated payload:  {updated.payload}")
        log.info("Update confirmed: %s", updated.payload)

    # ── 8. Delete (skipped) ───────────────────────────────────────────────
    _sep("8. Delete last record (skipped)")
    last_id = ids[-1]
    log.info("Skipping delete for record id=%s", last_id)
    print(f"Delete skipped — record id={last_id} kept as-is.")

    # ── 9. Collection info ────────────────────────────────────────────────
    _sep("9. Collection info")
    log.info("Fetching collection info for table %s …", TABLE_NAME)
    info = store.col_info()
    print(f"Table info: {info}")
    log.info("col_info: %s", info)
    cols = store.list_cols()
    log.debug("All tables in schema: %s", cols)
    db2_tables = [t for t in cols if "MEM0" in t.upper()]
    print(f"MEM0 tables in schema: {db2_tables}")

    # # ── 10. Reset ─────────────────────────────────────────────────────────
    # _sep("10. Reset — drop and recreate the table")
    # log.info("Resetting table %s (drop + recreate) …", TABLE_NAME)
    # store.reset()
    # after_reset = store.list(top_k=10)[0]
    # print(f"Records after reset: {len(after_reset)} (expected 0)")
    # log.info("Reset complete — %d records remain (expected 0)", len(after_reset))

    # _sep("Done")
    # print("Example completed successfully.")
    # log.info("Example finished successfully")


if __name__ == "__main__":
    main()
