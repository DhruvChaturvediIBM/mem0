"""IBM Db2 vector store example.

Runs insert, search, filter, update, delete, and reset against a live Db2
instance using hand-crafted vectors — no LLM required.

Set connection details in examples/misc/.env (copied from .env.example) or
export the variables manually before running:

    export DB2_DATABASE=TESTDB
    export DB2_HOST=...
    export DB2_PORT=50000
    export DB2_USERNAME=...
    export DB2_PASSWORD=...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)
        print(f"Loaded env from {_ENV_FILE}")
    except ImportError:
        pass

CONNECTION_PARAMS = {
    "database": os.environ.get("DB2_DATABASE", "TESTDB"),
    "host": os.environ.get("DB2_HOST", ""),
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
    if not CONNECTION_PARAMS["password"]:
        print(
            "ERROR: DB2_PASSWORD is not set.\n"
            "Run:  export DB2_PASSWORD=<your-password>",
            file=sys.stderr,
        )
        sys.exit(1)

    from mem0.vector_stores.db2 import Db2VectorStore

    _sep("1. Connecting to Db2 and creating the vector table")
    store = Db2VectorStore(
        connection_params=CONNECTION_PARAMS,
        collection_name=TABLE_NAME,
        embedding_model_dims=EMBEDDING_DIM,
        distance_strategy="EUCLIDEAN",
    )
    print(f"Connected.  Table: {TABLE_NAME}  Dim: {EMBEDDING_DIM}")

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
    ids = store.insert(vectors=vectors, payloads=payloads)
    print(f"Inserted {len(ids)} records")
    for hid, p in zip(ids, payloads):
        print(f"  id={hid}  data='{p['data']}'")

    _sep("3. Similarity search — closest to first vector")
    query_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = store.search(query="", vectors=[query_vector], top_k=3)
    print(f"Top-{len(results)} results:")
    for r in results:
        print(f"  id={r.id}  score={r.score:.4f}  data='{r.payload.get('data', '')}'")

    _sep("4. Search with filter  user_id=alice")
    results_filtered = store.search(
        query="",
        vectors=[query_vector],
        top_k=5,
        filters={"user_id": "alice"},
    )
    print(f"Filtered results ({len(results_filtered)} found):")
    for r in results_filtered:
        print(f"  id={r.id}  user_id={r.payload.get('user_id')}  data='{r.payload.get('data', '')}'")

    _sep("5. List all  category=tech")
    tech_records = store.list(filters={"category": "tech"}, top_k=10)[0]
    print(f"Tech records ({len(tech_records)}):")
    for r in tech_records:
        print(f"  id={r.id}  data='{r.payload.get('data', '')}'")

    _sep("6. Get record by ID")
    first_id = ids[0]
    record = store.get(first_id)
    if record:
        print(f"Found:  id={record.id}  payload={record.payload}")
    else:
        print("Record not found.")

    _sep("7. Update record")
    store.update(
        vector_id=first_id,
        payload={"data": "Alice loves cats AND dogs now", "user_id": "alice", "category": "pets", "updated": True},
    )
    updated = store.get(first_id)
    if updated:
        print(f"Updated payload:  {updated.payload}")

    _sep("8. Delete last record")
    last_id = ids[-1]
    store.delete(vector_id=last_id)
    gone = store.get(last_id)
    print(f"After delete, get({last_id}) → {'None ✓' if gone is None else gone.payload}")

    _sep("9. Collection info")
    info = store.col_info()
    print(f"Table info: {info}")
    cols = store.list_cols()
    db2_tables = [t for t in cols if "MEM0" in t.upper()]
    print(f"MEM0 tables in schema: {db2_tables}")

    _sep("10. Reset — drop and recreate the table")
    store.reset()
    after_reset = store.list(top_k=10)[0]
    print(f"Records after reset: {len(after_reset)} (expected 0)")

    _sep("Done")
    print("Example completed successfully.")


if __name__ == "__main__":
    main()
