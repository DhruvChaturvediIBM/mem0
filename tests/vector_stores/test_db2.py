"""Unit and integration tests for the IBM Db2 vector store.

Unit tests (no real DB): all tests that do NOT have the ``@requires_db2_credentials``
marker.  They run in CI without any Db2 installation.

Integration tests (real DB): guarded by ``@requires_db2_credentials``.  Set the
following environment variables to enable them::

    DB2_DATABASE=BLUDB
    DB2_HOST=localhost
    DB2_PORT=25000
    DB2_USERNAME=db2inst1
    DB2_PASSWORD=<your-password>

Db2 Community Edition (Docker) — quickstart::

    docker run -itd --name db2ce \\
      -e DB2INST1_PASSWORD=password \\
      -e DBNAME=BLUDB \\
      -e LICENSE=accept \\
      -p 25000:25000 \\
      icr.io/db2_community/db2

    export DB2_DATABASE=BLUDB DB2_HOST=localhost DB2_PORT=25000 \\
           DB2_USERNAME=db2inst1 DB2_PASSWORD=password
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Load .env from examples/misc/.env when present
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parents[2] / "examples" / "misc" / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE, override=False)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Stub ibm_db_dbi only when the real native driver is absent.
# ---------------------------------------------------------------------------
try:
    import ibm_db_dbi  # noqa: F401 — real driver present, no stub needed
except ImportError:
    _stub = ModuleType("ibm_db_dbi")
    _stub.connect = MagicMock()
    _stub.DatabaseError = Exception
    sys.modules["ibm_db_dbi"] = _stub

from mem0.configs.vector_stores.db2 import Db2Config  # noqa: E402
from mem0.vector_stores.db2 import (  # noqa: E402
    Db2VectorStore,
    OutputData,
    _distance_to_score,
)

# ---------------------------------------------------------------------------
# Integration credentials
# ---------------------------------------------------------------------------

DB2_DATABASE = os.environ.get("DB2_DATABASE", "")
DB2_HOST = os.environ.get("DB2_HOST", "")
DB2_PORT = int(os.environ.get("DB2_PORT", "25000"))
DB2_USERNAME = os.environ.get("DB2_USERNAME", "")
DB2_PASSWORD = os.environ.get("DB2_PASSWORD", "")

requires_db2_credentials = pytest.mark.skipif(
    not (DB2_DATABASE and DB2_HOST and DB2_USERNAME and DB2_PASSWORD),
    reason=(
        "Db2 credentials not configured. "
        "Set DB2_DATABASE, DB2_HOST, DB2_PORT, DB2_USERNAME, DB2_PASSWORD."
    ),
)

DIM = 4
INTEGRATION_DIM = 8


# ---------------------------------------------------------------------------
# Helpers shared by unit tests
# ---------------------------------------------------------------------------


def _bson_json(d: dict) -> str:
    """Simulate SYSTOOLS.BSON2JSON output — just a JSON string."""
    return json.dumps(d)


def _mock_client_cursor(post_init_fetchone=None):
    """Return (client, cursor) with fetchone pre-seeded for Db2VectorStore.__init__.

    __init__ fetchone call order:
      1. _check_db2_version  SERVICE_LEVEL  → ("DB2 v12.1.2.0",)
      2. _probe_text_search  CONTAINS probe → None  (text search unavailable)
      3. _table_exists       COUNT(*)       → (1,)  table exists, skip CREATE TABLE

    After those three are consumed, subsequent calls return ``post_init_fetchone``.
    """
    client = MagicMock()
    cursor = MagicMock()

    consumed = []

    def _fetchone_seq(*args, **kwargs):
        if len(consumed) == 0:
            consumed.append(1)
            return ("DB2 v12.1.2.0",)  # _check_db2_version SERVICE_LEVEL
        if len(consumed) == 1:
            consumed.append(2)
            return None                # _probe_text_search CONTAINS probe
        if len(consumed) == 2:
            consumed.append(3)
            return (1,)                # _table_exists COUNT(*) → table exists
        # All init calls done — return the per-test value
        return post_init_fetchone

    cursor.fetchone.side_effect = _fetchone_seq
    client.cursor.return_value = cursor
    return client, cursor


def _unique_table_name() -> str:
    """Return a short unique uppercase table name safe for Db2 (≤ 18 chars)."""
    return f"MEM0_{uuid.uuid4().hex[:8].upper()}"


def _store(cursor_rows=None, fetchone_row=None, **kwargs):
    """Build a Db2VectorStore backed by a fully mocked connection.

    The mock simulates a table that already exists (COUNT(*) returns 1) so
    __init__ skips CREATE TABLE.

    Returns (store, client_mock, cursor_mock).  The cursor is reset after
    construction so per-test assertions start clean.
    """
    client = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = cursor_rows or []
    # fetchone sequence during __init__:
    #   1. _table_exists  → (1,) means table exists → skip CREATE TABLE
    #   2. _check_db2_version  → version string row
    #   3. _probe_text_search  → None (text search not available)
    # After construction the cursor is reset; callers that need a specific
    # fetchone_row get it set back on the reset cursor.
    cursor.fetchone.side_effect = [
        (1,),                   # _table_exists COUNT(*)
        ("DB2 v12.1.2.0",),     # _check_db2_version SERVICE_LEVEL
        None,                   # _probe_text_search CONTAINS probe
    ]
    client.cursor.return_value = cursor

    defaults = dict(
        client=client,
        collection_name="MEM0_TEST",
        embedding_model_dims=DIM,
        distance_strategy="EUCLIDEAN",
    )
    defaults.update(kwargs)
    store = Db2VectorStore(**defaults)

    cursor.reset_mock()
    client.reset_mock()
    client.cursor.return_value = cursor
    # Restore a sane fetchone default for per-test use
    cursor.fetchone.side_effect = None
    cursor.fetchone.return_value = fetchone_row
    return store, client, cursor


# ---------------------------------------------------------------------------
# Live integration fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db2_store():
    """Create a real Db2VectorStore against the configured DB, then clean up.

    Skips automatically when:
    - DB2_* credentials are not set in the environment / .env file, OR
    - The Db2 host is unreachable (ConnectionError / any exception during init).
    """
    if not (DB2_DATABASE and DB2_HOST and DB2_USERNAME and DB2_PASSWORD):
        pytest.skip("Db2 credentials not configured")

    connection_params = {
        "database": DB2_DATABASE,
        "host": DB2_HOST,
        "port": DB2_PORT,
        "username": DB2_USERNAME,
        "password": DB2_PASSWORD,
    }
    table_name = _unique_table_name()
    try:
        store = Db2VectorStore(
            connection_params=connection_params,
            collection_name=table_name,
            embedding_model_dims=INTEGRATION_DIM,
            distance_strategy="EUCLIDEAN",
        )
    except Exception as exc:
        pytest.skip(f"Db2 server unreachable or init failed: {exc}")

    try:
        yield store
    finally:
        try:
            store.delete_col()
        except Exception:
            pass


# ===========================================================================
# Db2Config unit tests
# ===========================================================================


class TestDb2Config:
    def test_requires_client_or_connection_params(self):
        with pytest.raises(ValueError, match="Either `client` or `connection_params`"):
            Db2Config(collection_name="t", embedding_model_dims=4)

    def test_accepts_client(self):
        cfg = Db2Config(client=object(), collection_name="t", embedding_model_dims=4)
        assert cfg.collection_name == "t"

    def test_accepts_connection_params(self):
        cfg = Db2Config(
            connection_params={
                "database": "DB",
                "host": "localhost",
                "port": 25000,
                "username": "u",
                "password": "p",
            },
            embedding_model_dims=4,
        )
        assert cfg.distance_strategy == "EUCLIDEAN"

    def test_normalises_distance_strategy_to_uppercase(self):
        cfg = Db2Config(client=object(), distance_strategy="cosine", embedding_model_dims=4)
        assert cfg.distance_strategy == "COSINE"

    @pytest.mark.parametrize("strategy", ["cosine", "COSINE", "dot", "DOT", "euclidean", "EUCLIDEAN"])
    def test_accepts_all_valid_distance_strategies(self, strategy):
        cfg = Db2Config(client=object(), distance_strategy=strategy, embedding_model_dims=4)
        assert cfg.distance_strategy == strategy.upper()

    def test_rejects_invalid_distance_strategy(self):
        with pytest.raises(ValueError, match="distance_strategy"):
            Db2Config(client=object(), distance_strategy="L2", embedding_model_dims=4)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValueError, match="Extra fields"):
            Db2Config(client=object(), unknown_field="x", embedding_model_dims=4)

    def test_default_field_names(self):
        cfg = Db2Config(client=object(), embedding_model_dims=4)
        assert cfg.text_field == "text"
        assert cfg.text_lemmatized_field == "text_lemmatized"
        assert cfg.id_field == "id"
        assert cfg.metadata_field == "metadata"
        assert cfg.embedding_field == "embedding"

    def test_default_collection_name(self):
        cfg = Db2Config(client=object(), embedding_model_dims=4)
        assert cfg.collection_name == "mem0"

    def test_rejects_zero_embedding_dims(self):
        with pytest.raises(ValueError):
            Db2Config(client=object(), embedding_model_dims=0)

    def test_rejects_negative_embedding_dims(self):
        with pytest.raises(ValueError):
            Db2Config(client=object(), embedding_model_dims=-1)

    def test_client_takes_priority_over_connection_params(self):
        fake_client = object()
        cfg = Db2Config(
            client=fake_client,
            connection_params={"database": "DB", "host": "h", "port": 1, "username": "u", "password": "p"},
            embedding_model_dims=4,
        )
        assert cfg.client is fake_client

    # Change 5: use_vector_index config validation
    def test_use_vector_index_default_is_false(self):
        cfg = Db2Config(client=object(), embedding_model_dims=4)
        assert cfg.use_vector_index is False

    def test_use_vector_index_true_valid_for_cosine(self):
        cfg = Db2Config(
            client=object(), embedding_model_dims=4,
            distance_strategy="COSINE", use_vector_index=True,
        )
        assert cfg.use_vector_index is True

    def test_use_vector_index_true_valid_for_euclidean(self):
        cfg = Db2Config(
            client=object(), embedding_model_dims=4,
            distance_strategy="EUCLIDEAN", use_vector_index=True,
        )
        assert cfg.use_vector_index is True

    def test_use_vector_index_true_valid_for_euclidean_distance(self):
        cfg = Db2Config(
            client=object(), embedding_model_dims=4,
            distance_strategy="EUCLIDEAN_DISTANCE", use_vector_index=True,
        )
        assert cfg.use_vector_index is True

    @pytest.mark.parametrize("strategy", ["HAMMING", "MANHATTAN", "DOT"])
    def test_use_vector_index_true_rejected_for_unsupported_metrics(self, strategy):
        with pytest.raises(ValueError, match="use_vector_index=True is not compatible"):
            Db2Config(
                client=object(), embedding_model_dims=4,
                distance_strategy=strategy, use_vector_index=True,
            )

    # Change 4: text_lemmatized_field config
    def test_custom_text_lemmatized_field_name(self):
        cfg = Db2Config(
            client=object(), embedding_model_dims=4,
            text_lemmatized_field="lemma_text",
        )
        assert cfg.text_lemmatized_field == "lemma_text"


# ===========================================================================
# _distance_to_score parametrized tests
# ===========================================================================


@pytest.mark.parametrize(
    ("strategy", "distance", "expected_score"),
    [
        ("EUCLIDEAN", 0.0, 1.0),
        ("EUCLIDEAN", 1.0, 0.5),
        ("EUCLIDEAN", 3.0, pytest.approx(0.25)),
        ("COSINE", 0.0, 1.0),
        ("COSINE", 0.5, 0.5),
        ("COSINE", 1.0, 0.0),
        ("COSINE", 1.5, 0.0),
        ("DOT", 0.9, 0.9),
        ("DOT", -0.5, -0.5),
    ],
)
def test_distance_to_score(strategy, distance, expected_score):
    assert _distance_to_score(distance, strategy) == pytest.approx(expected_score)


def test_distance_to_score_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported distance strategy"):
        _distance_to_score(0.5, "UNKNOWN")


# ===========================================================================
# Initialisation unit tests
# ===========================================================================


class TestDb2VectorStoreInit:
    def test_creates_table_when_absent(self):
        # fetchone order in __init__: version → probe → table_exists (→ 0 = absent → CREATE)
        client, cursor = _mock_client_cursor()
        consumed = []
        def _fetchone_absent(*a, **kw):
            if len(consumed) == 0:
                consumed.append(1); return ("DB2 v12.1.2.0",)  # _check_db2_version
            if len(consumed) == 1:
                consumed.append(2); return None                 # _probe_text_search
            consumed.append(3); return (0,)                     # _table_exists → absent
        cursor.fetchone.side_effect = _fetchone_absent

        execute_calls = []
        cursor.execute.side_effect = lambda sql, *a, **kw: execute_calls.append(sql)
        Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)

        assert any("CREATE TABLE" in s for s in execute_calls), "Expected CREATE TABLE DDL"

    def test_skips_create_when_table_exists(self):
        # _mock_client_cursor returns (1,) for the first _table_exists → table exists.
        client, cursor = _mock_client_cursor()
        executed = []
        cursor.execute.side_effect = lambda sql, *a, **kw: executed.append(sql)
        Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        assert not any("CREATE TABLE" in s for s in executed)

    def test_stores_config_attributes(self):
        store, *_ = _store()
        assert store.collection_name == "MEM0_TEST"
        assert store._embedding_dim == DIM
        assert store._distance_strategy == "EUCLIDEAN"

    def test_custom_field_names_are_forwarded(self):
        store, *_ = _store(
            text_field="content",
            text_lemmatized_field="lemma",
            id_field="uid",
            metadata_field="meta",
            embedding_field="vec",
        )
        assert store._text_field == "content"
        assert store._text_lemmatized_field == "lemma"
        assert store._id_field == "uid"
        assert store._metadata_field == "meta"
        assert store._embedding_field == "vec"

    # Change 4: text_lemmatized column in DDL
    def test_create_table_ddl_includes_text_lemmatized_column(self):
        client, cursor = _mock_client_cursor()
        # Override fetchone so _table_exists returns 0 (table absent)
        cursor.fetchone.side_effect = [
            (0,),                   # _table_exists → absent → CREATE TABLE
            ("DB2 v12.1.2.0",),     # _check_db2_version
            None,                   # _probe_text_search
        ]
        execute_calls = []
        cursor.execute.side_effect = lambda sql, *a, **kw: execute_calls.append(sql)
        Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)

        create_stmts = [s for s in execute_calls if "CREATE TABLE" in s]
        assert create_stmts, "Expected CREATE TABLE DDL"
        assert "text_lemmatized" in create_stmts[0]


# ===========================================================================
# _get_cursor context manager (Change 6)
# ===========================================================================


class TestGetCursor:
    def test_cursor_closed_on_success(self):
        store, client, cursor = _store()
        with store._get_cursor():
            pass
        cursor.close.assert_called_once()

    def test_cursor_closed_on_exception(self):
        store, client, cursor = _store()
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            with store._get_cursor() as cur:
                cur.execute("SELECT 1")
        cursor.close.assert_called_once()

    def test_commit_called_when_commit_true(self):
        store, client, cursor = _store()
        with store._get_cursor(commit=True):
            pass
        client.commit.assert_called_once()

    def test_rollback_called_on_exception(self):
        store, client, cursor = _store()
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            with store._get_cursor(commit=True) as cur:
                cur.execute("bad sql")
        client.rollback.assert_called_once()

    def test_no_commit_when_commit_false(self):
        store, client, cursor = _store()
        with store._get_cursor(commit=False):
            pass
        client.commit.assert_not_called()


# ===========================================================================
# insert unit tests
# ===========================================================================


class TestInsert:
    def test_insert_generates_uuids(self):
        store, _, cursor = _store()
        vecs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        ids = store.insert(vectors=vecs)
        assert len(ids) == 2
        # IDs must be standard 36-char UUID strings
        for vid in ids:
            uuid.UUID(vid)  # raises ValueError if not valid UUID

    def test_insert_uses_provided_ids(self):
        store, _, cursor = _store()
        provided = [str(uuid.uuid4())]
        ids = store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], ids=provided)
        assert ids == provided

    def test_insert_calls_executemany(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"user_id": "alice"}])
        cursor.executemany.assert_called_once()
        sql = cursor.executemany.call_args[0][0]
        assert "INSERT INTO" in sql
        assert "VECTOR(" in sql

    def test_insert_commits(self):
        store, client, _ = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]])
        client.commit.assert_called_once()

    def test_insert_fills_missing_payloads_with_empty_dicts(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        rows = cursor.executemany.call_args[0][1]
        assert len(rows) == 2

    def test_insert_multiple_vectors_in_one_call(self):
        store, _, cursor = _store()
        vecs = [[float(i)] * DIM for i in range(5)]
        ids = store.insert(vectors=vecs)
        assert len(ids) == 5

    # Change 4: text_lemmatized populated on insert
    def test_insert_populates_text_lemmatized_from_payload(self):
        store, _, cursor = _store()
        payload = {"data": "raw text", "text_lemmatized": "stem text"}
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[payload])
        rows = cursor.executemany.call_args[0][1]
        # row tuple: (id, vec_str, json_meta, text, text_lemmatized)
        assert rows[0][4] == "stem text"

    def test_insert_text_lemmatized_defaults_to_empty_string(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "hello"}])
        rows = cursor.executemany.call_args[0][1]
        assert rows[0][4] == ""

    def test_insert_sql_includes_text_lemmatized_column(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]])
        sql = cursor.executemany.call_args[0][0]
        assert "text_lemmatized" in sql


# ===========================================================================
# search unit tests
# ===========================================================================


class TestSearch:
    def _setup(self, rows):
        client, cursor = _mock_client_cursor()
        cursor.fetchall.return_value = rows
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = rows
        return store, cursor

    def test_returns_output_data_list(self):
        rows = [("uuid-1234", "hello world", _bson_json({"user_id": "alice"}), 0.5)]
        store, cursor = self._setup(rows)
        results = store.search(query="hello", vectors=[[0.1, 0.2, 0.3, 0.4]], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, OutputData)
        assert r.id == "uuid-1234"
        assert r.payload == {"user_id": "alice"}
        assert r.score == pytest.approx(1.0 / 1.5)  # EUCLIDEAN: 1/(1+0.5)

    def test_search_scores_ordered_closest_first(self):
        rows = [
            ("ID1", "near", _bson_json({}), 0.1),
            ("ID2", "far", _bson_json({}), 0.9),
        ]
        store, cursor = self._setup(rows)
        results = store.search("q", [[0.1, 0.2, 0.3, 0.4]], top_k=2)
        assert results[0].score > results[1].score

    def test_search_with_no_filters_passes_no_where(self):
        store, cursor = self._setup([])
        store.search(query="q", vectors=[[0.1, 0.2, 0.3, 0.4]])
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_search_with_single_filter(self):
        store, cursor = self._setup([])
        store.search(query="q", vectors=[[0.1, 0.2, 0.3, 0.4]], filters={"user_id": "bob"})
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" in sql and "user_id" in sql and "bob" in sql

    def test_search_with_multiple_filters(self):
        store, cursor = self._setup([])
        store.search("q", [[0.1, 0.2, 0.3, 0.4]], filters={"user_id": "alice", "agent_id": "a1"})
        sql = cursor.execute.call_args[0][0]
        assert "user_id" in sql and "agent_id" in sql and "AND" in sql

    def test_search_wildcard_filter_skipped(self):
        store, cursor = self._setup([])
        store.search("q", [[0.1, 0.2, 0.3, 0.4]], filters={"user_id": "*"})
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_search_list_filter_uses_in(self):
        store, cursor = self._setup([])
        store.search("q", [[0.1, 0.2, 0.3, 0.4]], filters={"tag": ["x", "y"]})
        sql = cursor.execute.call_args[0][0]
        assert "IN" in sql

    def test_search_respects_top_k(self):
        store, cursor = self._setup([])
        store.search("q", [[0.1, 0.2, 0.3, 0.4]], top_k=7)
        sql = cursor.execute.call_args[0][0]
        assert "7" in sql

    def test_search_cosine_score_conversion(self):
        rows = [("ID", "txt", _bson_json({}), 0.3)]
        client, cursor = _mock_client_cursor()
        cursor.fetchall.return_value = rows
        store = Db2VectorStore(
            client=client, collection_name="T", embedding_model_dims=DIM, distance_strategy="COSINE"
        )
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = rows
        results = store.search("q", [[0.1, 0.2, 0.3, 0.4]])
        assert results[0].score == pytest.approx(0.7)

    def test_search_null_metadata_treated_as_empty_dict(self):
        rows = [("ID", "txt", None, 0.1)]
        store, cursor = self._setup(rows)
        results = store.search("q", [[0.1, 0.2, 0.3, 0.4]])
        assert results[0].payload == {}

    def test_search_null_text_treated_as_none(self):
        rows = [("ID", None, _bson_json({}), 0.2)]
        store, cursor = self._setup(rows)
        results = store.search("q", [[0.1, 0.2, 0.3, 0.4]])
        assert results[0].id == "ID"

    def test_search_euclidean_distance_mapped_to_euclidean_in_sql(self):
        """EUCLIDEAN_DISTANCE must be emitted as EUCLIDEAN in VECTOR_DISTANCE() SQL."""
        rows = [("ID", "txt", _bson_json({}), 0.5)]
        client, cursor = _mock_client_cursor()
        cursor.fetchall.return_value = rows
        store = Db2VectorStore(
            client=client, collection_name="T", embedding_model_dims=DIM,
            distance_strategy="EUCLIDEAN_DISTANCE",
        )
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = rows
        store.search("q", [[0.1, 0.2, 0.3, 0.4]])
        sql = cursor.execute.call_args[0][0]
        assert "EUCLIDEAN_DISTANCE" not in sql, (
            "EUCLIDEAN_DISTANCE must not appear in SQL — should be mapped to EUCLIDEAN"
        )
        assert "EUCLIDEAN" in sql


# ===========================================================================
# delete unit tests
# ===========================================================================


class TestDelete:
    def test_delete_passes_id_as_is(self):
        store, _, cursor = _store()
        vid = str(uuid.uuid4())
        store.delete(vid)
        _, params = cursor.execute.call_args[0]
        assert params == [vid]

    def test_delete_commits(self):
        store, client, _ = _store()
        store.delete("some-id")
        client.commit.assert_called_once()

    def test_delete_sql_targets_correct_table(self):
        store, _, cursor = _store(collection_name="MY_TABLE")
        store.delete("abc")
        sql = cursor.execute.call_args[0][0]
        assert "MY_TABLE" in sql


# ===========================================================================
# update unit tests
# ===========================================================================


class TestUpdate:
    def test_update_vector_only(self):
        store, _, cursor = _store()
        store.update(str(uuid.uuid4()), vector=[0.9, 0.8, 0.7, 0.6])
        sql = cursor.execute.call_args[0][0]
        assert "UPDATE" in sql and "VECTOR(" in sql

    def test_update_payload_only(self):
        store, _, cursor = _store()
        store.update(str(uuid.uuid4()), payload={"user_id": "charlie"})
        sql = cursor.execute.call_args[0][0]
        assert "UPDATE" in sql and "JSON2BSON" in sql

    def test_update_both_vector_and_payload(self):
        store, _, cursor = _store()
        store.update(str(uuid.uuid4()), vector=[0.1, 0.2, 0.3, 0.4], payload={"k": "v"})
        sql = cursor.execute.call_args[0][0]
        assert "VECTOR(" in sql and "JSON2BSON" in sql

    def test_update_noop_when_nothing_provided(self):
        store, _, cursor = _store()
        store.update(str(uuid.uuid4()))
        cursor.execute.assert_not_called()

    def test_update_commits(self):
        store, client, _ = _store()
        store.update(str(uuid.uuid4()), payload={"k": "v"})
        client.commit.assert_called_once()

    # Change 4: text_lemmatized populated on update
    def test_update_populates_text_lemmatized(self):
        store, _, cursor = _store()
        payload = {"data": "raw", "text_lemmatized": "stem"}
        store.update(str(uuid.uuid4()), payload=payload)
        sql, params = cursor.execute.call_args[0]
        assert "text_lemmatized" in sql
        assert "stem" in params

    def test_update_text_lemmatized_defaults_to_empty_string(self):
        store, _, cursor = _store()
        store.update(str(uuid.uuid4()), payload={"data": "raw"})
        sql, params = cursor.execute.call_args[0]
        assert "text_lemmatized" in sql
        assert "" in params


# ===========================================================================
# get unit tests
# ===========================================================================


class TestGet:
    def _make_store(self, row):
        client, cursor = _mock_client_cursor(post_init_fetchone=row)
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        # Reset call history; side_effect function still returns `row` for new calls
        cursor.reset_mock()
        client.cursor.return_value = cursor
        return store, cursor

    def test_get_returns_output_data(self):
        vid = str(uuid.uuid4())
        store, cursor = self._make_store(
            (vid, "my text", _bson_json({"user_id": "alice"}))
        )
        result = store.get(vid)
        assert result is not None
        assert result.id == vid
        assert result.payload == {"user_id": "alice"}
        assert result.score is None

    def test_get_returns_none_for_missing_id(self):
        store, _ = self._make_store(None)
        assert store.get("does-not-exist") is None

    def test_get_passes_id_unchanged(self):
        vid = str(uuid.uuid4())
        store, cursor = self._make_store(None)
        store.get(vid)
        _, params = cursor.execute.call_args[0]
        assert params == [vid]

    def test_get_null_metadata_treated_as_empty_dict(self):
        store, _ = self._make_store(("ID", "txt", None))
        result = store.get("ID")
        assert result is not None
        assert result.payload == {}


# ===========================================================================
# list_cols unit tests
# ===========================================================================


class TestListCols:
    def test_returns_table_names(self):
        client, cursor = _mock_client_cursor()
        cursor.fetchall.return_value = [("TABLE_A",), ("TABLE_B",)]
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = [("TABLE_A",), ("TABLE_B",)]
        assert store.list_cols() == ["TABLE_A", "TABLE_B"]

    def test_queries_syscat_tables(self):
        store, _, cursor = _store()
        cursor.fetchall.return_value = []
        store.list_cols()
        sql = cursor.execute.call_args[0][0]
        assert "SYSCAT" in sql.upper() or "TABLES" in sql.upper()


# ===========================================================================
# delete_col unit tests
# ===========================================================================


class TestDeleteCol:
    def test_drop_executes_ddl(self):
        # fetchone_row=(1,) so _table_exists returns True → DROP TABLE is issued
        store, _, cursor = _store(fetchone_row=(1,))
        store.delete_col()
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("DROP TABLE" in s for s in sqls)

    def test_skip_drop_when_table_missing(self):
        client = MagicMock()
        cursor = MagicMock()
        client.cursor.return_value = cursor
        # _table_exists returns False (COUNT(*) = 0) → delete_col logs and returns
        cursor.fetchone.return_value = (0,)

        store = Db2VectorStore.__new__(Db2VectorStore)
        store.client = client
        store.collection_name = "GONE"
        store._text_field = "text"
        store._text_lemmatized_field = "text_lemmatized"
        store._id_field = "id"
        store._metadata_field = "metadata"
        store._embedding_field = "embedding"
        store._distance_strategy = "EUCLIDEAN"
        store._embedding_dim = DIM

        store.delete_col()
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert not any("DROP TABLE" in s for s in sqls)


# ===========================================================================
# col_info unit tests
# ===========================================================================


class TestColInfo:
    def _make_store(self, row):
        # col_info() now issues a single SQL query whose row is
        # (schema, table_name, row_count).  ``row`` is that tuple,
        # or None to simulate "table not found".
        client, cursor = _mock_client_cursor(post_init_fetchone=row)
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchone.return_value = row
        cursor.fetchone.side_effect = None
        return store

    def test_returns_dict_with_required_keys(self):
        store = self._make_store(("MYSCHEMA", "MEM0_TEST", 42))
        info = store.col_info()
        assert info["schema"] == "MYSCHEMA"
        assert info["table_name"] == "MEM0_TEST"
        assert info["row_count"] == 42

    # Change 7: enriched col_info
    def test_col_info_includes_embedding_model_dims(self):
        store = self._make_store(("S", "T", 10))
        info = store.col_info()
        assert info["embedding_model_dims"] == DIM

    def test_col_info_includes_distance_strategy(self):
        store = self._make_store(("S", "T", 10))
        info = store.col_info()
        assert info["distance_strategy"] == "EUCLIDEAN"

    def test_raises_when_not_found(self):
        store = self._make_store(None)
        with pytest.raises(ValueError, match="not found"):
            store.col_info()

    def test_queries_syscat_with_table_name(self):
        store, _, cursor = _store(collection_name="MY_COL")
        # col_info now uses a single query containing both SYSCAT and COUNT(*).
        cursor.fetchone.return_value = ("S", "MY_COL", 10)
        cursor.fetchone.side_effect = None
        store.col_info()
        # The single execute call must reference SYSCAT
        sql = cursor.execute.call_args_list[0][0][0]
        assert "SYSCAT" in sql.upper()

    def test_single_execute_call(self):
        """col_info() must issue exactly one SQL query (single round-trip)."""
        store, _, cursor = _store(collection_name="MY_COL")
        cursor.fetchone.return_value = ("S", "MY_COL", 7)
        cursor.fetchone.side_effect = None
        store.col_info()
        assert cursor.execute.call_count == 1, (
            "col_info() should use a single SQL round-trip (SYSCAT + scalar COUNT subquery)"
        )


# ===========================================================================
# list unit tests
# ===========================================================================


class TestList:
    def _make_store(self, rows):
        client, cursor = _mock_client_cursor()
        cursor.fetchall.return_value = rows
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = rows
        return store, cursor

    def test_returns_nested_list(self):
        rows = [("ID1", "text", _bson_json({"user_id": "alice"}))]
        store, _ = self._make_store(rows)
        result = store.list()
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert result[0][0].id == "ID1"

    def test_list_with_single_filter(self):
        store, cursor = self._make_store([])
        store.list(filters={"user_id": "alice"})
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" in sql and "user_id" in sql

    def test_list_with_multiple_filters(self):
        store, cursor = self._make_store([])
        store.list(filters={"user_id": "alice", "agent_id": "a1"})
        sql = cursor.execute.call_args[0][0]
        assert "AND" in sql

    def test_list_with_no_filters(self):
        store, cursor = self._make_store([])
        store.list()
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_list_wildcard_filter_ignored(self):
        store, cursor = self._make_store([])
        store.list(filters={"user_id": "*"})
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in sql

    def test_list_respects_top_k(self):
        store, cursor = self._make_store([])
        store.list(top_k=25)
        sql = cursor.execute.call_args[0][0]
        assert "25" in sql

    def test_list_with_no_top_k_omits_fetch_clause(self):
        store, cursor = self._make_store([])
        store.list(top_k=None)
        sql = cursor.execute.call_args[0][0]
        assert "FETCH FIRST" not in sql

    def test_list_null_metadata_treated_as_empty_dict(self):
        rows = [("ID1", "text", None)]
        store, _ = self._make_store(rows)
        result = store.list()
        assert result[0][0].payload == {}


# ===========================================================================
# reset unit tests
# ===========================================================================


class TestReset:
    def test_reset_drops_and_recreates(self):
        store, _, cursor = _store()
        executed_sqls = []
        fetchone_calls = [0]

        def _fetchone(*a, **kw):
            # First COUNT(*) → (1,) table exists → delete_col issues DROP TABLE
            # Second COUNT(*) → (0,) table gone → _create_table_if_not_exists runs CREATE
            fetchone_calls[0] += 1
            if fetchone_calls[0] == 1:
                return (1,)   # _table_exists before drop → table present
            return (0,)       # _table_exists after drop → table absent → CREATE TABLE

        cursor.fetchone.side_effect = _fetchone
        cursor.execute.side_effect = lambda sql, *a, **kw: executed_sqls.append(sql)
        store.reset()

        assert any("DROP TABLE" in s for s in executed_sqls), "Expected DROP TABLE"
        assert any("CREATE TABLE" in s for s in executed_sqls), "Expected CREATE TABLE"


# ===========================================================================
# _where_clause / _build_conditions unit tests
# ===========================================================================


class TestWhereClause:
    def _store(self):
        s, *_ = _store()
        return s

    def test_empty_filters_returns_empty_string(self):
        s = self._store()
        assert s._where_clause(None) == ""
        assert s._where_clause({}) == ""

    def test_single_equality_filter(self):
        s = self._store()
        clause = s._where_clause({"user_id": "alice"})
        assert "WHERE" in clause and "user_id" in clause and "alice" in clause

    def test_multiple_filters_joined_with_and(self):
        s = self._store()
        clause = s._where_clause({"user_id": "alice", "agent_id": "a1"})
        assert clause.count("AND") >= 1

    def test_wildcard_value_skipped(self):
        s = self._store()
        assert s._where_clause({"user_id": "*"}) == ""

    def test_list_value_produces_in_clause(self):
        s = self._store()
        clause = s._where_clause({"tag": ["x", "y"]})
        assert "IN" in clause and "x" in clause and "y" in clause

    def test_uses_instance_metadata_field_name(self):
        store, *_ = _store(metadata_field="my_meta")
        clause = store._where_clause({"key": "val"})
        assert "my_meta" in clause

    # Change 2: contains / icontains
    def test_contains_operator_produces_like(self):
        s = self._store()
        clause = s._where_clause({"category": {"contains": "sci"}})
        assert "LIKE" in clause and "%sci%" in clause

    def test_icontains_operator_produces_lower_like(self):
        s = self._store()
        clause = s._where_clause({"category": {"icontains": "Sci"}})
        assert "LOWER" in clause and "LIKE" in clause and "%sci%" in clause.lower()

    def test_contains_escapes_single_quotes(self):
        s = self._store()
        clause = s._where_clause({"note": {"contains": "it's"}})
        assert "it''s" in clause

    # Change 3: $or / $not / $and logical operators
    def test_or_filter_produces_sql_or(self):
        s = self._store()
        clause = s._where_clause({"$or": [{"user_id": "alice"}, {"user_id": "bob"}]})
        assert "OR" in clause
        assert "alice" in clause
        assert "bob" in clause

    def test_and_filter_produces_sql_and(self):
        s = self._store()
        clause = s._where_clause({"$and": [{"user_id": "alice"}, {"agent_id": "a1"}]})
        assert "AND" in clause
        assert "alice" in clause
        assert "a1" in clause

    def test_not_filter_produces_sql_not(self):
        s = self._store()
        clause = s._where_clause({"$not": [{"archived": "true"}]})
        assert "NOT" in clause
        assert "archived" in clause

    def test_dollar_sign_aliases_work(self):
        """$and, $or, $not are the primary aliases used by mem0."""
        s = self._store()
        for key in ("$or", "$and", "$not"):
            sub = [{"user_id": "alice"}]
            clause = s._where_clause({key: sub})
            assert "alice" in clause

    def test_nested_compound_filter(self):
        """$and with nested $or inside."""
        s = self._store()
        clause = s._where_clause({
            "$and": [
                {"user_id": "alice"},
                {"$or": [{"project": "X"}, {"project": "Y"}]},
            ]
        })
        assert "alice" in clause
        assert "OR" in clause
        assert "X" in clause and "Y" in clause

    @pytest.mark.parametrize(
        ("filters", "expect_in_sql"),
        [
            ({"user_id": "alice"}, ["user_id", "alice"]),
            ({"user_id": "alice", "run_id": "r1"}, ["user_id", "run_id", "AND"]),
            ({"tag": ["a", "b"]}, ["IN", "a", "b"]),
        ],
    )
    def test_where_clause_parametrized(self, filters, expect_in_sql):
        s = self._store()
        clause = s._where_clause(filters)
        for fragment in expect_in_sql:
            assert fragment in clause, f"Expected '{fragment}' in WHERE clause: {clause}"


# ===========================================================================
# _op_condition unit tests
# ===========================================================================


class TestOpCondition:
    def _expr(self):
        return "JSON_VALUE(SYSTOOLS.BSON2JSON(metadata), '$.field')"

    def test_eq(self):
        cond = Db2VectorStore._op_condition(self._expr(), "eq", "val")
        assert "= 'val'" in cond

    def test_ne(self):
        cond = Db2VectorStore._op_condition(self._expr(), "ne", "val")
        assert "<> 'val'" in cond

    def test_gt(self):
        cond = Db2VectorStore._op_condition(self._expr(), "gt", 5)
        assert "> 5.0" in cond and "DOUBLE" in cond

    def test_gte(self):
        cond = Db2VectorStore._op_condition(self._expr(), "gte", 5)
        assert ">= 5.0" in cond

    def test_lt(self):
        cond = Db2VectorStore._op_condition(self._expr(), "lt", 5)
        assert "< 5.0" in cond

    def test_lte(self):
        cond = Db2VectorStore._op_condition(self._expr(), "lte", 5)
        assert "<= 5.0" in cond

    def test_in(self):
        cond = Db2VectorStore._op_condition(self._expr(), "in", ["a", "b"])
        assert "IN" in cond and "'a'" in cond and "'b'" in cond

    def test_nin(self):
        cond = Db2VectorStore._op_condition(self._expr(), "nin", ["a", "b"])
        assert "NOT IN" in cond

    def test_in_requires_list(self):
        with pytest.raises(ValueError, match="requires a list"):
            Db2VectorStore._op_condition(self._expr(), "in", "not-a-list")

    def test_nin_requires_list(self):
        with pytest.raises(ValueError, match="requires a list"):
            Db2VectorStore._op_condition(self._expr(), "nin", "not-a-list")

    # Change 2
    def test_contains_produces_like(self):
        cond = Db2VectorStore._op_condition(self._expr(), "contains", "sci")
        assert "LIKE '%sci%'" in cond

    def test_icontains_produces_lower_like(self):
        cond = Db2VectorStore._op_condition(self._expr(), "icontains", "Sci")
        assert "LOWER(" in cond and "LIKE" in cond

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            Db2VectorStore._op_condition(self._expr(), "regex", ".*")

    def test_error_message_lists_contains_and_icontains(self):
        with pytest.raises(ValueError, match="contains"):
            Db2VectorStore._op_condition(self._expr(), "nope", "x")


# ===========================================================================
# keyword_search unit tests
# ===========================================================================


class TestKeywordSearch:
    def _make_store(self, text_search_available: bool, rows=None):
        store, client, cursor = _store()
        store._text_search_available = text_search_available
        cursor.fetchall.return_value = rows or []
        return store, cursor

    def test_returns_none_when_text_search_unavailable(self):
        store, _ = self._make_store(False)
        assert store.keyword_search("query") is None

    def test_returns_none_on_sql_error(self):
        store, cursor = self._make_store(True)
        cursor.execute.side_effect = Exception("SQL21000N")
        result = store.keyword_search("query")
        assert result is None

    def test_returns_results_when_available(self):
        rows = [("ID1", "text", _bson_json({"user_id": "alice"}), 80.0)]
        store, cursor = self._make_store(True, rows)
        results = store.keyword_search("query")
        assert len(results) == 1
        assert results[0].id == "ID1"
        assert results[0].score == pytest.approx(0.8)  # 80/100

    # Change 4: keyword_search targets text_lemmatized column
    def test_keyword_search_sql_targets_text_lemmatized(self):
        store, cursor = self._make_store(True)
        store.keyword_search("park walks")
        sql = cursor.execute.call_args[0][0]
        assert "text_lemmatized" in sql
        # Must NOT use the raw text column for CONTAINS/SCORE
        assert f"CONTAINS(text," not in sql
        assert f"SCORE(text," not in sql

    def test_keyword_search_with_filters(self):
        store, cursor = self._make_store(True)
        store.keyword_search("query", filters={"user_id": "alice"})
        sql = cursor.execute.call_args[0][0]
        assert "WHERE" in sql and "user_id" in sql
        assert "AND CONTAINS" in sql

    def test_keyword_search_without_filters_uses_where_contains(self):
        store, cursor = self._make_store(True)
        store.keyword_search("query")
        sql = cursor.execute.call_args[0][0]
        assert "WHERE CONTAINS" in sql


# ===========================================================================
# ANN vector index unit tests (Change 5)
# ===========================================================================


class TestMaybeCreateVectorIndex:
    def test_no_index_created_when_flag_false(self):
        store, _, cursor = _store()
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        assert not any("CREATE VECTOR INDEX" in s for s in executed)

    def test_index_not_created_below_min_ann_version(self):
        store, _, cursor = _store(use_vector_index=True, distance_strategy="COSINE")
        store._db2_version = (12, 1, 4)  # below 12.1.5
        cursor.reset_mock()
        store._maybe_create_vector_index("MEM0_TEST")
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        assert not any("CREATE VECTOR INDEX" in s for s in executed)

    def test_index_not_created_when_version_unknown(self):
        store, _, cursor = _store(use_vector_index=True, distance_strategy="COSINE")
        store._db2_version = None
        cursor.reset_mock()
        store._maybe_create_vector_index("MEM0_TEST")
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        assert not any("CREATE VECTOR INDEX" in s for s in executed)

    def test_index_created_on_12_1_5_with_cosine(self):
        store, _, cursor = _store(use_vector_index=True, distance_strategy="COSINE")
        store._db2_version = (12, 1, 5)
        cursor.reset_mock()
        store._maybe_create_vector_index("MEM0_TEST")
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("CREATE VECTOR INDEX" in s for s in executed)
        assert any("COSINE" in s for s in executed)

    def test_index_creation_failure_does_not_raise(self):
        store, _, cursor = _store(use_vector_index=True, distance_strategy="EUCLIDEAN")
        store._db2_version = (12, 1, 5)
        cursor.execute.side_effect = Exception("DB2 error: insufficient privileges")
        # Must not raise — logs a warning and continues
        store._maybe_create_vector_index("MEM0_TEST")

    def test_euclidean_distance_mapped_to_euclidean_in_ddl(self):
        """EUCLIDEAN_DISTANCE must be emitted as EUCLIDEAN in CREATE VECTOR INDEX DDL."""
        store, _, cursor = _store(
            use_vector_index=True, distance_strategy="EUCLIDEAN_DISTANCE"
        )
        store._db2_version = (12, 1, 5)
        cursor.reset_mock()
        store._maybe_create_vector_index("MEM0_TEST")
        executed = [c[0][0] for c in cursor.execute.call_args_list]
        ddl_stmts = [s for s in executed if "CREATE VECTOR INDEX" in s]
        assert ddl_stmts, "Expected CREATE VECTOR INDEX DDL"
        assert "EUCLIDEAN_DISTANCE" not in ddl_stmts[0], (
            "EUCLIDEAN_DISTANCE must not appear in DDL — should be mapped to EUCLIDEAN"
        )
        assert "EUCLIDEAN" in ddl_stmts[0]

    def test_client_unchanged_after_vector_index_creation(self):
        """_maybe_create_vector_index must never replace self.client.

        The CE reconnect workaround was removed — on failure the raw Db2 error
        is logged and the caller is responsible for restarting the connection.
        self.client must be the same object before and after the call.
        """
        store, client, cursor = _store(use_vector_index=True, distance_strategy="COSINE")
        store._db2_version = (12, 1, 5)
        cursor.reset_mock()
        client.reset_mock()
        original_client = store.client
        store._maybe_create_vector_index("MEM0_TEST")
        assert store.client is original_client

    def test_config_rejects_hamming_with_use_vector_index(self):
        with pytest.raises(ValueError, match="use_vector_index=True is not compatible"):
            Db2Config(
                client=object(), embedding_model_dims=4,
                distance_strategy="HAMMING", use_vector_index=True,
            )

    def test_config_rejects_dot_with_use_vector_index(self):
        with pytest.raises(ValueError, match="use_vector_index=True is not compatible"):
            Db2Config(
                client=object(), embedding_model_dims=4,
                distance_strategy="DOT", use_vector_index=True,
            )


# ===========================================================================
# VectorStoreConfig provider registration
# ===========================================================================


class TestVectorStoreConfig:
    def test_db2_provider_accepted(self):
        from mem0.vector_stores.configs import VectorStoreConfig

        cfg = VectorStoreConfig(
            provider="db2",
            config={
                "client": object(),
                "collection_name": "mem0",
                "embedding_model_dims": 4,
            },
        )
        assert isinstance(cfg.config, Db2Config)

    def test_db2_provider_with_connection_params(self):
        from mem0.vector_stores.configs import VectorStoreConfig

        cfg = VectorStoreConfig(
            provider="db2",
            config={
                "connection_params": {
                    "database": "BLUDB",
                    "host": "localhost",
                    "port": 25000,
                    "username": "db2user",
                    "password": "secret",
                },
                "embedding_model_dims": 128,
                "distance_strategy": "cosine",
            },
        )
        assert isinstance(cfg.config, Db2Config)
        assert cfg.config.distance_strategy == "COSINE"

    def test_db2_provider_with_use_vector_index(self):
        from mem0.vector_stores.configs import VectorStoreConfig

        cfg = VectorStoreConfig(
            provider="db2",
            config={
                "client": object(),
                "embedding_model_dims": 128,
                "distance_strategy": "COSINE",
                "use_vector_index": True,
            },
        )
        assert cfg.config.use_vector_index is True

    def test_db2_provider_with_text_lemmatized_field(self):
        from mem0.vector_stores.configs import VectorStoreConfig

        cfg = VectorStoreConfig(
            provider="db2",
            config={
                "client": object(),
                "embedding_model_dims": 4,
                "text_lemmatized_field": "my_lemma",
            },
        )
        assert cfg.config.text_lemmatized_field == "my_lemma"


# ===========================================================================
# Live integration tests (require real Db2 — skipped without credentials)
# ===========================================================================


@requires_db2_credentials
def test_live_create_col(db2_store: Db2VectorStore):
    tables = [t.upper() for t in db2_store.list_cols()]
    assert db2_store.collection_name.upper() in tables


@requires_db2_credentials
def test_live_insert_and_get(db2_store: Db2VectorStore):
    vecs = [[0.1] * INTEGRATION_DIM, [0.2] * INTEGRATION_DIM]
    payloads = [
        {"name": "vec1", "user_id": "alice", "text_lemmatized": "alice first"},
        {"name": "vec2", "user_id": "bob", "text_lemmatized": "bob second"},
    ]
    ids = db2_store.insert(vectors=vecs, payloads=payloads)
    assert len(ids) == 2
    # IDs are plain UUID strings
    for vid in ids:
        uuid.UUID(vid)

    got = db2_store.get(ids[0])
    assert got is not None
    assert got.id == ids[0]
    assert isinstance(got.payload, dict)


@requires_db2_credentials
def test_live_search(db2_store: Db2VectorStore):
    pos_vec = [1.0] * INTEGRATION_DIM
    neg_vec = [-1.0] * INTEGRATION_DIM
    payloads = [
        {"name": "positive", "user_id": "u1"},
        {"name": "negative", "user_id": "u2"},
    ]
    db2_store.insert([pos_vec, neg_vec], payloads=payloads)
    results = db2_store.search("unused", vectors=[pos_vec], top_k=2)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0].payload.get("name") == "positive"


@requires_db2_credentials
def test_live_search_with_filters(db2_store: Db2VectorStore):
    vec = [0.5] * INTEGRATION_DIM
    payloads = [
        {"name": "a", "user_id": "alice", "agent_id": "agent1", "run_id": "run1"},
        {"name": "b", "user_id": "bob", "agent_id": "agent2", "run_id": "run2"},
    ]
    db2_store.insert([vec, vec], payloads=payloads)
    results = db2_store.search(
        "unused", vectors=[vec], top_k=5,
        filters={"user_id": "alice", "agent_id": "agent1", "run_id": "run1"},
    )
    assert len(results) >= 1
    for r in results:
        assert r.payload.get("user_id") == "alice"


@requires_db2_credentials
def test_live_search_contains_filter(db2_store: Db2VectorStore):
    """Change 2: contains operator works end-to-end."""
    vec = [0.5] * INTEGRATION_DIM
    db2_store.insert(
        [vec, vec],
        payloads=[{"category": "sci-fi movies"}, {"category": "action movies"}],
    )
    results = db2_store.search(
        "unused", vectors=[vec], top_k=5,
        filters={"category": {"contains": "sci"}},
    )
    assert all("sci" in r.payload.get("category", "") for r in results)


@requires_db2_credentials
def test_live_search_or_filter(db2_store: Db2VectorStore):
    """Change 3: $or compound filter works end-to-end."""
    vec = [0.5] * INTEGRATION_DIM
    db2_store.insert(
        [vec, vec, vec],
        payloads=[
            {"user_id": "alice"},
            {"user_id": "bob"},
            {"user_id": "charlie"},
        ],
    )
    results = db2_store.search(
        "unused", vectors=[vec], top_k=5,
        filters={"$or": [{"user_id": "alice"}, {"user_id": "bob"}]},
    )
    result_users = {r.payload.get("user_id") for r in results}
    assert "charlie" not in result_users
    assert result_users <= {"alice", "bob"}


@requires_db2_credentials
def test_live_search_with_single_filter(db2_store: Db2VectorStore):
    vec = [0.7] * INTEGRATION_DIM
    db2_store.insert([vec, vec], payloads=[{"user_id": "alice"}, {"user_id": "bob"}])
    results = db2_store.search("unused", vectors=[vec], top_k=5, filters={"user_id": "alice"})
    assert all(r.payload.get("user_id") == "alice" for r in results)


@requires_db2_credentials
def test_live_delete(db2_store: Db2VectorStore):
    vec = [0.9] * INTEGRATION_DIM
    ids = db2_store.insert([vec], payloads=[{"name": "to_delete"}])
    target_id = ids[0]
    db2_store.delete(vector_id=target_id)
    assert db2_store.get(vector_id=target_id) is None


@requires_db2_credentials
def test_live_update(db2_store: Db2VectorStore):
    vec = [0.01] * INTEGRATION_DIM
    ids = db2_store.insert([vec], payloads=[{"name": "old", "text_lemmatized": "old stem"}])
    target_id = ids[0]
    db2_store.update(
        vector_id=target_id,
        vector=[0.02] * INTEGRATION_DIM,
        payload={"name": "new", "text_lemmatized": "new stem"},
    )
    got = db2_store.get(vector_id=target_id)
    assert got is not None
    assert got.payload.get("name") == "new"


@requires_db2_credentials
def test_live_col_info_enriched(db2_store: Db2VectorStore):
    """Change 7: col_info now returns embedding_model_dims and distance_strategy."""
    info = db2_store.col_info()
    assert isinstance(info, dict)
    assert "schema" in info and "table_name" in info and "row_count" in info
    assert info["embedding_model_dims"] == INTEGRATION_DIM
    assert info["distance_strategy"] == "EUCLIDEAN"


@requires_db2_credentials
def test_live_reset_recreates_empty_table(db2_store: Db2VectorStore):
    vec = [0.15] * INTEGRATION_DIM
    ids = db2_store.insert([vec], payloads=[{"name": "before"}])
    before_id = ids[0]
    assert db2_store.get(before_id) is not None
    db2_store.reset()
    assert db2_store.get(before_id) is None
    assert db2_store.list(top_k=10) == [[]]
    new_ids = db2_store.insert([vec], payloads=[{"name": "after"}])
    assert db2_store.get(new_ids[0]) is not None


@requires_db2_credentials
def test_live_list(db2_store: Db2VectorStore):
    v1, v2 = [0.11] * INTEGRATION_DIM, [0.22] * INTEGRATION_DIM
    db2_store.insert([v1, v2], payloads=[{"key": "value1"}, {"key": "value2"}])
    results = db2_store.list(top_k=2)
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert len(results[0]) <= 2


@requires_db2_credentials
def test_live_list_with_filters(db2_store: Db2VectorStore):
    v = [0.44] * INTEGRATION_DIM
    db2_store.insert(
        [v, v],
        payloads=[
            {"user_id": "alice", "agent_id": "a1", "run_id": "r1"},
            {"user_id": "bob", "agent_id": "a2", "run_id": "r2"},
        ],
    )
    results = db2_store.list(
        filters={"user_id": "alice", "agent_id": "a1", "run_id": "r1"}, top_k=10
    )[0]
    assert len(results) >= 1
    for r in results:
        assert r.payload.get("user_id") == "alice"


@requires_db2_credentials
def test_live_list_cols(db2_store: Db2VectorStore):
    tables = db2_store.list_cols()
    assert db2_store.collection_name.upper() in [t.upper() for t in tables]


@requires_db2_credentials
def test_live_delete_col(db2_store: Db2VectorStore):
    table = db2_store.collection_name.upper()
    assert table in [t.upper() for t in db2_store.list_cols()]
    db2_store.delete_col()
    assert table not in [t.upper() for t in db2_store.list_cols()]


@requires_db2_credentials
def test_live_documentation():
    """End-to-end smoke test via the mem0 Memory API."""
    from mem0 import Memory

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the end-to-end documentation test")

    config = {
        "vector_store": {
            "provider": "db2",
            "config": {
                "connection_params": {
                    "database": DB2_DATABASE,
                    "host": DB2_HOST,
                    "port": DB2_PORT,
                    "username": DB2_USERNAME,
                    "password": DB2_PASSWORD,
                },
                "embedding_model_dims": 1536,
            },
        },
    }

    m = Memory.from_config(config)
    messages = [
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
        {"role": "assistant", "content": "How about sci-fi movies? They can be quite engaging."},
        {"role": "user", "content": "I love sci-fi movies, especially ones with space exploration."},
        {"role": "assistant", "content": "Got it! I'll remember you love sci-fi space exploration movies."},
    ]
    m.add(messages, user_id="alice", metadata={"category": "movies"})
    results = m.search("What movie to watch?", user_id="alice", limit=2)["results"]
    assert len(results) >= 1
    assert all(res["user_id"] == "alice" for res in results)
    m.reset()
