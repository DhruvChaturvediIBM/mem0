"""Unit and integration tests for the IBM Db2 vector store.

Unit tests (no real DB): all tests that do NOT have the ``@requires_db2_credentials``
marker.  They run in CI without any Db2 installation.

Integration tests (real DB): guarded by ``@requires_db2_credentials``.  Set the
following environment variables to enable them::

    DB2_DATABASE=TESTDB
    DB2_HOST=Geetika-5y420-x86.dev.fyre.ibm.com
    DB2_PORT=50000
    DB2_USERNAME=Geetika
    DB2_PASSWORD=Geet#246
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load .env from examples/misc/.env when present so that running
#   pytest tests/vector_stores/test_db2.py
# picks up credentials without needing a manual `source` or `export`.
# python-dotenv is an optional dependency; if absent the file is silently
# skipped and the usual env-var export approach still works.
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
# When ibm_db IS installed (integration test run) we leave sys.modules alone
# so the real connect() is used.
# ---------------------------------------------------------------------------

try:
    import ibm_db_dbi  # noqa: F401 — real driver present, no stub needed
except ImportError:
    _stub = ModuleType("ibm_db_dbi")
    _stub.connect = MagicMock()
    _stub.DatabaseError = Exception
    sys.modules["ibm_db_dbi"] = _stub

from mem0.configs.vector_stores.db2 import Db2Config  # noqa: E402
from mem0.vector_stores.db2 import Db2VectorStore, OutputData, _distance_to_score, _hash_id  # noqa: E402

# ---------------------------------------------------------------------------
# Integration credentials (read from env — populated either by .env above
# or by the caller's environment).
# ---------------------------------------------------------------------------

DB2_DATABASE = os.environ.get("DB2_DATABASE", "")
DB2_HOST = os.environ.get("DB2_HOST", "")
DB2_PORT = int(os.environ.get("DB2_PORT", "50000"))
DB2_USERNAME = os.environ.get("DB2_USERNAME", "")
DB2_PASSWORD = os.environ.get("DB2_PASSWORD", "")

requires_db2_credentials = pytest.mark.skipif(
    not (DB2_DATABASE and DB2_HOST and DB2_USERNAME and DB2_PASSWORD),
    reason=(
        "Db2 credentials not configured. "
        "Set DB2_DATABASE, DB2_HOST, DB2_PORT, DB2_USERNAME, DB2_PASSWORD."
    ),
)

DIM = 4  # small dimension for unit tests
INTEGRATION_DIM = 8  # slightly larger for live tests to detect shape errors early


# ---------------------------------------------------------------------------
# Helpers shared by unit tests
# ---------------------------------------------------------------------------


def _bson_json(d: dict) -> str:
    """Simulate SYSTOOLS.BSON2JSON output — just a JSON string."""
    return json.dumps(d)


def _unique_table_name() -> str:
    """Return a short unique uppercase table name safe for Db2 (≤ 18 chars)."""
    return f"MEM0_{uuid.uuid4().hex[:8].upper()}"


def _store(cursor_rows=None, fetchone_row=None, **kwargs):
    """Build a Db2VectorStore backed by a fully mocked connection.

    The mock simulates a table that already exists (SELECT COUNT(*) succeeds)
    so ``__init__`` skips CREATE TABLE.  ``cursor_rows`` / ``fetchone_row``
    are pre-loaded onto the cursor for callers that need fetchall / fetchone.

    Returns (store, client_mock, cursor_mock).  The cursor is reset after
    construction so per-test assertions start clean.
    """
    client = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = cursor_rows or []
    cursor.fetchone.return_value = fetchone_row
    client.cursor.return_value = cursor

    defaults = dict(
        client=client,
        collection_name="MEM0_TEST",
        embedding_model_dims=DIM,
        distance_strategy="EUCLIDEAN",
    )
    defaults.update(kwargs)
    store = Db2VectorStore(**defaults)

    # Clean slate so per-test assert_called / call_args checks are unambiguous
    cursor.reset_mock()
    client.reset_mock()
    client.cursor.return_value = cursor
    return store, client, cursor


# ---------------------------------------------------------------------------
# Live integration fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db2_store():
    """Create a real Db2VectorStore against TESTDB, then clean up."""
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
    store = Db2VectorStore(
        connection_params=connection_params,
        collection_name=table_name,
        embedding_model_dims=INTEGRATION_DIM,
        distance_strategy="EUCLIDEAN",
    )
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
                "port": 50000,
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
        """When both are supplied client wins (no error)."""
        fake_client = object()
        cfg = Db2Config(
            client=fake_client,
            connection_params={"database": "DB", "host": "h", "port": 1, "username": "u", "password": "p"},
            embedding_model_dims=4,
        )
        assert cfg.client is fake_client


# ===========================================================================
# _hash_id unit tests
# ===========================================================================


class TestHashId:
    def test_produces_16_char_uppercase_hex(self):
        result = _hash_id("hello")
        assert len(result) == 16
        assert result == result.upper()
        assert all(c in "0123456789ABCDEF" for c in result)

    def test_deterministic(self):
        assert _hash_id("same") == _hash_id("same")

    def test_different_inputs_differ(self):
        assert _hash_id("a") != _hash_id("b")

    def test_already_hashed_format_passthrough(self):
        """IDs already in CHAR(16) hex format are hashed again via the normal path —
        callers that want to pass pre-hashed IDs must use them with delete() which
        has the re-hash detection, not _hash_id directly."""
        result = _hash_id("ABCDEF1234567890")
        assert len(result) == 16


# ===========================================================================
# _distance_to_score parametrized tests (matching oracle test style)
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
        ("COSINE", 1.5, 0.0),   # clamped at 0
        ("DOT", 0.9, 0.9),
        ("DOT", -0.5, -0.5),    # dot preserves sign (higher = more similar)
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
        client = MagicMock()
        cursor = MagicMock()
        client.cursor.return_value = cursor

        execute_calls = []

        def _exec(sql, *a, **kw):
            execute_calls.append(sql)
            if "SELECT COUNT(*)" in sql and len(execute_calls) == 1:
                raise Exception("SQL0204N table not found")

        cursor.execute.side_effect = _exec
        Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)

        assert any("CREATE TABLE" in s for s in execute_calls), "Expected CREATE TABLE DDL"

    def test_skips_create_when_table_exists(self):
        client = MagicMock()
        cursor = MagicMock()
        client.cursor.return_value = cursor
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
            id_field="uid",
            metadata_field="meta",
            embedding_field="vec",
        )
        assert store._text_field == "content"
        assert store._id_field == "uid"
        assert store._metadata_field == "meta"
        assert store._embedding_field == "vec"


# ===========================================================================
# insert unit tests
# ===========================================================================


class TestInsert:
    def test_insert_generates_hashed_ids(self):
        store, _, cursor = _store()
        vecs = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        ids = store.insert(vectors=vecs)
        assert len(ids) == 2
        for hid in ids:
            assert len(hid) == 16 and hid == hid.upper()

    def test_insert_uses_provided_ids(self):
        store, _, cursor = _store()
        ids = store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], ids=["my-id"])
        assert ids == [_hash_id("my-id")]

    def test_insert_calls_executemany(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{"user_id": "alice"}])
        cursor.executemany.assert_called_once()
        sql = cursor.executemany.call_args[0][0]
        assert "INSERT INTO" in sql
        assert "VECTOR(" in sql

    def test_insert_commits(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]])
        commit_calls = [c for c in cursor.execute.call_args_list if "COMMIT" in str(c)]
        assert commit_calls

    def test_insert_fills_missing_payloads_with_empty_dicts(self):
        store, _, cursor = _store()
        store.insert(vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        # executemany rows — second arg is a list of tuples
        rows = cursor.executemany.call_args[0][1]
        assert len(rows) == 2

    def test_insert_multiple_vectors_in_one_call(self):
        store, _, cursor = _store()
        vecs = [[float(i)] * DIM for i in range(5)]
        ids = store.insert(vectors=vecs)
        assert len(ids) == 5


# ===========================================================================
# search unit tests
# ===========================================================================


class TestSearch:
    def _setup(self, rows):
        """Return (store, cursor) with fetchall pre-loaded."""
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        client.cursor.return_value = cursor
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchall.return_value = rows
        return store, cursor

    def test_returns_output_data_list(self):
        rows = [("ABCDEF1234567890", "hello world", _bson_json({"user_id": "alice"}), 0.5)]
        store, cursor = self._setup(rows)
        results = store.search(query="hello", vectors=[[0.1, 0.2, 0.3, 0.4]], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, OutputData)
        assert r.id == "ABCDEF1234567890"
        assert r.payload == {"user_id": "alice"}
        assert r.score == pytest.approx(1.0 / 1.5)  # EUCLIDEAN: 1/(1+0.5)

    def test_search_scores_ordered_closest_first(self):
        """Lower distance → higher score → first in list."""
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
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        client.cursor.return_value = cursor
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

    def test_search_null_text_treated_as_empty_string(self):
        """text column NULL must not crash the parser."""
        rows = [("ID", None, _bson_json({}), 0.2)]
        store, cursor = self._setup(rows)
        results = store.search("q", [[0.1, 0.2, 0.3, 0.4]])
        assert results[0].id == "ID"


# ===========================================================================
# delete unit tests
# ===========================================================================


class TestDelete:
    def test_delete_with_raw_id_hashes_it(self):
        store, _, cursor = _store()
        store.delete("my-raw-id")
        sql, params = cursor.execute.call_args_list[0][0]
        assert "DELETE FROM" in sql
        assert params == [_hash_id("my-raw-id")]

    def test_delete_with_already_hashed_id_uses_it_as_is(self):
        store, _, cursor = _store()
        hid = "ABCDEF1234567890"
        store.delete(hid)
        _, params = cursor.execute.call_args_list[0][0]
        assert params == [hid]

    def test_delete_commits(self):
        store, _, cursor = _store()
        store.delete("x")
        assert any("COMMIT" in str(c) for c in cursor.execute.call_args_list)

    def test_delete_sql_targets_correct_table(self):
        store, _, cursor = _store(collection_name="MY_TABLE")
        store.delete("abc")
        sql = cursor.execute.call_args_list[0][0][0]
        assert "MY_TABLE" in sql


# ===========================================================================
# update unit tests
# ===========================================================================


class TestUpdate:
    def test_update_vector_only(self):
        store, _, cursor = _store()
        store.update("ABCDEF1234567890", vector=[0.9, 0.8, 0.7, 0.6])
        sql = cursor.execute.call_args_list[0][0][0]
        assert "UPDATE" in sql and "VECTOR(" in sql

    def test_update_payload_only(self):
        store, _, cursor = _store()
        store.update("ABCDEF1234567890", payload={"user_id": "charlie"})
        sql = cursor.execute.call_args_list[0][0][0]
        assert "UPDATE" in sql and "JSON2BSON" in sql

    def test_update_both_vector_and_payload(self):
        store, _, cursor = _store()
        store.update("ABCDEF1234567890", vector=[0.1, 0.2, 0.3, 0.4], payload={"k": "v"})
        sql = cursor.execute.call_args_list[0][0][0]
        assert "VECTOR(" in sql and "JSON2BSON" in sql

    def test_update_noop_when_nothing_provided(self):
        store, _, cursor = _store()
        store.update("ABCDEF1234567890")
        cursor.execute.assert_not_called()

    def test_update_commits(self):
        store, _, cursor = _store()
        store.update("ABCDEF1234567890", payload={"k": "v"})
        assert any("COMMIT" in str(c) for c in cursor.execute.call_args_list)

    def test_update_hashes_raw_id(self):
        store, _, cursor = _store()
        store.update("raw-id", payload={"k": "v"})
        sql, params = cursor.execute.call_args_list[0][0]
        assert params[-1] == _hash_id("raw-id")


# ===========================================================================
# get unit tests
# ===========================================================================


class TestGet:
    def _make_store(self, row):
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        client.cursor.return_value = cursor
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchone.return_value = row
        return store, cursor

    def test_get_returns_output_data(self):
        store, cursor = self._make_store(
            ("ABCDEF1234567890", "my text", _bson_json({"user_id": "alice"}))
        )
        result = store.get("ABCDEF1234567890")
        assert result is not None
        assert result.id == "ABCDEF1234567890"
        assert result.payload == {"user_id": "alice"}
        assert result.score is None  # get() does not compute a score

    def test_get_returns_none_for_missing_id(self):
        store, _ = self._make_store(None)
        assert store.get("does-not-exist") is None

    def test_get_hashes_raw_id(self):
        store, cursor = self._make_store(None)
        store.get("raw-id")
        _, params = cursor.execute.call_args[0]
        assert params == [_hash_id("raw-id")]

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
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [("TABLE_A",), ("TABLE_B",)]
        client.cursor.return_value = cursor
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
        store, _, cursor = _store()
        store.delete_col()
        sqls = [c[0][0] for c in cursor.execute.call_args_list]
        assert any("DROP TABLE" in s for s in sqls)

    def test_skip_drop_when_table_missing(self):
        """If _table_exists returns False, no DROP TABLE must be issued."""
        client = MagicMock()
        cursor = MagicMock()
        client.cursor.return_value = cursor
        cursor.execute.side_effect = Exception("SQL0204N table not found")

        store = Db2VectorStore.__new__(Db2VectorStore)
        store.client = client
        store.collection_name = "GONE"
        store._text_field = "text"
        store._id_field = "id"
        store._metadata_field = "metadata"
        store._embedding_field = "embedding"
        store._distance_strategy = "EUCLIDEAN"
        store._embedding_dim = DIM

        store.delete_col()
        assert not any("DROP TABLE" in str(c) for c in cursor.execute.call_args_list)


# ===========================================================================
# col_info unit tests
# ===========================================================================


class TestColInfo:
    def _make_store(self, row):
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        client.cursor.return_value = cursor
        store = Db2VectorStore(client=client, collection_name="T", embedding_model_dims=DIM)
        cursor.reset_mock()
        client.cursor.return_value = cursor
        cursor.fetchone.return_value = row
        return store

    def test_returns_dict_with_schema_table_and_row_count(self):
        store = self._make_store(("MYSCHEMA", "MEM0_TEST", 42))
        info = store.col_info()
        assert info == {"schema": "MYSCHEMA", "table_name": "MEM0_TEST", "row_count": 42}

    def test_raises_when_not_found(self):
        store = self._make_store(None)
        with pytest.raises(ValueError, match="not found"):
            store.col_info()

    def test_queries_syscat_with_table_name(self):
        store, _, cursor = _store(collection_name="MY_COL")
        cursor.fetchone.return_value = ("S", "MY_COL", 0)
        store.col_info()
        sql = cursor.execute.call_args[0][0]
        assert "SYSCAT" in sql.upper()


# ===========================================================================
# list unit tests
# ===========================================================================


class TestList:
    def _make_store(self, rows):
        client = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        client.cursor.return_value = cursor
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
        drop_issued = []

        def _exec(sql, *a, **kw):
            executed_sqls.append(sql)
            if "SELECT COUNT(*)" in sql and drop_issued:
                raise Exception("SQL0204N table not found")
            if "DROP TABLE" in sql:
                drop_issued.append(True)

        cursor.execute.side_effect = _exec
        store.reset()

        assert any("DROP TABLE" in s for s in executed_sqls), "Expected DROP TABLE"
        assert any("CREATE TABLE" in s for s in executed_sqls), "Expected CREATE TABLE"


# ===========================================================================
# _where_clause unit tests
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
                    "database": "TESTDB",
                    "host": "localhost",
                    "port": 50000,
                    "username": "db2user",
                    "password": "secret",
                },
                "embedding_model_dims": 128,
                "distance_strategy": "cosine",
            },
        )
        assert isinstance(cfg.config, Db2Config)
        assert cfg.config.distance_strategy == "COSINE"


# ===========================================================================
# Live integration tests (require real Db2 — skipped without credentials)
# ===========================================================================


@requires_db2_credentials
def test_live_create_col(db2_store: Db2VectorStore):
    """Table must appear in list_cols after construction."""
    tables = [t.upper() for t in db2_store.list_cols()]
    assert db2_store.collection_name.upper() in tables


@requires_db2_credentials
def test_live_insert_and_get(db2_store: Db2VectorStore):
    vecs = [[0.1] * INTEGRATION_DIM, [0.2] * INTEGRATION_DIM]
    payloads = [{"name": "vec1", "user_id": "alice"}, {"name": "vec2", "user_id": "bob"}]
    ids = db2_store.insert(vectors=vecs, payloads=payloads)
    assert len(ids) == 2

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
    # Closest match should be the positive vector
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
def test_live_search_with_single_filter(db2_store: Db2VectorStore):
    vec = [0.7] * INTEGRATION_DIM
    db2_store.insert([vec, vec], payloads=[{"user_id": "alice"}, {"user_id": "bob"}])
    results = db2_store.search("unused", vectors=[vec], top_k=5, filters={"user_id": "alice"})
    assert all(r.payload.get("user_id") == "alice" for r in results)


@requires_db2_credentials
def test_live_search_with_no_filters(db2_store: Db2VectorStore):
    vec = [0.33] * INTEGRATION_DIM
    db2_store.insert([vec], payloads=[{"k": "v"}])
    results = db2_store.search("unused", vectors=[vec], top_k=1, filters=None)
    assert len(results) == 1


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
    ids = db2_store.insert([vec], payloads=[{"name": "old"}])
    target_id = ids[0]

    db2_store.update(vector_id=target_id, vector=[0.02] * INTEGRATION_DIM, payload={"name": "new"})

    got = db2_store.get(vector_id=target_id)
    assert got is not None
    assert got.payload.get("name") == "new"


@requires_db2_credentials
def test_live_reset_recreates_empty_table(db2_store: Db2VectorStore):
    vec = [0.15] * INTEGRATION_DIM
    ids = db2_store.insert([vec], payloads=[{"name": "before"}])
    before_id = ids[0]
    assert db2_store.get(before_id) is not None

    db2_store.reset()

    assert db2_store.get(before_id) is None
    assert db2_store.list(top_k=10) == [[]]

    # Table must be usable after reset
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
def test_live_list_returns_nested_output(db2_store: Db2VectorStore):
    db2_store.insert([[0.12] * INTEGRATION_DIM], payloads=[{"name": "nested"}])
    results = db2_store.list(top_k=10)
    assert isinstance(results, list) and isinstance(results[0], list)
    assert any(r.payload.get("name") == "nested" for r in results[0])


@requires_db2_credentials
def test_live_list_cols(db2_store: Db2VectorStore):
    tables = db2_store.list_cols()
    assert db2_store.collection_name.upper() in [t.upper() for t in tables]


@requires_db2_credentials
def test_live_col_info(db2_store: Db2VectorStore):
    info = db2_store.col_info()
    assert isinstance(info, dict)
    assert "schema" in info and "table_name" in info and "row_count" in info


@requires_db2_credentials
def test_live_delete_col(db2_store: Db2VectorStore):
    """Explicitly drop the table and verify it disappears from list_cols."""
    table = db2_store.collection_name.upper()
    assert table in [t.upper() for t in db2_store.list_cols()]

    db2_store.delete_col()

    assert table not in [t.upper() for t in db2_store.list_cols()]


@requires_db2_credentials
def test_live_documentation():
    """End-to-end smoke test via the mem0 Memory API.

    Mirrors the style of test_documentation in test_oracledb.py.
    Requires DB2_* env vars AND OPENAI_API_KEY.
    """
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
