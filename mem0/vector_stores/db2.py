"""IBM Db2 vector store for mem0."""

from __future__ import annotations

import functools
import json
import logging
import re
import math
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel

try:
    import ibm_db_dbi
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The 'ibm_db_dbi' library is required for the Db2 vector store. "
        "Install it with: pip install ibm_db"
    ) from exc

from mem0.configs.vector_stores.db2 import Db2Config
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# Minimum Db2 version for AI Vector Search.
_MIN_DB2_VERSION = (12, 1, 2)

# Minimum Db2 version for the native ANN vector index (CREATE VECTOR INDEX).
_MIN_ANN_VERSION = (12, 1, 5)

# HAMMING, MANHATTAN, and DOT are not supported by the Db2 ANN index.
_ANN_SUPPORTED_METRICS = {"COSINE", "EUCLIDEAN", "EUCLIDEAN_DISTANCE"}

# Db2's VECTOR_DISTANCE() only accepts "EUCLIDEAN" as the SQL keyword —
# "EUCLIDEAN_DISTANCE" raises SQL0104N. Map the alias before emitting SQL.
_SQL_METRIC = {
    "EUCLIDEAN_DISTANCE": "EUCLIDEAN",
}

_SCORE_FROM_DISTANCE = {
    "EUCLIDEAN":          lambda d: 1.0 / (1.0 + d),
    "EUCLIDEAN_DISTANCE": lambda d: 1.0 / (1.0 + d),
    "COSINE":             lambda d: max(0.0, 1.0 - d),
    "DOT":                lambda d: d,
    "HAMMING":            lambda d: 1.0 / (1.0 + d),
    "MANHATTAN":          lambda d: 1.0 / (1.0 + d),
}

_ORDER_BY_DIRECTION = {
    "EUCLIDEAN":          "ASC",
    "EUCLIDEAN_DISTANCE": "ASC",
    "COSINE":             "ASC",
    "DOT":                "DESC",
    "HAMMING":            "ASC",
    "MANHATTAN":          "ASC",
}

_LOGICAL_OPS = {
    "$and": "AND",
    "$or":  "OR",
    "$not": "NOT",
    "AND":  "AND",
    "OR":   "OR",
    "NOT":  "NOT",
}

_DUPLICATE_INDICATORS = (
    "sql0803n",
    "sqlstate=23505",
    "sqlcode=-803",
    "duplicate",
    "unique",
    "primary key",
)


def _distance_to_score(distance: float, strategy: str) -> float:
    fn = _SCORE_FROM_DISTANCE.get(strategy)
    if fn is None:
        raise ValueError(f"Unsupported distance strategy: '{strategy}'")
    return fn(distance)


def _handle_db_exceptions(func):
    """Decorator that re-raises DB and validation errors with context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, ValueError, TypeError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Db2VectorStore.{func.__name__} failed: {exc}"
            ) from exc
    return wrapper


def _normalize_filter_value(value: Any) -> str:
    """Normalise a Python value for inlining in a SQL string literal.

    Booleans in JSON are stored as ``true``/``false`` text — Python ``True``
    would otherwise be stringified as ``"True"`` (capital T) which never
    matches.  Also escapes single quotes per the SQL standard.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("'", "''")


def _is_iso_date(value: Any) -> bool:
    """Return True if *value* is a string Python recognises as ISO-8601 datetime."""
    if not isinstance(value, str):
        return False
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        datetime.fromisoformat(normalized)
        return True
    except ValueError:
        return False


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict[str, Any]]


def _table_exists(client: Any, table_name: str) -> bool:
    """Check table existence via SYSCAT.TABLES — no data scan required."""
    bare = table_name.strip('"').upper()
    sql = (
        "SELECT COUNT(*) FROM SYSCAT.TABLES "  # noqa: S608
        "WHERE TABNAME = ? AND TABSCHEMA = CURRENT SCHEMA"
    )
    cursor = client.cursor()
    try:
        cursor.execute(sql, [bare])
        row = cursor.fetchone()
        return bool(row and row[0] > 0)
    finally:
        cursor.close()


def _create_table_if_not_exists(
    client: Any,
    table_name: str,
    embedding_dim: int,
    text_field: str,
    id_field: str,
    metadata_field: str,
    embedding_field: str,
) -> None:
    if _table_exists(client, table_name):
        logger.info("Table %s already exists.", table_name)
        return

    cols = (
        f"{id_field} VARCHAR(36) PRIMARY KEY NOT NULL, "
        f"{text_field} CLOB, "
        f"{metadata_field} BLOB, "
        f"{embedding_field} VECTOR({embedding_dim}, FLOAT32)"
    )
    ddl = f"CREATE TABLE {table_name} ({cols})"
    cursor = client.cursor()
    try:
        cursor.execute(ddl)
        client.commit()
        logger.info("Table %s created.", table_name)
    except Exception:
        client.rollback()
        raise
    finally:
        cursor.close()


def _validate_embedding(embedding: Any, allow_none: bool = True) -> None:
    """Validate an embedding vector's type and contents.

    Args:
        embedding: Value to validate.
        allow_none: When ``True`` (default) ``None`` is accepted silently.

    Raises:
        ValueError: If the embedding is ``None`` when ``allow_none=False``,
            or is an empty list.
        TypeError: If the embedding is not a ``list``, or contains non-numeric
            values.
    """
    if embedding is None:
        if not allow_none:
            raise ValueError("Embedding cannot be None.")
        return
    if not isinstance(embedding, list):
        raise TypeError(f"Embedding must be a list, got {type(embedding).__name__}.")
    if len(embedding) == 0:
        raise ValueError("Embedding cannot be empty.")
    if not all(isinstance(x, (int, float)) for x in embedding):
        raise TypeError("All embedding values must be numeric (int or float).")


class Db2VectorStore(VectorStoreBase):
    """IBM Db2 AI Vector Search vector store.

    Supported ``distance_strategy`` values:
    ``"EUCLIDEAN"`` (default), ``"COSINE"``, ``"DOT"``,
    ``"EUCLIDEAN_DISTANCE"``, ``"HAMMING"``, ``"MANHATTAN"``.

    Args:
        collection_name: Db2 table name (created automatically if absent).
        embedding_model_dims: Dimensionality of the embedding vectors.
        client: Existing ``ibm_db_dbi.Connection`` (takes priority over
            ``connection_params``).
        connection_params: Dict with keys ``database``, ``host``, ``port``,
            ``username``, ``password`` and optionally ``security`` / ``ssl_cert``.
            When supplied without ``client``, a fresh connection is created
            via ``ibm_db_dbi.connect()`` giving this instance an isolated handle.
        distance_strategy: Distance function — ``"EUCLIDEAN"`` (default),
            ``"COSINE"``, ``"DOT"``, ``"EUCLIDEAN_DISTANCE"``,
            ``"HAMMING"``, or ``"MANHATTAN"``.
        db_schema: Optional Db2 schema name.  When set, ``SET SCHEMA <name>`` is
            issued immediately after connecting so all unqualified table
            references resolve to that schema.
        use_vector_index: When ``True`` and the Db2 server is 12.1.5+, create
            a native ANN vector index for approximate nearest-neighbour search.
            Only compatible with ``COSINE``, ``EUCLIDEAN``, and
            ``EUCLIDEAN_DISTANCE`` (validated at config time).  Defaults to
            ``False`` (exact scan, works on all versions ≥ 12.1.2).

            **Production deployments only.**  Requires Db2 12.1.5+
            Standard/Advanced Edition or Db2 on IBM Cloud/watsonx.data.
            Do **not** use with Db2 Community Edition (CE) containers
            (Podman/Docker) — CE drops TCP connections after the DDL due to
            in-memory ANN graph reconstruction, causing connection failures.
            Keep ``use_vector_index=False`` (the default) on CE containers.
        text_field: Column name for raw text (default ``"text"``).
        id_field: Column name for the primary key (default ``"id"``).
        metadata_field: Column name for JSON metadata (default ``"metadata"``).
        embedding_field: Column name for the stored vector (default ``"embedding"``).
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = Db2Config(**kwargs)

        # Pre-built client takes priority (test / advanced usage).
        # Otherwise build a fresh connection so each instance owns an isolated handle.
        if self.config.client is not None:
            self.client = self.config.client
        else:
            self.client = self._build_connection()

        # Version check — fail fast if Db2 is too old for AI Vector Search.
        self._db2_version: Optional[Tuple[int, int, int]] = None
        self._check_db2_version()

        self.collection_name = self.config.collection_name
        self._text_field = self.config.text_field
        self._id_field = self.config.id_field
        self._metadata_field = self.config.metadata_field
        self._embedding_field = self.config.embedding_field
        self._distance_strategy = self.config.distance_strategy
        self._embedding_dim = self.config.embedding_model_dims

        # Probe once at startup whether Db2 Text Search is installed.
        self._text_search_available: bool = self._probe_text_search()

        # Ensure table exists up front so failures surface at construction time.
        _create_table_if_not_exists(
            self.client,
            self.collection_name,
            self._embedding_dim,
            self._text_field,
            self._id_field,
            self._metadata_field,
            self._embedding_field,
        )

        # Optionally create ANN vector index (requires 12.1.5+, opt-in via use_vector_index).
        self._maybe_create_vector_index(self.collection_name)

    def _build_connection(self) -> Any:
        """Build a fresh ``ibm_db_dbi.connect()`` connection.

        Each instance gets its own isolated handle — ``connect()`` (not
        ``pconnect()``) prevents DDL auto-commits from corrupting a shared
        pooled handle across instances.
        """
        cp = self.config.connection_params or {}
        conn_str = (
            f"DATABASE={cp.get('database')};"
            f"HOSTNAME={cp.get('host')};"
            f"PORT={cp.get('port', 50000)};"
            f"PROTOCOL=TCPIP;"
            f"UID={cp.get('username')};"
            f"PWD={cp.get('password')};"
            f"Authentication=SERVER;"
        )
        if "security" in cp:
            conn_str += f"SECURITY={cp['security']};"
            ssl_cert = cp.get("ssl_cert", "")
            if ssl_cert:
                conn_str += f"SSLServerCertificate={ssl_cert};"
        try:
            conn = ibm_db_dbi.connect(conn_str, "", "")
        except Exception as exc:
            safe = re.sub(r"PWD=[^;]*", "PWD=***", conn_str)
            raise ConnectionError(
                f"Db2 connection failed: {exc}  (conn={safe})"
            ) from exc

        if self.config.db_schema:
            cursor = conn.cursor()
            try:
                cursor.execute(f"SET SCHEMA {self.config.db_schema}")
                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise RuntimeError(
                    f"Failed to set schema '{self.config.db_schema}': {exc}"
                ) from exc
            finally:
                cursor.close()

        return conn

    @contextmanager
    def _get_cursor(self, commit: bool = False) -> Iterator[Any]:
        """Yield a cursor; commit or rollback on exit; always close the cursor.

        Args:
            commit: When ``True``, call ``self.client.commit()`` on success.
                    On any exception, ``rollback()`` is attempted before
                    re-raising.
        """
        cursor = self.client.cursor()
        try:
            yield cursor
            if commit:
                self.client.commit()
        except Exception:
            try:
                self.client.rollback()
            except Exception:
                pass
            raise
        finally:
            cursor.close()

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """Create (or verify existence of) a Db2 vector table."""
        _create_table_if_not_exists(
            self.client,
            name,
            vector_size,
            self._text_field,
            self._id_field,
            self._metadata_field,
            self._embedding_field,
        )
        self._maybe_create_vector_index(name)

    @_handle_db_exceptions
    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        upsert: bool = False,
    ) -> List[str]:
        """Insert vectors (with optional payloads / ids) into the table.

        Args:
            vectors: Embedding vectors to store.
            payloads: Optional list of metadata dicts (one per vector).
            ids: Optional list of string IDs. If omitted, UUIDs are generated.
            upsert: When ``True``, use ``MERGE INTO`` so existing rows are
                updated instead of raising a duplicate-key error.
                When ``False`` (default), a plain ``INSERT`` is used; a
                duplicate primary key raises ``ValueError``.

        Returns:
            List of stored IDs.
        """
        n = len(vectors)
        if payloads is None:
            payloads = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        for i, vec in enumerate(vectors):
            try:
                _validate_embedding(vec, allow_none=False)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"Invalid embedding at index {i}: {exc}") from exc

        embedding_len = len(vectors[0]) if vectors else self._embedding_dim

        rows = [
            (
                vid,
                "[" + ", ".join(str(v) for v in vec) + "]",
                json.dumps(meta),
                meta.get("data", ""),
            )
            for vid, vec, meta in zip(ids, vectors, payloads)
        ]

        if upsert:
            sql = (
                f"MERGE INTO {self.collection_name} AS t "  # noqa: S608
                f"USING (VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), "
                f"SYSTOOLS.JSON2BSON(?), ?)) "
                f"AS s({self._id_field}, {self._embedding_field}, "
                f"{self._metadata_field}, {self._text_field}) "
                f"ON t.{self._id_field} = s.{self._id_field} "
                f"WHEN MATCHED THEN UPDATE SET "
                f"t.{self._embedding_field} = s.{self._embedding_field}, "
                f"t.{self._metadata_field} = s.{self._metadata_field}, "
                f"t.{self._text_field} = s.{self._text_field} "
                f"WHEN NOT MATCHED THEN INSERT "
                f"({self._id_field}, {self._embedding_field}, "
                f"{self._metadata_field}, {self._text_field}) "
                f"VALUES (s.{self._id_field}, s.{self._embedding_field}, "
                f"s.{self._metadata_field}, s.{self._text_field})"
            )
            with self._get_cursor(commit=True) as cursor:
                for row in rows:
                    cursor.execute(sql, row)
        else:
            sql = (
                f"INSERT INTO {self.collection_name} "  # noqa: S608
                f"({self._id_field}, {self._embedding_field}, "
                f"{self._metadata_field}, {self._text_field}) "
                f"VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), SYSTOOLS.JSON2BSON(?), ?)"
            )
            try:
                with self._get_cursor(commit=True) as cursor:
                    cursor.executemany(sql, rows)
            except Exception as exc:
                err = str(exc).lower()
                if any(ind in err for ind in _DUPLICATE_INDICATORS):
                    raise ValueError(
                        f"Duplicate ID detected. Use upsert=True to overwrite "
                        f"existing records. Original error: {exc}"
                    ) from exc
                raise

        return ids

    @_handle_db_exceptions
    def insert_skip_duplicates(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors, silently skipping any IDs that already exist.

        Uses ``MERGE INTO … WHEN NOT MATCHED THEN INSERT`` — existing rows are
        left unchanged.

        Args:
            vectors: Embedding vectors to store.
            payloads: Optional list of metadata dicts (one per vector).
            ids: Optional list of string IDs. If omitted, UUIDs are generated.

        Returns:
            List of IDs that were actually inserted (subset of ``ids``).
        """
        n = len(vectors)
        if payloads is None:
            payloads = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        for i, vec in enumerate(vectors):
            try:
                _validate_embedding(vec, allow_none=False)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"Invalid embedding at index {i}: {exc}") from exc

        embedding_len = len(vectors[0]) if vectors else self._embedding_dim

        sql = (
            f"MERGE INTO {self.collection_name} AS t "  # noqa: S608
            f"USING (VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), "
            f"SYSTOOLS.JSON2BSON(?), ?)) "
            f"AS s({self._id_field}, {self._embedding_field}, "
            f"{self._metadata_field}, {self._text_field}) "
            f"ON t.{self._id_field} = s.{self._id_field} "
            f"WHEN NOT MATCHED THEN INSERT "
            f"({self._id_field}, {self._embedding_field}, "
            f"{self._metadata_field}, {self._text_field}) "
            f"VALUES (s.{self._id_field}, s.{self._embedding_field}, "
            f"s.{self._metadata_field}, s.{self._text_field})"
        )

        inserted: List[str] = []
        with self._get_cursor(commit=True) as cursor:
            for vid, vec, meta in zip(ids, vectors, payloads):
                row = (
                    vid,
                    "[" + ", ".join(str(v) for v in vec) + "]",
                    json.dumps(meta),
                    meta.get("data", ""),
                )
                cursor.execute(sql, row)
                if cursor.rowcount > 0:
                    inserted.append(vid)

        return inserted

    @_handle_db_exceptions
    def search(
        self,
        query: str,
        vectors: List[list],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """Search for the *top_k* nearest vectors."""
        if vectors and isinstance(vectors[0], (int, float)):
            embedding = vectors
        else:
            embedding = vectors[0] if vectors else []
        embedding_len = len(embedding) if embedding else self._embedding_dim

        embedding_str = "[" + ", ".join(str(v) for v in embedding) + "]"
        where_clause, filter_params = self._where_clause(filters)
        order_dir = _ORDER_BY_DIRECTION[self._distance_strategy]
        sql_metric = _SQL_METRIC.get(self._distance_strategy, self._distance_strategy)

        null_check = f"{self._embedding_field} IS NOT NULL"
        if where_clause:
            full_where = f"{where_clause} AND {null_check}"
        else:
            full_where = f"WHERE {null_check}"

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"VECTOR_DISTANCE({self._embedding_field}, "
            f"VECTOR('{embedding_str}', {embedding_len}, FLOAT32), "
            f"{sql_metric}) AS distance "
            f"FROM {self.collection_name} "
            f"{full_where} "
            f"ORDER BY distance {order_dir} "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql, filter_params) if filter_params else cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as exc:
            # SQL0801N: COSINE on a zero-vector causes division by zero.
            err = str(exc)
            if "SQL0801N" in err or "Division by zero" in err:
                logger.debug("search() returned empty — SQL0801N (zero-vector): %s", exc)
                return []
            raise

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            score = _distance_to_score(row[3], self._distance_strategy)
            results.append(OutputData(id=row[0], score=score, payload=metadata))
        return results

    @_handle_db_exceptions
    def delete(self, vector_id: str) -> None:
        """Delete a single vector by ID."""
        sql = f"DELETE FROM {self.collection_name} WHERE {self._id_field} = ?"  # noqa: S608
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(sql, [vector_id])

    @_handle_db_exceptions
    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """Update the embedding and/or payload of an existing record."""
        if vector is None and payload is None:
            return

        if vector is not None:
            _validate_embedding(vector, allow_none=False)

        set_parts = []
        params: List[Any] = []

        if vector is not None:
            embedding_len = len(vector)
            vec_str = "[" + ", ".join(str(v) for v in vector) + "]"
            set_parts.append(
                f"{self._embedding_field} = "
                f"VECTOR('{vec_str}', {embedding_len}, FLOAT32)"
            )

        if payload is not None:
            set_parts.append(f"{self._text_field} = ?")
            params.append(payload.get("data", ""))
            set_parts.append(f"{self._metadata_field} = SYSTOOLS.JSON2BSON(?)")
            params.append(json.dumps(payload))

        params.append(vector_id)
        sql = (
            f"UPDATE {self.collection_name} "  # noqa: S608
            f"SET {', '.join(set_parts)} "
            f"WHERE {self._id_field} = ?"
        )

        with self._get_cursor(commit=True) as cursor:
            cursor.execute(sql, params)

    @_handle_db_exceptions
    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a single record by ID."""
        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}) "
            f"FROM {self.collection_name} "
            f"WHERE {self._id_field} = ?"
        )

        with self._get_cursor() as cursor:
            cursor.execute(sql, [vector_id])
            row = cursor.fetchone()

        if row is None:
            return None
        metadata = json.loads(row[2] if row[2] is not None else "{}")
        return OutputData(id=row[0], score=None, payload=metadata)

    @_handle_db_exceptions
    def list_cols(self) -> List[str]:
        """Return the names of all user tables in the current schema."""
        sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TYPE = 'T' AND TABSCHEMA = CURRENT SCHEMA"  # noqa: S608
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    @_handle_db_exceptions
    def delete_col(self) -> None:
        """Drop the collection table if it exists."""
        if not _table_exists(self.client, self.collection_name):
            logger.info("Table %s not found; nothing to drop.", self.collection_name)
            return
        # DROP TABLE is DDL — auto-commits in Db2.  Commit any open transaction
        # first, execute the DDL, commit again while the cursor is still open.
        self.client.commit()
        drop_cursor = self.client.cursor()
        try:
            drop_cursor.execute(f"DROP TABLE {self.collection_name}")
            self.client.commit()
        except Exception:
            try:
                self.client.rollback()
            except Exception:
                pass
            raise
        finally:
            drop_cursor.close()
        logger.info("Table %s dropped.", self.collection_name)

    @_handle_db_exceptions
    def col_info(self) -> Dict[str, Any]:
        """Return metadata about the collection table.

        Uses a live ``COUNT(*)`` scalar subquery for ``row_count`` —
        ``SYSCAT.TABLES.CARD`` returns ``-1`` until ``RUNSTATS`` is run.
        Both the catalog lookup and the row count are fetched in a single
        SQL round-trip to minimise latency.

        Returns:
            Dict with keys ``schema``, ``table_name``, ``row_count``,
            ``embedding_model_dims``, and ``distance_strategy``.
            The last two are in-memory values requiring no extra SQL.
        """
        sql = (  # noqa: S608
            "SELECT TABSCHEMA, TABNAME, "
            f"(SELECT COUNT(*) FROM {self.collection_name}) AS row_count "
            "FROM SYSCAT.TABLES "
            "WHERE TABNAME = UPPER(?) AND TABSCHEMA = CURRENT SCHEMA"
        )
        with self._get_cursor() as cursor:
            cursor.execute(sql, [self.collection_name.strip('"')])
            row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Collection '{self.collection_name}' not found.")

        return {
            "schema": row[0],
            "table_name": row[1],
            "row_count": row[2],
            "embedding_model_dims": self._embedding_dim,
            "distance_strategy": self._distance_strategy,
        }

    @_handle_db_exceptions
    def list(
        self,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = 100,
    ) -> List[List[OutputData]]:
        """List records in the collection."""
        where_clause, filter_params = self._where_clause(filters)
        limit = f"FETCH FIRST {top_k} ROWS ONLY" if top_k is not None else ""

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}) "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"{limit}"
        )

        with self._get_cursor() as cursor:
            cursor.execute(sql, filter_params) if filter_params else cursor.execute(sql)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            results.append(OutputData(id=row[0], score=None, payload=metadata))
        return [results]

    @_handle_db_exceptions
    def reset(self) -> None:
        """Drop and recreate the collection table."""
        logger.warning("Resetting collection %s …", self.collection_name)
        self.delete_col()
        _create_table_if_not_exists(
            self.client,
            self.collection_name,
            self._embedding_dim,
            self._text_field,
            self._id_field,
            self._metadata_field,
            self._embedding_field,
        )
        self._maybe_create_vector_index(self.collection_name)

    @_handle_db_exceptions
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Optional[List[OutputData]]:
        """Full-text keyword search using Db2 Text Search (``CONTAINS()``).

        Searches the ``text`` column.  Falls back to ``None`` (which
        triggers mem0's semantic-only fallback) when:

        * Db2 Text Search addon is not installed/configured (detected at
          startup by :meth:`_probe_text_search`).
        * The Text Search index does not exist on ``text`` yet.

        To enable keyword search, create a Text Search index on the
        ``text`` column::

            CALL SYSPROC.SYSTS_CREATE(
                CURRENT SCHEMA, '<TABLE>', 'text',
                'MAXIMUM CHARACTERS 10000 LANGUAGE EN FORMAT NONE'
            );

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of :class:`OutputData` ordered by relevance score descending,
            or ``None`` if Db2 Text Search is not available.
        """
        if not self._text_search_available:
            return None

        esc_query = self._escape_literal(query)
        where_clause, filter_params = self._where_clause(filters)

        if where_clause:
            text_pred = (
                f"AND CONTAINS({self._text_field}, '{esc_query}') = 1"
            )
        else:
            text_pred = (
                f"WHERE CONTAINS({self._text_field}, '{esc_query}') = 1"
            )

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"SCORE({self._text_field}, '{esc_query}') AS relevance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"{text_pred} "
            f"ORDER BY relevance DESC "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql, filter_params) if filter_params else cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as exc:
            logger.debug(
                "keyword_search() fell back to None (Text Search unavailable): %s", exc
            )
            return None

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            score = float(row[3]) / 100.0  # normalise 0–100 → 0.0–1.0
            results.append(OutputData(id=row[0], score=score, payload=metadata))
        return results

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self.client.close()
            logger.debug("Db2 connection closed.")
        except Exception:
            pass

    def __del__(self) -> None:
        """Best-effort connection cleanup on garbage collection."""
        try:
            self.client.close()
        except Exception:
            pass

    def _probe_text_search(self) -> bool:
        """Return ``True`` if Db2 Text Search is active on this database."""
        sql = "SELECT CONTAINS(v, 'probe') FROM (VALUES ('probe text')) AS t(v)"  # noqa: S608
        available = False
        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql)
                cursor.fetchone()
                available = True
        except Exception:
            available = False

        if available:
            logger.info(
                "Db2 Text Search detected — keyword_search() enabled for %s.",
                self.collection_name,
            )
        else:
            logger.debug(
                "Db2 Text Search not installed/configured — keyword_search() will "
                "return None (mem0 will use semantic-only search)."
            )
        return available

    def _check_db2_version(self) -> None:
        """Raise ``RuntimeError`` if the connected Db2 is below ``_MIN_DB2_VERSION``."""
        sql = "SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO"  # noqa: S608
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()

        if row is None:
            logger.warning("Could not determine Db2 version — proceeding anyway.")
            return

        raw = str(row[0])
        m = re.search(r"v?(\d+)\.(\d+)\.(\d+)", raw)
        if m is None:
            logger.warning("Could not parse Db2 version from %r — proceeding anyway.", raw)
            return

        actual: Tuple[int, int, int] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        self._db2_version = actual

        if actual < _MIN_DB2_VERSION:
            req = ".".join(str(x) for x in _MIN_DB2_VERSION)
            act = ".".join(str(x) for x in actual)
            raise RuntimeError(
                f"IBM Db2 {req}+ is required for AI Vector Search "
                f"(connected server reports {act}).  "
                f"Please upgrade your Db2 instance."
            )
        logger.debug("Db2 version check passed: %s", raw.strip())

    def _maybe_create_vector_index(self, table_name: str) -> None:
        """Create a native ANN vector index when ``use_vector_index=True``."""
        if not self.config.use_vector_index:
            return

        if self._db2_version is None or self._db2_version < _MIN_ANN_VERSION:
            req = ".".join(str(x) for x in _MIN_ANN_VERSION)
            actual_str = (
                ".".join(str(x) for x in self._db2_version)
                if self._db2_version else "unknown"
            )
            logger.warning(
                "use_vector_index=True requires Db2 %s+ (server is %s). "
                "Falling back to exact scan — set use_vector_index=False to silence this.",
                req, actual_str,
            )
            return

        if self._distance_strategy not in _ANN_SUPPORTED_METRICS:
            return

        bare_name = table_name.strip('"')
        idx_name = f"{bare_name}_vec_idx"
        sql_metric = _SQL_METRIC.get(self._distance_strategy, self._distance_strategy)
        ddl = (
            f"CREATE VECTOR INDEX {idx_name} "
            f"ON {table_name}({self._embedding_field}) "
            f"WITH DISTANCE {sql_metric}"
        )
        try:
            with self._get_cursor(commit=True) as cursor:
                cursor.execute(ddl)
            logger.info(
                "Vector index %s created on %s (%s).",
                idx_name, table_name, sql_metric,
            )
        except Exception as exc:
            logger.warning(
                "Could not create vector index on %s: %s. "
                "If running on Db2 Community Edition, this is likely a memory "
                "resource constraint — increase the container memory limit "
                "(e.g. --memory=4g), then manually restart the Db2 instance or "
                "container and reconnect.  Alternatively, set "
                "use_vector_index=False to use exact scan (works on all versions).",
                table_name, exc,
            )

    @staticmethod
    def _escape_literal(value: str) -> str:
        """Escape a string for safe inline use in a SQL string literal.

        Only used for CONTAINS()/SCORE() in keyword_search() — ibm_db cannot
        bind ``?`` parameters inside Text Search predicates.
        """
        return str(value).replace("'", "''")

    def _where_clause(self, filters: Optional[Dict[str, Any]]) -> "Tuple[str, List[Any]]":
        """Build a WHERE clause from a filter dict.

        Returns a ``(sql_fragment, params)`` tuple for use with
        ``cursor.execute(sql, params)``.

        Field values are bound with parameterized ``?`` placeholders inside
        ``JSON_VALUE(… RETURNING VARCHAR(1000))``.

        Supports flat equality, operator dicts (eq/ne/gt/gte/lt/lte/in/nin/
        contains/icontains), wildcard ``"*"``, list shorthand (→ IN), and
        compound logical keys ``AND`` / ``OR`` / ``NOT`` (also ``$and`` /
        ``$or`` / ``$not``).

        Args:
            filters: Filter dict as passed by the mem0 Memory layer.

        Returns:
            ``(sql_fragment, params)`` where ``sql_fragment`` starts with
            ``"WHERE "`` or is ``""`` when no filters apply.
        """
        if not filters:
            return "", []
        params: List[Any] = []
        conditions = self._build_conditions(filters, params)
        sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return sql, params

    def _build_conditions(self, filters: Dict[str, Any], params: List[Any]) -> List[str]:
        """Recursively translate a filter dict into SQL condition strings."""
        conditions: List[str] = []

        for key, value in filters.items():
            logical_op = _LOGICAL_OPS.get(key)
            if logical_op is not None:
                if logical_op == "NOT":
                    if not isinstance(value, list):
                        value = [value]
                    sub_parts: List[str] = []
                    for sub in value:
                        sub_conds = self._build_conditions(sub, params)
                        if sub_conds:
                            sub_parts.append("(" + " AND ".join(sub_conds) + ")")
                    if sub_parts:
                        conditions.append("NOT (" + " OR ".join(sub_parts) + ")")
                else:
                    if not isinstance(value, list):
                        value = [value]
                    sub_parts = []
                    for sub in value:
                        sub_conds = self._build_conditions(sub, params)
                        if sub_conds:
                            sub_parts.append("(" + " AND ".join(sub_conds) + ")")
                    if sub_parts:
                        joiner = " AND " if logical_op == "AND" else " OR "
                        conditions.append("(" + joiner.join(sub_parts) + ")")
                continue

            mf = self._metadata_field
            json_expr = (
                f"JSON_VALUE(SYSTOOLS.BSON2JSON({mf}), '$.{key}' RETURNING VARCHAR(1000))"
            )

            if value == "*":
                continue

            if isinstance(value, dict):
                for op, op_val in value.items():
                    cond = self._op_condition(json_expr, op, op_val, params)
                    if cond:
                        conditions.append(cond)

            elif isinstance(value, list):
                placeholders = ", ".join("?" for _ in value)
                params.extend(_normalize_filter_value(v) for v in value)
                conditions.append(f"{json_expr} IN ({placeholders})")

            else:
                if value is None:
                    conditions.append(f"{json_expr} IS NULL")
                else:
                    params.append(_normalize_filter_value(value))
                    conditions.append(f"({json_expr} IS NOT NULL AND {json_expr} = ?)")

        return conditions

    def _op_condition(
        self, json_expr: str, op: str, value: Any, params: List[Any]
    ) -> str:
        """Translate a single operator dict entry into a SQL condition fragment.

        Supported operators: eq, ne, gt, gte, lt, lte, in, nin,
        contains, icontains.  ``ne`` and ``nin`` are NULL-safe.
        ISO date strings pass through as VARCHAR for range comparisons.
        """
        op = op.lower()

        if op == "eq":
            if value is None:
                return f"{json_expr} IS NULL"
            params.append(_normalize_filter_value(value))
            return f"({json_expr} IS NOT NULL AND {json_expr} = ?)"
        if op == "ne":
            if value is None:
                return f"{json_expr} IS NOT NULL"
            # NULL-safe !=: rows where the field is absent also match.
            params.append(_normalize_filter_value(value))
            return f"({json_expr} IS NULL OR {json_expr} <> ?)"
        if op == "gt":
            if _is_iso_date(value):
                params.append(_normalize_filter_value(value))
                return f"{json_expr} > ?"
            return f"CAST({json_expr} AS DOUBLE) > {float(value)}"
        if op == "gte":
            if _is_iso_date(value):
                params.append(_normalize_filter_value(value))
                return f"{json_expr} >= ?"
            return f"CAST({json_expr} AS DOUBLE) >= {float(value)}"
        if op == "lt":
            if _is_iso_date(value):
                params.append(_normalize_filter_value(value))
                return f"{json_expr} < ?"
            return f"CAST({json_expr} AS DOUBLE) < {float(value)}"
        if op == "lte":
            if _is_iso_date(value):
                params.append(_normalize_filter_value(value))
                return f"{json_expr} <= ?"
            return f"CAST({json_expr} AS DOUBLE) <= {float(value)}"
        if op == "in":
            if not isinstance(value, list):
                raise ValueError(
                    f"Filter operator 'in' requires a list, got {type(value).__name__}"
                )
            placeholders = ", ".join("?" for _ in value)
            params.extend(_normalize_filter_value(v) for v in value)
            return f"{json_expr} IN ({placeholders})"
        if op == "nin":
            if not isinstance(value, list):
                raise ValueError(
                    f"Filter operator 'nin' requires a list, got {type(value).__name__}"
                )
            placeholders = ", ".join("?" for _ in value)
            params.extend(_normalize_filter_value(v) for v in value)
            return f"({json_expr} IS NULL OR {json_expr} NOT IN ({placeholders}))"
        if op == "contains":
            params.append(f"%{_normalize_filter_value(value)}%")
            return f"{json_expr} LIKE ?"
        if op == "icontains":
            params.append(f"%{_normalize_filter_value(value)}%")
            return f"LOWER({json_expr}) LIKE LOWER(?)"
        raise ValueError(
            f"Unsupported filter operator '{op}'. "
            f"Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"
        )
