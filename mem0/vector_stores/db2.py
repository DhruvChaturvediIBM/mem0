"""IBM Db2 vector store for mem0."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

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

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

_SCORE_FROM_DISTANCE = {
    "EUCLIDEAN":          lambda d: 1.0 / (1.0 + d),
    "EUCLIDEAN_DISTANCE": lambda d: 1.0 / (1.0 + d),
    "COSINE":             lambda d: max(0.0, 1.0 - d),
    "DOT":                lambda d: d,       # higher = more similar, return as-is
    "HAMMING":            lambda d: 1.0 / (1.0 + d),
    "MANHATTAN":          lambda d: 1.0 / (1.0 + d),
}

# DOT product: higher = more similar → ORDER BY DESC.
# All other metrics: lower distance = more similar → ORDER BY ASC.
_ORDER_BY_DIRECTION = {
    "EUCLIDEAN":          "ASC",
    "EUCLIDEAN_DISTANCE": "ASC",
    "COSINE":             "ASC",
    "DOT":                "DESC",
    "HAMMING":            "ASC",
    "MANHATTAN":          "ASC",
}

# Minimum Db2 version required for AI Vector Search.
# Vector Search was introduced in Db2 12.1.0; 12.1.2 added HAMMING/MANHATTAN.
_MIN_DB2_VERSION = (12, 1, 2)


def _distance_to_score(distance: float, strategy: str) -> float:
    fn = _SCORE_FROM_DISTANCE.get(strategy)
    if fn is None:
        raise ValueError(f"Unsupported distance strategy: '{strategy}'")
    return fn(distance)


# ---------------------------------------------------------------------------
# OutputData
# ---------------------------------------------------------------------------


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _table_exists(client: Any, table_name: str) -> bool:
    """Check table existence via SYSCAT.TABLES — no data scan required."""
    # Strip any schema prefix or quotes; SYSCAT.TABLES stores bare upper-case names.
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

    # id_field is VARCHAR(36) to hold standard 36-character UUID strings
    # produced by str(uuid.uuid4()) — no hashing or truncation needed.
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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


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
        distance_strategy: Distance function — ``"EUCLIDEAN"`` (default),
            ``"COSINE"``, or ``"DOT"``.
        text_field: Column name for raw text (default ``"text"``).
        id_field: Column name for the primary key (default ``"id"``).
        metadata_field: Column name for JSON metadata (default ``"metadata"``).
        embedding_field: Column name for the stored vector (default ``"embedding"``).
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = Db2Config(**kwargs)

        # Establish connection ------------------------------------------------
        if self.config.client is not None:
            self.client = self.config.client
        else:
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
                self.client = ibm_db_dbi.connect(conn_str, "", "")
            except Exception as exc:
                # Mask the password before surfacing the error so credentials
                # never appear in logs or tracebacks.
                safe = re.sub(r"PWD=[^;]*", "PWD=***", conn_str)
                raise ConnectionError(
                    f"Db2 connection failed: {exc}  (conn={safe})"
                ) from exc

        # Version check — fail fast if Db2 is too old for AI Vector Search. --
        self._check_db2_version()

        self.collection_name = self.config.collection_name
        self._text_field = self.config.text_field
        self._id_field = self.config.id_field
        self._metadata_field = self.config.metadata_field
        self._embedding_field = self.config.embedding_field
        self._distance_strategy = self.config.distance_strategy
        self._embedding_dim = self.config.embedding_model_dims

        # Probe once at startup whether Db2 Text Search is installed.
        # Result is cached on self._text_search_available so keyword_search()
        # pays zero overhead on every call.
        self._text_search_available: bool = self._probe_text_search()

        # Ensure table exists --------------------------------------------------
        _create_table_if_not_exists(
            self.client,
            self.collection_name,
            self._embedding_dim,
            self._text_field,
            self._id_field,
            self._metadata_field,
            self._embedding_field,
        )

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

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

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors (with optional payloads / ids) into the table.

        Args:
            vectors: Embedding vectors to store.
            payloads: Optional list of metadata dicts (one per vector).
            ids: Optional list of string IDs. If omitted, UUIDs are generated.

        Returns:
            List of stored (hashed) IDs.
        """
        n = len(vectors)
        if payloads is None:
            payloads = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

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

        sql = (
            f"INSERT INTO {self.collection_name} "  # noqa: S608
            f"({self._id_field}, {self._embedding_field}, "
            f"{self._metadata_field}, {self._text_field}) "
            f"VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), SYSTOOLS.JSON2BSON(?), ?)"
        )

        cursor = self.client.cursor()
        try:
            cursor.executemany(sql, rows)
            self.client.commit()
        except Exception:
            self.client.rollback()
            raise
        finally:
            cursor.close()

        return ids

    def search(
        self,
        query: str,
        vectors: List[list],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """Search for the *top_k* nearest vectors.

        Args:
            query: The original query text (unused for the SQL search but kept
                for API parity).
            vectors: Query embedding(s). The first element is used.
            top_k: Maximum number of results to return.
            filters: Optional equality filters applied as ``AND`` predicates on
                the JSON metadata column.

        Returns:
            List of :class:`OutputData` ordered by ascending distance
            (highest similarity first).
        """
        # Handle both list[list[float]] (our examples) and list[float] (Memory internal calls)
        if vectors and isinstance(vectors[0], (int, float)):
            embedding = vectors          # already a flat vector
        else:
            embedding = vectors[0] if vectors else []
        embedding_len = len(embedding) if embedding else self._embedding_dim

        embedding_str = "[" + ", ".join(str(v) for v in embedding) + "]"
        where_clause = self._where_clause(filters)
        order_dir = _ORDER_BY_DIRECTION[self._distance_strategy]

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"VECTOR_DISTANCE({self._embedding_field}, "
            f"VECTOR('{embedding_str}', {embedding_len}, FLOAT32), "
            f"{self._distance_strategy}) AS distance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"ORDER BY distance {order_dir} "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        cursor = self.client.cursor()
        results = []
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                metadata = json.loads(row[2] if row[2] is not None else "{}")
                distance = row[3]
                score = _distance_to_score(distance, self._distance_strategy)
                results.append(
                    OutputData(
                        id=row[0],
                        score=score,
                        payload=metadata,
                    )
                )
        finally:
            cursor.close()

        return results

    def delete(self, vector_id: str) -> None:
        """Delete a single vector by ID.

        Args:
            vector_id: The ID as originally passed to ``insert`` (UUID string).
        """
        sql = f"DELETE FROM {self.collection_name} WHERE {self._id_field} = ?"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(sql, [vector_id])
            self.client.commit()
        except Exception:
            self.client.rollback()
            raise
        finally:
            cursor.close()

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """Update the embedding and/or payload of an existing record.

        Args:
            vector_id: ID of the record to update (the original UUID).
            vector: New embedding vector. If ``None``, the embedding is not changed.
            payload: New metadata dict. If ``None``, the metadata is not changed.
        """
        if vector is None and payload is None:
            return

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
            # Always sync text column — fall back to "" when "data" key is absent
            # so the stored text never goes stale relative to the metadata.
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

        cursor = self.client.cursor()
        try:
            cursor.execute(sql, params)
            self.client.commit()
        except Exception:
            self.client.rollback()
            raise
        finally:
            cursor.close()

    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a single record by ID.

        Args:
            vector_id: The original UUID of the record.

        Returns:
            :class:`OutputData` if found, otherwise ``None``.
        """
        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}) "
            f"FROM {self.collection_name} "
            f"WHERE {self._id_field} = ?"
        )

        cursor = self.client.cursor()
        try:
            cursor.execute(sql, [vector_id])
            row = cursor.fetchone()
        finally:
            cursor.close()

        if row is None:
            return None

        metadata = json.loads(row[2] if row[2] is not None else "{}")
        return OutputData(id=row[0], score=None, payload=metadata)

    def list_cols(self) -> List[str]:
        """Return the names of all user tables in the current schema."""
        sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TYPE = 'T' AND TABSCHEMA = CURRENT SCHEMA"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [row[0] for row in rows]

    def delete_col(self) -> None:
        """Drop the collection table if it exists."""
        if not _table_exists(self.client, self.collection_name):
            logger.info("Table %s not found; nothing to drop.", self.collection_name)
            return
        cursor = self.client.cursor()
        try:
            cursor.execute(f"DROP TABLE {self.collection_name}")
            self.client.commit()
            logger.info("Table %s dropped.", self.collection_name)
        except Exception:
            self.client.rollback()
            raise
        finally:
            cursor.close()

    def col_info(self) -> Dict[str, Any]:
        """Return basic metadata about the collection table.

        Uses a live ``COUNT(*)`` for ``row_count`` — ``SYSCAT.TABLES.CARD``
        returns ``-1`` until ``RUNSTATS`` is run.

        Returns:
            Dict with keys ``schema``, ``table_name``, ``row_count``.
        """
        cat_sql = (
            "SELECT TABSCHEMA, TABNAME "  # noqa: S608
            "FROM SYSCAT.TABLES "
            "WHERE TABNAME = UPPER(?) AND TABSCHEMA = CURRENT SCHEMA"
        )
        cursor = self.client.cursor()
        try:
            cursor.execute(cat_sql, [self.collection_name.strip('"')])
            row = cursor.fetchone()
        finally:
            cursor.close()

        if row is None:
            raise ValueError(f"Collection '{self.collection_name}' not found.")

        schema, table_name = row[0], row[1]

        # Live count — avoids the stale -1 from SYSCAT.CARD
        count_sql = f"SELECT COUNT(*) FROM {self.collection_name}"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(count_sql)
            count_row = cursor.fetchone()
            row_count = count_row[0] if count_row else 0
        finally:
            cursor.close()

        return {
            "schema": schema,
            "table_name": table_name,
            "row_count": row_count,
        }

    def list(
        self,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = 100,
    ) -> List[List[OutputData]]:
        """List records in the collection.

        Args:
            filters: Optional equality filters on metadata fields.
            top_k: Maximum number of rows to return (default 100).

        Returns:
            A single-element list containing the list of :class:`OutputData`.
        """
        where_clause = self._where_clause(filters)
        limit = f"FETCH FIRST {top_k} ROWS ONLY" if top_k is not None else ""

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}) "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"{limit}"
        )

        cursor = self.client.cursor()
        results = []
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                metadata = json.loads(row[2] if row[2] is not None else "{}")
                results.append(OutputData(id=row[0], score=None, payload=metadata))
        finally:
            cursor.close()

        return [results]

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

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Optional[List[OutputData]]:
        """Full-text keyword search using Db2 Text Search (``CONTAINS()``).

        Only available when Db2 Text Search is installed **and** a text index
        exists on the collection table's ``text`` column.  If either condition
        is not met this method returns ``None`` — mem0 then falls back to
        semantic-only search automatically (same behaviour as Chroma, Faiss,
        Redis, etc.).

        A text index can be created with::

            CALL SYSPROC.SYSTS_CREATE(
                CURRENT SCHEMA, '<TABLE>', 'text',
                'MAXIMUM CHARACTERS 10000 LANGUAGE EN FORMAT NONE'
            );

        Args:
            query: The (lemmatized) search query text.
            top_k: Maximum number of results to return.
            filters: Optional equality filters on metadata fields.

        Returns:
            List of :class:`OutputData` ordered by relevance score descending,
            or ``None`` if Db2 Text Search is not available.
        """
        if not self._text_search_available:
            return None

        where_clause = self._where_clause(filters)
        # Prepend AND when _where_clause already produced a WHERE clause so
        # the CONTAINS() predicate joins correctly; start a fresh WHERE
        # when there are no metadata filters.
        if where_clause:
            text_pred = f"AND CONTAINS({self._text_field}, '{self._escape_literal(query)}') = 1"
        else:
            text_pred = f"WHERE CONTAINS({self._text_field}, '{self._escape_literal(query)}') = 1"

        # SCORE() returns Db2 Text Search relevance rank (0–100 float).
        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"SCORE({self._text_field}, '{self._escape_literal(query)}') AS relevance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"{text_pred} "
            f"ORDER BY relevance DESC "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        cursor = self.client.cursor()
        results = []
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                metadata = json.loads(row[2] if row[2] is not None else "{}")
                score = float(row[3]) / 100.0  # normalise 0-100 → 0.0-1.0
                results.append(OutputData(id=row[0], score=score, payload=metadata))
        except Exception as exc:
            # Text Search index may have been dropped since startup probe.
            # Return None so mem0 falls back to semantic search gracefully.
            logger.debug(
                "keyword_search() fell back to None (Text Search unavailable): %s", exc
            )
            return None
        finally:
            cursor.close()

        return results

    def close(self) -> None:
        """Close the underlying database connection.

        Call this when you are done with the store to release the connection
        back to the Db2 server.  After calling ``close()`` the instance must
        not be used again.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_text_search(self) -> bool:
        """Return ``True`` if Db2 Text Search is active on this database.

        ``SYSTS_CREATE`` and friends exist in ``SYSCAT.PROCEDURES`` as catalogue
        stubs on every Db2 12.1 installation regardless of whether Text Search
        has actually been enabled.  The only reliable test is to attempt a real
        ``CONTAINS()`` call — it raises ``SQL21000N`` when the feature is not
        active, and succeeds (returning 0 or 1) when it is.

        We run ``CONTAINS()`` against a trivial ``VALUES`` subquery so no real
        table permission is needed and the query completes instantly.
        """
        sql = "SELECT CONTAINS(v, 'probe') FROM (VALUES ('probe text')) AS t(v)"  # noqa: S608
        cursor = self.client.cursor()
        available = False
        try:
            cursor.execute(sql)
            cursor.fetchone()
            available = True
        except Exception:
            # SQL21000N → Text Search not configured; any other error → treat
            # as unavailable so startup never fails.
            available = False
        finally:
            cursor.close()

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
        """Raise ``RuntimeError`` if the connected Db2 is below ``_MIN_DB2_VERSION``.

        AI Vector Search requires Db2 12.1.0+.  HAMMING and MANHATTAN distance
        functions were added in 12.1.2.  Failing fast here gives a clear error
        message rather than an obscure SQL function-not-found error later.
        """
        sql = "SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(sql)
            row = cursor.fetchone()
        finally:
            cursor.close()

        if row is None:
            logger.warning("Could not determine Db2 version — proceeding anyway.")
            return

        # SERVICE_LEVEL looks like "DB2 v12.1.5.0" — parse the vX.Y.Z.P part.
        raw = str(row[0])
        m = re.search(r"v?(\d+)\.(\d+)\.(\d+)", raw)
        if m is None:
            logger.warning("Could not parse Db2 version from %r — proceeding anyway.", raw)
            return

        actual = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if actual < _MIN_DB2_VERSION:
            req = ".".join(str(x) for x in _MIN_DB2_VERSION)
            act = ".".join(str(x) for x in actual)
            raise RuntimeError(
                f"IBM Db2 {req}+ is required for AI Vector Search "
                f"(connected server reports {act}).  "
                f"Please upgrade your Db2 instance."
            )
        logger.debug("Db2 version check passed: %s", raw.strip())

    @staticmethod
    def _escape_literal(value: str) -> str:
        """Escape a string value for safe inline use in a SQL string literal.

        ibm_db cannot bind ``?`` parameters in queries that contain
        ``SYSTOOLS.BSON2JSON`` or ``VECTOR_DISTANCE`` — the same limitation
        Databricks has with ``ARRAY`` types in ``StatementParameterListItem``.
        Both drivers handle it identically: inline the value as an escaped
        SQL string literal. Only the single-quote character needs escaping
        per the SQL standard (double it: ``'`` → ``''``).
        """
        return str(value).replace("'", "''")

    def _where_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """Build a WHERE clause with safely-escaped inline literals.

        ibm_db cannot bind ``?`` parameters in queries containing
        ``SYSTOOLS.BSON2JSON`` or ``VECTOR_DISTANCE``, so filter values are
        always inlined as escaped SQL string literals — the same approach
        Databricks uses for ``ARRAY`` types that its driver cannot parameterise.

        Args:
            filters: Dict of ``{field: value}`` equality filters.

            Supported value forms:

            * Scalar string/int/bool — equality: ``{"user_id": "alice"}``
            * ``"*"`` — wildcard, match any value (condition skipped)
            * List — ``IN`` clause: ``{"user_id": ["alice", "bob"]}``
            * Dict with operator keys:

              +---------+----------------------------------------------+
              | ``eq``  | equal (same as scalar shorthand)             |
              +---------+----------------------------------------------+
              | ``ne``  | not equal                                    |
              +---------+----------------------------------------------+
              | ``gt``  | greater than (numeric cast)                  |
              +---------+----------------------------------------------+
              | ``gte`` | greater than or equal (numeric cast)         |
              +---------+----------------------------------------------+
              | ``lt``  | less than (numeric cast)                     |
              +---------+----------------------------------------------+
              | ``lte`` | less than or equal (numeric cast)            |
              +---------+----------------------------------------------+
              | ``in``  | value in list                                |
              +---------+----------------------------------------------+
              | ``nin`` | value not in list                            |
              +---------+----------------------------------------------+

        Returns:
            SQL fragment starting with ``WHERE``, or empty string when no
            filters apply.
        """
        if not filters:
            return ""

        conditions: list[str] = []

        for key, value in filters.items():
            mf = self._metadata_field
            json_expr = f"JSON_VALUE(SYSTOOLS.BSON2JSON({mf}), '$.{key}')"

            if value == "*":
                # Wildcard — match any value; skip condition entirely.
                continue

            if isinstance(value, dict):
                # Operator-dict form: {"field": {"gte": 18, "lte": 65}}
                for op, op_val in value.items():
                    cond = self._op_condition(json_expr, op, op_val)
                    if cond:
                        conditions.append(cond)

            elif isinstance(value, list):
                # List shorthand → IN clause
                escaped = ", ".join(f"'{self._escape_literal(v)}'" for v in value)
                conditions.append(f"{json_expr} IN ({escaped})")

            else:
                # Scalar equality shorthand
                conditions.append(f"{json_expr} = '{self._escape_literal(value)}'")

        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    @staticmethod
    def _op_condition(json_expr: str, op: str, value: Any) -> str:
        """Translate a single operator dict entry into a SQL condition fragment.

        All values are inlined as escaped literals (ibm_db cannot bind ``?``
        params alongside ``BSON2JSON``/``VECTOR_DISTANCE`` — see Phase 1 Fix 6).

        Numeric operators (gt, gte, lt, lte) cast the JSON string value to
        DOUBLE so comparisons work correctly for integer/float metadata.
        """
        op = op.lower()
        esc = Db2VectorStore._escape_literal

        if op == "eq":
            return f"{json_expr} = '{esc(value)}'"
        if op == "ne":
            return f"{json_expr} <> '{esc(value)}'"
        if op == "gt":
            return f"CAST({json_expr} AS DOUBLE) > {float(value)}"
        if op == "gte":
            return f"CAST({json_expr} AS DOUBLE) >= {float(value)}"
        if op == "lt":
            return f"CAST({json_expr} AS DOUBLE) < {float(value)}"
        if op == "lte":
            return f"CAST({json_expr} AS DOUBLE) <= {float(value)}"
        if op == "in":
            if not isinstance(value, list):
                raise ValueError(f"Filter operator 'in' requires a list, got {type(value).__name__}")
            escaped = ", ".join(f"'{esc(v)}'" for v in value)
            return f"{json_expr} IN ({escaped})"
        if op == "nin":
            if not isinstance(value, list):
                raise ValueError(f"Filter operator 'nin' requires a list, got {type(value).__name__}")
            escaped = ", ".join(f"'{esc(v)}'" for v in value)
            return f"{json_expr} NOT IN ({escaped})"
        raise ValueError(
            f"Unsupported filter operator '{op}'. "
            f"Supported: eq, ne, gt, gte, lt, lte, in, nin"
        )
