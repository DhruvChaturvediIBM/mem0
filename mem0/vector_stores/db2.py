"""IBM Db2 vector store for mem0."""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
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

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

# Minimum version for AI Vector Search (EUCLIDEAN/COSINE/DOT).
_MIN_DB2_VERSION = (12, 1, 2)

# Minimum version for the native ANN graph index (CREATE VECTOR INDEX).
_MIN_ANN_VERSION = (12, 1, 5)

# Distance metrics supported by the Db2 ANN index.
# HAMMING, MANHATTAN, and DOT are NOT supported — they stay on exact scan.
_ANN_SUPPORTED_METRICS = {"COSINE", "EUCLIDEAN", "EUCLIDEAN_DISTANCE"}

# ---------------------------------------------------------------------------
# Known limitations (documented, no code workaround needed)
# ---------------------------------------------------------------------------
#
# 1. EUCLIDEAN_DISTANCE SQL token (all versions)
#    Db2's VECTOR_DISTANCE() and CREATE VECTOR INDEX only accept "EUCLIDEAN"
#    as the SQL keyword — "EUCLIDEAN_DISTANCE" raises SQL0104N on every tested
#    version (12.1.2, 12.1.3, 12.1.5, 12.1.6).  The alias is kept valid at
#    the Python/config level for backwards-compatibility and is silently mapped
#    to "EUCLIDEAN" before any SQL is emitted.
#
# 2. ANN vector index (use_vector_index=True) and Db2 Community Edition (CE)
#    -------------------------------------------------------------------------
#    CREATE VECTOR INDEX is a resource-intensive operation: after the DDL
#    commits, Db2 builds the HNSW graph entirely in memory.  On Db2 CE
#    containers (Podman / Docker) the process exhausts the CE memory cap and
#    drops all TCP connections for ~15-30 s while it recovers.  Any Python
#    call on the same connection object will raise:
#
#        SQL30081N  A communication error has been detected.
#        Communication function detecting the error: "recv".
#        Protocol specific error code(s): "54", "*", "0".  SQLSTATE=08001
#
#    This is a hard CE resource constraint — NOT a bug in this driver.
#
#    Suggestions if you hit this error:
#      a) Increase the container's memory limit (recommended ≥ 4 GB):
#             podman run --memory=4g ...  (or docker run --memory=4g ...)
#      b) After the error, manually restart the Db2 instance / container,
#         then reconnect — the index will already be on disk and does not
#         need to be recreated.
#      c) On very small machines simply set  use_vector_index=False  (the
#         default).  Exact-scan search works correctly on all versions and
#         has no connection-stability issues.
#
#    Production Db2 servers (Standard / Advanced Edition, Db2 on Cloud) are
#    unaffected — confirmed on Db2 12.1.6.

# Bug fix: Db2's VECTOR_DISTANCE() only accepts "EUCLIDEAN" as the SQL token —
# "EUCLIDEAN_DISTANCE" is not a valid metric keyword in any version of Db2 AI
# Vector Search (confirmed on 12.1.3 and 12.1.5, SQL0104N).  We keep the alias
# valid at config / Python level for backwards-compatibility, but silently map
# it to "EUCLIDEAN" for all SQL generation.
_SQL_METRIC = {
    "EUCLIDEAN_DISTANCE": "EUCLIDEAN",
}

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

# Logical filter key aliases recognised by _where_clause.
_LOGICAL_OPS = {
    "$and": "AND",
    "$or":  "OR",
    "$not": "NOT",
    "AND":  "AND",
    "OR":   "OR",
    "NOT":  "NOT",
}


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
    text_lemmatized_field: str,
) -> None:
    if _table_exists(client, table_name):
        logger.info("Table %s already exists.", table_name)
        return

    # id_field is VARCHAR(36) to hold standard 36-character UUID strings.
    # text_lemmatized_field stores the pre-processed (stemmed) text that
    # mem0's pipeline writes to payload["text_lemmatized"] — used by
    # keyword_search() for higher-recall full-text matching.
    cols = (
        f"{id_field} VARCHAR(36) PRIMARY KEY NOT NULL, "
        f"{text_field} CLOB, "
        f"{text_lemmatized_field} CLOB, "
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
            ``"COSINE"``, ``"DOT"``, ``"EUCLIDEAN_DISTANCE"``,
            ``"HAMMING"``, or ``"MANHATTAN"``.
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
        text_lemmatized_field: Column name for pre-processed (lemmatized) text
            (default ``"text_lemmatized"``).  Used by ``keyword_search()`` for
            higher-recall full-text matching when Db2 Text Search is installed.
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
                safe = re.sub(r"PWD=[^;]*", "PWD=***", conn_str)
                raise ConnectionError(
                    f"Db2 connection failed: {exc}  (conn={safe})"
                ) from exc

        # Version check — fail fast if Db2 is too old for AI Vector Search.
        # Also stores self._db2_version for use_vector_index gating below.
        self._db2_version: Optional[Tuple[int, int, int]] = None
        self._check_db2_version()

        self.collection_name = self.config.collection_name
        self._text_field = self.config.text_field
        self._text_lemmatized_field = self.config.text_lemmatized_field
        self._id_field = self.config.id_field
        self._metadata_field = self.config.metadata_field
        self._embedding_field = self.config.embedding_field
        self._distance_strategy = self.config.distance_strategy
        self._embedding_dim = self.config.embedding_model_dims

        # Probe once at startup whether Db2 Text Search is installed.
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
            self._text_lemmatized_field,
        )

        # Optionally create ANN vector index (requires 12.1.5+, opt-in).
        self._maybe_create_vector_index(self.collection_name)

    # ------------------------------------------------------------------
    # Change 6: unified cursor context manager
    # Replaces 12 identical try/finally cursor.close() blocks.
    # Guarantees cursor.close() even when rollback itself raises.
    # ------------------------------------------------------------------

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
            self._text_lemmatized_field,
        )
        self._maybe_create_vector_index(name)

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
            List of stored IDs.
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
                meta.get("text_lemmatized", ""),
            )
            for vid, vec, meta in zip(ids, vectors, payloads)
        ]

        sql = (
            f"INSERT INTO {self.collection_name} "  # noqa: S608
            f"({self._id_field}, {self._embedding_field}, "
            f"{self._metadata_field}, {self._text_field}, "
            f"{self._text_lemmatized_field}) "
            f"VALUES (?, VECTOR(?, {embedding_len}, FLOAT32), SYSTOOLS.JSON2BSON(?), ?, ?)"
        )

        with self._get_cursor(commit=True) as cursor:
            cursor.executemany(sql, rows)

        return ids

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
        where_clause = self._where_clause(filters)
        order_dir = _ORDER_BY_DIRECTION[self._distance_strategy]
        # EUCLIDEAN_DISTANCE is not a valid SQL token in VECTOR_DISTANCE() —
        # only "EUCLIDEAN" is accepted (SQL0104N on 12.1.3 and 12.1.5).
        # Map it here; the alias remains valid at config/Python level.
        sql_metric = _SQL_METRIC.get(self._distance_strategy, self._distance_strategy)

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"VECTOR_DISTANCE({self._embedding_field}, "
            f"VECTOR('{embedding_str}', {embedding_len}, FLOAT32), "
            f"{sql_metric}) AS distance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"ORDER BY distance {order_dir} "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            score = _distance_to_score(row[3], self._distance_strategy)
            results.append(OutputData(id=row[0], score=score, payload=metadata))
        return results

    def delete(self, vector_id: str) -> None:
        """Delete a single vector by ID."""
        sql = f"DELETE FROM {self.collection_name} WHERE {self._id_field} = ?"  # noqa: S608
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(sql, [vector_id])

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """Update the embedding and/or payload of an existing record."""
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
            set_parts.append(f"{self._text_field} = ?")
            params.append(payload.get("data", ""))
            set_parts.append(f"{self._text_lemmatized_field} = ?")
            params.append(payload.get("text_lemmatized", ""))
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

    def list_cols(self) -> List[str]:
        """Return the names of all user tables in the current schema."""
        sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TYPE = 'T' AND TABSCHEMA = CURRENT SCHEMA"  # noqa: S608
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def delete_col(self) -> None:
        """Drop the collection table if it exists."""
        if not _table_exists(self.client, self.collection_name):
            logger.info("Table %s not found; nothing to drop.", self.collection_name)
            return
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(f"DROP TABLE {self.collection_name}")
        logger.info("Table %s dropped.", self.collection_name)

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
        # Single round-trip: catalog lookup + live COUNT(*) as a scalar subquery.
        # UPPER(?) normalises the caller-supplied table name to match SYSCAT casing.
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

    def list(
        self,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = 100,
    ) -> List[List[OutputData]]:
        """List records in the collection."""
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

        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            results.append(OutputData(id=row[0], score=None, payload=metadata))
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
            self._text_lemmatized_field,
        )
        self._maybe_create_vector_index(self.collection_name)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Optional[List[OutputData]]:
        """Full-text keyword search using Db2 Text Search (``CONTAINS()``).

        Searches the ``text_lemmatized`` column — the pre-processed (stemmed,
        stop-word-stripped) text populated by mem0's memory pipeline — for
        higher recall than searching raw text.  Falls back to ``None`` (which
        triggers mem0's semantic-only fallback) when:

        * Db2 Text Search addon is not installed/configured (detected at
          startup by :meth:`_probe_text_search`).
        * The Text Search index does not exist on ``text_lemmatized`` yet.

        To enable keyword search, create a Text Search index on the
        ``text_lemmatized`` column::

            CALL SYSPROC.SYSTS_CREATE(
                CURRENT SCHEMA, '<TABLE>', 'text_lemmatized',
                'MAXIMUM CHARACTERS 10000 LANGUAGE EN FORMAT NONE'
            );

        Args:
            query: The search query text (lemmatized for best results).
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of :class:`OutputData` ordered by relevance score descending,
            or ``None`` if Db2 Text Search is not available.
        """
        if not self._text_search_available:
            return None

        # The query string is inlined as an escaped SQL literal rather than
        # bound with a ``?`` parameter marker.  See ``_escape_literal()`` for
        # the full explanation.  Short version: ibm_db 3.3.0 segfaults
        # (SIGSEGV, exit 139) when ``?`` is used as the comparison value
        # inside a ``JSON_VALUE(SYSTOOLS.BSON2JSON(...)) = ?`` predicate —
        # verified on Db2 12.1.3.0 RHEL x86_64 with the native ibm_db driver.
        # Inlining via ``_escape_literal()`` is the only safe approach.
        esc_query = self._escape_literal(query)
        where_clause = self._where_clause(filters)

        # CONTAINS() must be the first predicate in WHERE or follow AND.
        if where_clause:
            text_pred = (
                f"AND CONTAINS({self._text_lemmatized_field}, '{esc_query}') = 1"
            )
        else:
            text_pred = (
                f"WHERE CONTAINS({self._text_lemmatized_field}, '{esc_query}') = 1"
            )

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"SCORE({self._text_lemmatized_field}, '{esc_query}') AS relevance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"{text_pred} "
            f"ORDER BY relevance DESC "
            f"FETCH FIRST {top_k} ROWS ONLY"
        )

        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as exc:
            # Text Search index may have been dropped since startup probe.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_text_search(self) -> bool:
        """Return ``True`` if Db2 Text Search is active on this database.

        Runs a real ``CONTAINS()`` call against a trivial VALUES subquery.
        ``SQL21000N`` is raised when Text Search is not configured; any
        exception is caught and treated as unavailable so startup never fails.
        """
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
        """Raise ``RuntimeError`` if the connected Db2 is below ``_MIN_DB2_VERSION``.

        Also stores the parsed version tuple on ``self._db2_version`` for use
        by :meth:`_maybe_create_vector_index` to gate ANN index creation on
        12.1.5+.  If the version cannot be parsed, ``self._db2_version`` stays
        ``None`` and downstream code treats it as below the ANN threshold.
        """
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
        self._db2_version = actual  # store for ANN gating

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
        """Create a native ANN vector index when ``use_vector_index=True``.

        Guards:

        1. Config flag ``use_vector_index`` must be ``True``.
        2. Db2 server must be 12.1.5+ (stored in ``self._db2_version``).
           If version is unknown (``None``) or below threshold, logs a warning
           and skips — store continues with exact scan.
        3. ``distance_strategy`` must be in ``_ANN_SUPPORTED_METRICS``
           (COSINE / EUCLIDEAN / EUCLIDEAN_DISTANCE).  HAMMING / MANHATTAN /
           DOT are rejected at config-validation time so this is a safety net.
        4. If ``CREATE VECTOR INDEX`` fails (e.g. insufficient permissions or
           index already exists), logs a warning and continues — never raises.
        """
        if not self.config.use_vector_index:
            return

        # Version gate --------------------------------------------------------
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

        # Metric gate (belt-and-suspenders; config validator already enforces this) -
        if self._distance_strategy not in _ANN_SUPPORTED_METRICS:
            return

        bare_name = table_name.strip('"')
        idx_name = f"{bare_name}_vec_idx"
        # Map EUCLIDEAN_DISTANCE → EUCLIDEAN for the DDL token too.
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
            # Surface the raw Db2 error so the caller sees the real cause.
            # The most common failure on Db2 Community Edition containers is a
            # TCP connection drop (SQL30081N) triggered by memory exhaustion
            # while building the HNSW graph.  This is a CE resource constraint,
            # not a driver bug.  Suggestions:
            #   • Increase container memory (--memory=4g or higher).
            #   • After the error, manually restart the Db2 instance/container
            #     and reconnect — the index is already on disk and does not need
            #     to be recreated.
            #   • Set use_vector_index=False to use exact scan instead.
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

        ibm_db cannot bind ``?`` parameters in queries containing
        ``SYSTOOLS.BSON2JSON`` or ``VECTOR_DISTANCE``.  Both drivers handle
        this identically: inline the value as an escaped SQL literal.  Only
        the single-quote character needs escaping per the SQL standard.
        """
        return str(value).replace("'", "''")

    def _where_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """Build a WHERE clause from a filter dict.

        Supports flat equality filters, operator dicts (eq/ne/gt/gte/lt/lte/
        in/nin/contains/icontains), wildcard ``"*"``, list shorthand (→ IN),
        and compound logical keys ``$and`` / ``$or`` / ``$not``
        (and their unadorned equivalents ``AND`` / ``OR`` / ``NOT``).

        Args:
            filters: Filter dict as passed by the mem0 Memory layer.

        Returns:
            SQL fragment starting with ``WHERE``, or ``""`` when no filters.
        """
        if not filters:
            return ""
        conditions = self._build_conditions(filters)
        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    def _build_conditions(self, filters: Dict[str, Any]) -> List[str]:
        """Recursively translate a filter dict into a list of SQL condition strings."""
        conditions: List[str] = []

        for key, value in filters.items():

            # -- Logical operators --------------------------------------------
            logical_op = _LOGICAL_OPS.get(key)
            if logical_op is not None:
                if logical_op == "NOT":
                    # $not expects a list of sub-filter dicts; combine with OR
                    # then negate: NOT (A OR B OR …)
                    if not isinstance(value, list):
                        value = [value]
                    sub_parts: List[str] = []
                    for sub in value:
                        sub_conds = self._build_conditions(sub)
                        if sub_conds:
                            sub_parts.append("(" + " AND ".join(sub_conds) + ")")
                    if sub_parts:
                        conditions.append("NOT (" + " OR ".join(sub_parts) + ")")
                else:
                    # $and / $or expect a list of sub-filter dicts
                    if not isinstance(value, list):
                        value = [value]
                    sub_parts = []
                    for sub in value:
                        sub_conds = self._build_conditions(sub)
                        if sub_conds:
                            sub_parts.append("(" + " AND ".join(sub_conds) + ")")
                    if sub_parts:
                        joiner = " AND " if logical_op == "AND" else " OR "
                        conditions.append("(" + joiner.join(sub_parts) + ")")
                continue

            # -- Field-level filters ------------------------------------------
            mf = self._metadata_field
            json_expr = f"JSON_VALUE(SYSTOOLS.BSON2JSON({mf}), '$.{key}')"

            if value == "*":
                # Wildcard — match any value; skip condition.
                continue

            if isinstance(value, dict):
                for op, op_val in value.items():
                    cond = self._op_condition(json_expr, op, op_val)
                    if cond:
                        conditions.append(cond)

            elif isinstance(value, list):
                escaped = ", ".join(f"'{self._escape_literal(v)}'" for v in value)
                conditions.append(f"{json_expr} IN ({escaped})")

            else:
                conditions.append(f"{json_expr} = '{self._escape_literal(value)}'")

        return conditions

    @staticmethod
    def _op_condition(json_expr: str, op: str, value: Any) -> str:
        """Translate a single operator dict entry into a SQL condition fragment.

        Supported operators: eq, ne, gt, gte, lt, lte, in, nin,
        contains, icontains.
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
                raise ValueError(
                    f"Filter operator 'in' requires a list, got {type(value).__name__}"
                )
            escaped = ", ".join(f"'{esc(v)}'" for v in value)
            return f"{json_expr} IN ({escaped})"
        if op == "nin":
            if not isinstance(value, list):
                raise ValueError(
                    f"Filter operator 'nin' requires a list, got {type(value).__name__}"
                )
            escaped = ", ".join(f"'{esc(v)}'" for v in value)
            return f"{json_expr} NOT IN ({escaped})"
        if op == "contains":
            # Substring match — JSON_VALUE returns VARCHAR so LIKE works directly.
            return f"{json_expr} LIKE '%{esc(value)}%'"
        if op == "icontains":
            # Case-insensitive substring match via LOWER on both sides.
            return f"LOWER({json_expr}) LIKE LOWER('%{esc(value)}%')"
        raise ValueError(
            f"Unsupported filter operator '{op}'. "
            f"Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"
        )
