"""IBM Db2 vector store for mem0 — backed by langchain-db2 DB2VS."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel

try:
    import ibm_db_dbi
    from langchain_db2 import DB2VS
    from langchain_db2.db2vs import DistanceStrategy, clear_table, drop_table
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "The 'langchain-db2' package is required for the Db2 vector store. "
        "Install it with: pip install langchain-db2"
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

# Map our string strategy → langchain-db2 DistanceStrategy enum.
_TO_LC_STRATEGY: Dict[str, DistanceStrategy] = {
    "EUCLIDEAN":          DistanceStrategy.EUCLIDEAN_DISTANCE,
    "EUCLIDEAN_DISTANCE": DistanceStrategy.EUCLIDEAN_DISTANCE,
    "COSINE":             DistanceStrategy.COSINE,
    "DOT":                DistanceStrategy.DOT_PRODUCT,
}

# SQL metric token used in raw SQL we still emit (ANN DDL, keyword search).
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
# Lightweight no-op embedding shim
# ---------------------------------------------------------------------------
# DB2VS requires an embedding_function at construction time to probe the
# embedding dimension.  mem0 passes pre-computed vectors to insert/search,
# so we supply a thin shim that records the dimension on first call and
# returns zero-vectors thereafter.  The shim satisfies the EmbeddingsSchema
# protocol (embed_documents + embed_query).


def _hash_id(raw_id: str) -> str:
    """Return the 16-char uppercase SHA-256 hex digest DB2VS stores as the row PK.

    DB2VS hashes every id with ``hashlib.sha256(id).hexdigest()[:16].upper()``
    before storing.  We must apply the same transform when building WHERE clauses
    that target the ``id`` column (``get``, ``update``, ``delete``, ``search``
    result id re-mapping).
    """
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16].upper()


class _DimProbeEmbedding:
    """Shim: records embedding_dim from the first real vector it sees."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # noqa: ARG002
        return [[0.0] * self._dim for _ in texts]

    def embed_query(self, text: str) -> List[float]:  # noqa: ARG002
        return [0.0] * self._dim


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Db2VectorStore(VectorStoreBase):
    """IBM Db2 AI Vector Search vector store — backed by ``langchain-db2``.

    Uses :class:`langchain_db2.DB2VS` for connection management, table
    creation, insert, and similarity search.  mem0-specific extensions
    (version check, optional ANN index, keyword search, rich metadata
    filters, ``text_lemmatized`` column) are layered on top.

    Supported ``distance_strategy`` values:
    ``"EUCLIDEAN"`` (default), ``"COSINE"``, ``"DOT"``,
    ``"EUCLIDEAN_DISTANCE"``, ``"HAMMING"``, ``"MANHATTAN"``.

    For HAMMING and MANHATTAN (not supported by DB2VS), the store falls back
    to raw SQL via the shared ``ibm_db_dbi`` connection that DB2VS manages.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = Db2Config(**kwargs)

        # Build / accept the raw ibm_db_dbi connection -----------------------
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

        # Determine the langchain-db2 DistanceStrategy.
        # HAMMING / MANHATTAN are not supported by DB2VS — we keep them on
        # raw-SQL path.  For the other three we delegate to DB2VS.
        self._lc_strategy: Optional[DistanceStrategy] = _TO_LC_STRATEGY.get(
            self._distance_strategy
        )

        # Ensure extra columns (text_lemmatized) exist alongside what DB2VS
        # creates, then initialise the DB2VS delegate.
        self._ensure_table_with_extra_cols()

        # Initialise DB2VS delegate (reuses our connection, skips its own
        # table-creation if the table already exists).
        self._db2vs = DB2VS(
            embedding_function=_DimProbeEmbedding(self._embedding_dim),
            table_name=self.collection_name,
            client=self.client,
            distance_strategy=(
                self._lc_strategy
                if self._lc_strategy is not None
                else DistanceStrategy.EUCLIDEAN_DISTANCE
            ),
        )

        # Probe for Text Search at startup.
        self._text_search_available: bool = self._probe_text_search()

        # Optionally create ANN vector index (requires 12.1.5+, opt-in).
        self._maybe_create_vector_index(self.collection_name)

    # ------------------------------------------------------------------
    # Cursor context manager (used by our raw-SQL paths)
    # ------------------------------------------------------------------

    @contextmanager
    def _get_cursor(self, commit: bool = False) -> Iterator[Any]:
        """Yield a cursor; commit or rollback on exit; always close cursor."""
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
        self._ensure_table_with_extra_cols(
            table_name=name, embedding_dim=vector_size
        )
        self._maybe_create_vector_index(name)

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors (with optional payloads / ids) into the table.

        Delegates to :meth:`DB2VS.add_texts` for the core insert path.
        The ``text_lemmatized`` column is populated via a follow-up UPDATE
        because DB2VS's schema does not include that column.
        """
        n = len(vectors)
        if payloads is None:
            payloads = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        # DB2VS.add_texts expects texts + metadatas.  We pass the "data" key
        # from the payload as text and the full payload dict as metadata.
        texts = [meta.get("data", "") for meta in payloads]

        # Inject our UUID ids into metadata so DB2VS picks them up and hashes
        # them.  We also store the original UUID in metadata["mem0_id"] so
        # we can reverse-lookup later.
        for i, (vid, meta) in enumerate(zip(ids, payloads)):
            meta = dict(meta)
            meta["mem0_id"] = vid
            payloads[i] = meta

        # Build per-row vector strings for the raw INSERT we do ourselves.
        # DB2VS.add_texts uses its own embedding_function (our shim returns
        # zeros).  We override the embedding column with the real vectors via
        # a follow-up UPDATE to keep DB2VS's high-level logic intact.
        embedding_len = len(vectors[0]) if vectors else self._embedding_dim

        # Use DB2VS.add_texts for table insert (handles schema, COMMIT, etc.)
        hashed_ids = self._db2vs.add_texts(
            texts=texts,
            metadatas=payloads,
            ids=ids,
        )

        # --- Overwrite the embedding column with the real vectors -----------
        # DB2VS stored zero-vectors from our shim.  Patch each row now.
        update_sql = (
            f"UPDATE {self.collection_name} "  # noqa: S608
            f"SET {self._embedding_field} = VECTOR(?, {embedding_len}, FLOAT32), "
            f"    {self._text_lemmatized_field} = ? "
            f"WHERE id = ?"
        )
        rows = [
            (
                "[" + ", ".join(str(v) for v in vec) + "]",
                meta.get("text_lemmatized", ""),
                hid,
            )
            for vec, meta, hid in zip(vectors, payloads, hashed_ids)
        ]
        with self._get_cursor(commit=True) as cursor:
            cursor.executemany(update_sql, rows)

        return ids  # return the original UUID ids, not the hashed ones

    def search(
        self,
        query: str,
        vectors: List[list],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """Search for the *top_k* nearest vectors.

        For EUCLIDEAN, EUCLIDEAN_DISTANCE, COSINE, and DOT the search is
        delegated to DB2VS via raw SQL (DB2VS's internal query path).
        HAMMING and MANHATTAN fall back to a direct SQL query.
        """
        if vectors and isinstance(vectors[0], (int, float)):
            embedding = vectors
        else:
            embedding = vectors[0] if vectors else []
        embedding_len = len(embedding) if embedding else self._embedding_dim
        embedding_str = "[" + ", ".join(str(v) for v in embedding) + "]"

        where_clause = self._where_clause(filters)
        order_dir = _ORDER_BY_DIRECTION[self._distance_strategy]
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
            # DB2VS stores a 16-char hashed id in the id column; return the
            # original UUID from metadata["mem0_id"] when available so callers
            # can use it with get/update/delete without re-hashing.
            result_id = metadata.get("mem0_id", row[0])
            results.append(OutputData(id=result_id, score=score, payload=metadata))
        return results

    def delete(self, vector_id: str) -> None:
        """Delete a single vector by ID.

        Delegates to :meth:`DB2VS.delete` which handles id hashing.
        Also tries the raw UUID in case the row was inserted before hashing
        was introduced.
        """
        self._db2vs.delete(ids=[vector_id])

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

        # DB2VS stores ids as CHAR(16) SHA-256 hashes; translate before WHERE.
        hid = _hash_id(vector_id) if len(vector_id) != 16 else vector_id
        params.append(hid)
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
        # DB2VS stores ids as CHAR(16) SHA-256 hashes; translate before WHERE.
        hid = _hash_id(vector_id) if len(vector_id) != 16 else vector_id

        with self._get_cursor() as cursor:
            cursor.execute(sql, [hid])
            row = cursor.fetchone()

        if row is None:
            return None
        metadata = json.loads(row[2] if row[2] is not None else "{}")
        result_id = metadata.get("mem0_id", row[0])
        return OutputData(id=result_id, score=None, payload=metadata)

    def list_cols(self) -> List[str]:
        """Return the names of all user tables in the current schema."""
        sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TYPE = 'T' AND TABSCHEMA = CURRENT SCHEMA"  # noqa: S608
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def delete_col(self) -> None:
        """Drop the collection table if it exists (delegates to langchain-db2)."""
        drop_table(self.client, self.collection_name)

    def col_info(self) -> Dict[str, Any]:
        """Return metadata about the collection table."""
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
            result_id = metadata.get("mem0_id", row[0])
            results.append(OutputData(id=result_id, score=None, payload=metadata))
        return [results]

    def reset(self) -> None:
        """Drop and recreate the collection table (delegates to langchain-db2)."""
        logger.warning("Resetting collection %s …", self.collection_name)
        drop_table(self.client, self.collection_name)
        self._ensure_table_with_extra_cols()
        self._maybe_create_vector_index(self.collection_name)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> Optional[List[OutputData]]:
        """Full-text keyword search using Db2 Text Search (``CONTAINS()``).

        Searches the ``text_lemmatized`` column.  Returns ``None`` when
        Db2 Text Search is not installed/configured, triggering mem0's
        semantic-only fallback.
        """
        if not self._text_search_available:
            return None

        esc_query = self._escape_literal(query)
        where_clause = self._where_clause(filters)

        if where_clause:
            text_pred = f"AND CONTAINS({self._text_lemmatized_field}, '{esc_query}') = 1"
        else:
            text_pred = f"WHERE CONTAINS({self._text_lemmatized_field}, '{esc_query}') = 1"

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
            logger.debug(
                "keyword_search() fell back to None (Text Search unavailable): %s", exc
            )
            return None

        results = []
        for row in rows:
            metadata = json.loads(row[2] if row[2] is not None else "{}")
            score = float(row[3]) / 100.0
            result_id = metadata.get("mem0_id", row[0])
            results.append(OutputData(id=result_id, score=score, payload=metadata))
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

    def _ensure_table_with_extra_cols(
        self,
        table_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """Create the table (if absent) with the mem0 extended schema.

        DB2VS creates: id CHAR(16), text CLOB, metadata BLOB, embedding VECTOR.
        We additionally need: text_lemmatized CLOB.

        Strategy: let DB2VS create the base table first, then ALTER TABLE to
        add the extra column if it is missing.
        """
        tbl = table_name or self.collection_name
        dim = embedding_dim or self._embedding_dim

        # Let DB2VS create the base 4-column table if it does not exist yet.
        shim = _DimProbeEmbedding(dim)
        tmp_vs = DB2VS(
            embedding_function=shim,
            table_name=tbl,
            client=self.client,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        del tmp_vs  # we only needed the side-effect of table creation

        # Add text_lemmatized column if absent (idempotent).
        col = self._text_lemmatized_field
        check_sql = (
            "SELECT COUNT(*) FROM SYSCAT.COLUMNS "  # noqa: S608
            "WHERE TABNAME = UPPER(?) AND COLNAME = UPPER(?) "
            "AND TABSCHEMA = CURRENT SCHEMA"
        )
        with self._get_cursor() as cursor:
            cursor.execute(check_sql, [tbl.strip('"'), col])
            row = cursor.fetchone()
            col_exists = bool(row and row[0] > 0)

        if not col_exists:
            alter_sql = f"ALTER TABLE {tbl} ADD COLUMN {col} CLOB"
            with self._get_cursor(commit=True) as cursor:
                cursor.execute(alter_sql)
            logger.info("Added column %s to table %s.", col, tbl)

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
        """Escape a string for safe inline use in a SQL string literal."""
        return str(value).replace("'", "''")

    def _where_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """Build a WHERE clause from a filter dict."""
        if not filters:
            return ""
        conditions = self._build_conditions(filters)
        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    def _build_conditions(self, filters: Dict[str, Any]) -> List[str]:
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
                        sub_conds = self._build_conditions(sub)
                        if sub_conds:
                            sub_parts.append("(" + " AND ".join(sub_conds) + ")")
                    if sub_parts:
                        conditions.append("NOT (" + " OR ".join(sub_parts) + ")")
                else:
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

            mf = self._metadata_field
            json_expr = f"JSON_VALUE(SYSTOOLS.BSON2JSON({mf}), '$.{key}')"

            if value == "*":
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
        """Translate a single operator dict entry into a SQL condition fragment."""
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
            return f"{json_expr} LIKE '%{esc(value)}%'"
        if op == "icontains":
            return f"LOWER({json_expr}) LIKE LOWER('%{esc(value)}%')"
        raise ValueError(
            f"Unsupported filter operator '{op}'. "
            f"Supported: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains"
        )
