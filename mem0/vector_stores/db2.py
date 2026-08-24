"""IBM Db2 vector store for mem0."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

try:
    import ibm_db_dbi
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'ibm_db_dbi' library is required. "
        "Install it with: pip install ibm_db"
    ) from exc

from mem0.configs.vector_stores.db2 import Db2Config
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

_SCORE_FROM_DISTANCE = {
    "EUCLIDEAN": lambda d: 1.0 / (1.0 + d),
    "COSINE": lambda d: max(0.0, 1.0 - d),
    "DOT": lambda d: d,  # dot product is already higher = better; return as-is
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
    cursor = client.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
    except Exception as ex:
        if "SQL0204N" in str(ex):
            return False
        raise
    finally:
        cursor.close()
    return True


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
        f"{id_field} CHAR(16) PRIMARY KEY NOT NULL, "
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


def _hash_id(raw_id: str) -> str:
    """Return a 16-character uppercase hex SHA-256 digest suitable for CHAR(16)."""
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16].upper()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Db2VectorStore(VectorStoreBase):
    """IBM Db2 AI Vector Search vector store.

    Supported ``distance_strategy`` values: ``"EUCLIDEAN"``, ``"COSINE"``, ``"DOT"``.

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
            self.client = ibm_db_dbi.connect(conn_str, "", "")

        self.collection_name = self.config.collection_name
        self._text_field = self.config.text_field
        self._id_field = self.config.id_field
        self._metadata_field = self.config.metadata_field
        self._embedding_field = self.config.embedding_field
        self._distance_strategy = self.config.distance_strategy
        self._embedding_dim = self.config.embedding_model_dims

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

        hashed_ids = [_hash_id(i) for i in ids]
        embedding_len = len(vectors[0]) if vectors else self._embedding_dim

        rows = [
            (
                hid,
                "[" + ", ".join(str(v) for v in vec) + "]",
                json.dumps(meta),
                meta.get("data", ""),
            )
            for hid, vec, meta in zip(hashed_ids, vectors, payloads)
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

        return hashed_ids

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

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}), "
            f"VECTOR_DISTANCE({self._embedding_field}, "
            f"VECTOR('{embedding_str}', {embedding_len}, FLOAT32), "
            f"{self._distance_strategy}) AS distance "
            f"FROM {self.collection_name} "
            f"{where_clause} "
            f"ORDER BY distance "
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
        """Delete a single vector by its original or hashed ID.

        Args:
            vector_id: The ID as stored (hashed CHAR(16)) or the raw original
                ID (which will be hashed before lookup).
        """
        import re

        if re.fullmatch(r"[A-F0-9]{16}", vector_id):
            hid = vector_id
        else:
            hid = _hash_id(vector_id)

        sql = f"DELETE FROM {self.collection_name} WHERE {self._id_field} = ?"  # noqa: S608
        cursor = self.client.cursor()
        try:
            cursor.execute(sql, [hid])
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
            vector_id: Original or hashed ID of the record to update.
            vector: New embedding vector. If ``None``, the embedding is not changed.
            payload: New metadata dict. If ``None``, the metadata is not changed.
        """
        import re

        if re.fullmatch(r"[A-F0-9]{16}", vector_id):
            hid = vector_id
        else:
            hid = _hash_id(vector_id)

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

        params.append(hid)
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
            vector_id: Original or hashed ID.

        Returns:
            :class:`OutputData` if found, otherwise ``None``.
        """
        import re

        if re.fullmatch(r"[A-F0-9]{16}", vector_id):
            hid = vector_id
        else:
            hid = _hash_id(vector_id)

        sql = (
            f"SELECT {self._id_field}, "  # noqa: S608
            f"{self._text_field}, "
            f"SYSTOOLS.BSON2JSON({self._metadata_field}) "
            f"FROM {self.collection_name} "
            f"WHERE {self._id_field} = ?"
        )

        cursor = self.client.cursor()
        try:
            cursor.execute(sql, [hid])
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

        Returns:
            Dict with keys ``table_name``, ``row_count``, ``schema``.
        """
        sql = (
            "SELECT TABSCHEMA, TABNAME, CARD "  # noqa: S608
            "FROM SYSCAT.TABLES "
            "WHERE TABNAME = UPPER(?) AND TABSCHEMA = CURRENT SCHEMA"
        )
        cursor = self.client.cursor()
        try:
            cursor.execute(sql, [self.collection_name.strip('"')])
            row = cursor.fetchone()
        finally:
            cursor.close()

        if row is None:
            raise ValueError(f"Collection '{self.collection_name}' not found.")

        return {
            "schema": row[0],
            "table_name": row[1],
            "row_count": row[2],
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            filters: Dict of ``{field: value}`` equality filters. A value of
                ``"*"`` is treated as "match all" and skips the condition.
                A list value generates an ``IN (…)`` clause.

        Returns:
            SQL fragment starting with ``WHERE``, or empty string when no
            filters apply.
        """
        if not filters:
            return ""

        conditions: list[str] = []

        for key, value in filters.items():
            if value == "*":
                # Wildcard — caller wants all values for this field; skip condition.
                continue
            mf = self._metadata_field
            json_expr = f"JSON_VALUE(SYSTOOLS.BSON2JSON({mf}), '$.{key}')"
            if isinstance(value, list):
                escaped = ", ".join(f"'{self._escape_literal(v)}'" for v in value)
                conditions.append(f"{json_expr} IN ({escaped})")
            else:
                conditions.append(f"{json_expr} = '{self._escape_literal(value)}'")

        return ("WHERE " + " AND ".join(conditions)) if conditions else ""
