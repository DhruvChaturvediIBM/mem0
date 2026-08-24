"""Pydantic configuration for the IBM Db2 vector store integration."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class Db2Config(BaseModel):
    """Configuration required to connect to an IBM Db2 database with vector search enabled.

    Either ``client`` (an existing ``ibm_db_dbi.Connection``) or
    ``connection_params`` must be provided.

    ``connection_params`` keys:
        database (str): Db2 database name.
        host (str): Hostname or IP of the Db2 server.
        port (str | int): Port number (default ``50000``).
        username (str): Db2 user.
        password (str): Db2 password.
        security (bool, optional): Enable SSL/TLS.
        ssl_cert (str, optional): Path to the server certificate (.arm/.pem).
    """

    client: Optional[Any] = Field(
        None,
        description="Existing ibm_db_dbi.Connection (overrides connection_params)",
    )
    connection_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Connection parameters dict with keys: database, host, port, username, password",
    )

    collection_name: str = Field("mem0", description="Db2 table name used as the vector store collection")
    embedding_model_dims: int = Field(1536, description="Dimension of the embedding vectors", gt=0)
    distance_strategy: str = Field(
        "EUCLIDEAN",
        description=(
            "Distance function: EUCLIDEAN (default), COSINE, DOT, "
            "EUCLIDEAN_DISTANCE, HAMMING, or MANHATTAN"
        ),
    )

    text_field: str = Field("text", description="Column name for the raw text (CLOB)")
    id_field: str = Field("id", description="Column name for the primary key (VARCHAR 36)")
    metadata_field: str = Field("metadata", description="Column name for JSON metadata (BLOB)")
    embedding_field: str = Field("embedding", description="Column name for the vector (FLOAT32)")

    @model_validator(mode="after")
    def _require_connection(self) -> "Db2Config":
        if self.client is None and not self.connection_params:
            raise ValueError("Either `client` or `connection_params` must be provided.")
        valid = {"EUCLIDEAN", "COSINE", "DOT", "EUCLIDEAN_DISTANCE", "HAMMING", "MANHATTAN"}
        if self.distance_strategy.upper() not in valid:
            raise ValueError(f"`distance_strategy` must be one of {valid}; got '{self.distance_strategy}'")
        self.distance_strategy = self.distance_strategy.upper()
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        extra_fields = set(values.keys()) - allowed_fields
        if extra_fields:
            raise ValueError(
                "Extra fields not allowed: {}. Please input only the following fields: {}".format(
                    ", ".join(sorted(extra_fields)), ", ".join(sorted(allowed_fields))
                )
            )
        return values
