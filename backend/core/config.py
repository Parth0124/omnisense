"""Typed settings, loaded once from the environment and validated at import time.

This module is the **only** place in OmniSense that reads `os.environ`
(`docs/coding-standards.md` §2.9). Scattered environment lookups make it
impossible to answer "what does this deployment actually need?" without grepping,
and they defer configuration errors until the first request that happens to touch
the misconfigured code path. Everything here fails at boot instead.

Settings are grouped into nested classes, one per subsystem, each with an
`env_prefix`. The environment variable names stay flat and conventional
(`POSTGRES_HOST`, `QDRANT_URL`), while the Python side reads as
`settings.postgres.host`.

Every field whose value is a choice is typed as an enum, so an invalid value is
rejected at startup with the full list of alternatives rather than producing a
confusing failure later. `.env.example` documents those alternatives inline.

Layer note: this is the **L1k kernel** (`docs/architecture.md` §6.1). It may be
imported by `services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`.
It may **not** be imported by `connectors/` -- a connector receives credentials
as constructor arguments so it stays testable with `respx` alone.
"""

from __future__ import annotations

import enum
import functools
import re
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AppSettings",
    "Environment",
    "EmbeddingProvider",
    "KafkaSettings",
    "LLMEffort",
    "LLMProvider",
    "LogFormat",
    "LogLevel",
    "Neo4jSettings",
    "ObservabilitySettings",
    "OpenSearchSettings",
    "PostgresSettings",
    "QdrantSettings",
    "RedisSettings",
    "Settings",
    "StorageSettings",
    "get_settings",
]

_ENV_FILE = ".env"
_PLACEHOLDER = "change-me"


def _config(prefix: str) -> SettingsConfigDict:
    """Shared settings config. `extra="ignore"` because one `.env` feeds every group."""
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# --------------------------------------------------------------------------- #
# Choice vocabularies
# --------------------------------------------------------------------------- #


class Environment(enum.StrEnum):
    """Deployment environment. Gates the production safety checks below."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @property
    def is_production_like(self) -> bool:
        """Whether placeholder secrets and permissive defaults must be rejected."""
        return self in (Environment.STAGING, Environment.PROD)


class LogLevel(enum.StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(enum.StrEnum):
    CONSOLE = "console"
    """Human-readable, coloured. For a terminal."""
    JSON = "json"
    """One JSON object per line. For anything that ships logs."""


class LLMProvider(enum.StrEnum):
    """Chat/completion backend. The AI layer is model-agnostic (Design Doc §15)."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    LITELLM = "litellm"


class LLMEffort(enum.StrEnum):
    """How much reasoning the model spends before it answers.

    The current Claude generation removed `temperature`/`top_p`/`top_k` and
    replaced the old fixed thinking budget with this: effort is the depth dial.
    Deliberately **unset** by default -- an omitted value means "the provider's
    own default", which is where the vendor's tuning lives, and pinning a level
    here would freeze that tuning at whatever was current the day it was typed.
    Set it when a deployment has measured a level that beats the default.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class EmbeddingProvider(enum.StrEnum):
    """Embedding backend.

    Changing this after ingestion has begun requires re-embedding every Signal --
    vectors from different models are not comparable and cannot share a Qdrant
    collection (`docs/signal-model.md` §9, open question 1).
    """

    VOYAGE = "voyage"
    OPENAI = "openai"
    COHERE = "cohere"
    JINA = "jina"
    HUGGINGFACE = "huggingface"
    MODAL = "modal"
    LOCAL = "local"


class VectorDistance(enum.StrEnum):
    """Qdrant distance metric. Fixed at collection creation; changing it rebuilds."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"
    MANHATTAN = "manhattan"


class FusionStrategy(enum.StrEnum):
    """How heterogeneous candidate lists are merged (`docs/retrieval.md`)."""

    RRF = "rrf"
    """Reciprocal rank fusion. Score-scale independent; the sane default."""
    WEIGHTED = "weighted"
    """Linear combination of normalized scores. Needs per-backend calibration."""
    MAX = "max"
    """Take the best rank per document. Cheap, loses corroboration."""


class KafkaSecurityProtocol(enum.StrEnum):
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class AutoOffsetReset(enum.StrEnum):
    EARLIEST = "earliest"
    LATEST = "latest"


# --------------------------------------------------------------------------- #
# Subsystem settings
# --------------------------------------------------------------------------- #


class AppSettings(BaseSettings):
    """Process-level runtime configuration."""

    model_config = _config("")

    environment: Environment = Field(default=Environment.LOCAL, alias="OMNISENSE_ENV")
    log_level: LogLevel = Field(default=LogLevel.INFO, alias="LOG_LEVEL")
    log_format: LogFormat = Field(default=LogFormat.CONSOLE, alias="LOG_FORMAT")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    api_workers: int = Field(default=1, ge=1, le=64, alias="API_WORKERS")
    cors_allowed_origins: str = Field(
        default="http://localhost:3000", alias="CORS_ALLOWED_ORIGINS"
    )

    @computed_field
    @property
    def cors_origin_list(self) -> list[str]:
        """`CORS_ALLOWED_ORIGINS` split into a list. Comma-separated in the env."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


class SecuritySettings(BaseSettings):
    """Secrets and auth parameters.

    Every secret is a `SecretStr`, so it renders as `**********` if a settings
    object is ever logged or included in an error payload. That is not paranoia:
    FastAPI's default validation-error handler echoes input, and structlog will
    happily serialize whatever it is handed.
    """

    model_config = _config("")

    secret_key: SecretStr = Field(default=SecretStr(_PLACEHOLDER), alias="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_ttl_seconds: int = Field(
        default=3600, ge=60, alias="ACCESS_TOKEN_TTL_SECONDS"
    )
    credential_encryption_key: SecretStr = Field(
        default=SecretStr(_PLACEHOLDER), alias="CREDENTIAL_ENCRYPTION_KEY"
    )

    def fernet_key_is_wellformed(self) -> bool:
        """Whether the credential key looks like a Fernet key (44-char base64).

        Checked rather than assumed because the failure mode otherwise surfaces
        as an unhandled exception the first time a connector credential is
        written -- long after startup, in a worker, at an inconvenient hour.
        """
        key = self.credential_encryption_key.get_secret_value()
        return bool(re.fullmatch(r"[A-Za-z0-9_\-]{43}=", key))


class PostgresSettings(BaseSettings):
    """Transactional metadata store (ADR-0004)."""

    model_config = _config("POSTGRES_")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = "omnisense"
    user: str = "omnisense"
    password: SecretStr = SecretStr("omnisense")

    url: str = Field(
        default="postgresql+asyncpg://omnisense:omnisense@localhost:5432/omnisense",
        alias="DATABASE_URL",
    )
    pool_size: int = Field(default=10, ge=1, alias="POSTGRES_POOL_SIZE")
    max_overflow: int = Field(default=5, ge=0, alias="POSTGRES_MAX_OVERFLOW")
    pool_timeout_seconds: int = Field(default=30, ge=1, alias="POSTGRES_POOL_TIMEOUT_SECONDS")
    connect_timeout_seconds: int = Field(
        default=10,
        ge=1,
        alias="POSTGRES_CONNECT_TIMEOUT_SECONDS",
        description="TCP connect ceiling. Distinct from pool_timeout_seconds, which "
        "bounds pool checkout and does nothing for a hung connect. asyncpg's own "
        "default is 60s, which is far too long to notice an unreachable host.",
    )
    command_timeout_seconds: int = Field(
        default=60,
        ge=1,
        alias="POSTGRES_COMMAND_TIMEOUT_SECONDS",
        description="Per-statement ceiling. Without it a single pathological query "
        "holds a pooled connection indefinitely and the pool starves.",
    )
    echo_sql: bool = Field(default=False, alias="POSTGRES_ECHO_SQL")

    @model_validator(mode="after")
    def _require_async_driver(self) -> Self:
        """The application is async end to end; a sync driver deadlocks the loop.

        `postgresql://` selects psycopg2 under SQLAlchemy, whose blocking calls
        stall the event loop under load in a way that looks like a slow database
        rather than a configuration error. Caught here instead.
        """
        if self.url.startswith("postgresql://") or self.url.startswith("postgres://"):
            raise ValueError(
                "DATABASE_URL must name an async driver, e.g. "
                "'postgresql+asyncpg://...'. A sync driver blocks the event loop. "
                "Alembic uses the same URL and handles the async driver itself."
            )
        return self


class RedisSettings(BaseSettings):
    """Cache, rate-limit buckets, dedup seen-sets and session state (ADR-0005)."""

    model_config = _config("REDIS_")

    url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    max_connections: int = Field(default=50, ge=1)


class Neo4jSettings(BaseSettings):
    """Knowledge graph (ADR-0002)."""

    model_config = _config("NEO4J_")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("omnisense")
    database: str = "neo4j"
    max_connection_pool_size: int = Field(default=50, ge=1)
    connection_timeout_seconds: int = Field(default=30, ge=1)


class QdrantSettings(BaseSettings):
    """Vector store (ADR-0003)."""

    model_config = _config("QDRANT_")

    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection: str = "omnisense_signals"
    distance: VectorDistance = VectorDistance.COSINE
    prefer_grpc: bool = False
    timeout_seconds: int = Field(default=30, ge=1)


class OpenSearchSettings(BaseSettings):
    """Keyword/BM25 retrieval."""

    model_config = _config("OPENSEARCH_")

    url: str = "http://localhost:9200"
    user: str | None = None
    password: SecretStr | None = None
    signal_index: str = "omnisense-signals"
    timeout_seconds: int = Field(default=30, ge=1)
    verify_certs: bool = True


class KafkaSettings(BaseSettings):
    """Event log for the ingestion path (ADR-0007)."""

    model_config = _config("KAFKA_")

    bootstrap_servers: str = "localhost:19092"
    consumer_group: str = "omnisense"
    security_protocol: KafkaSecurityProtocol = KafkaSecurityProtocol.PLAINTEXT
    sasl_username: str | None = None
    sasl_password: SecretStr | None = None
    auto_offset_reset: AutoOffsetReset = AutoOffsetReset.EARLIEST
    max_poll_records: int = Field(default=100, ge=1)

    topic_raw_records: str = "omnisense.records.raw"
    topic_signals: str = "omnisense.signals.enriched"
    topic_graph_updates: str = "omnisense.graph.updates"
    topic_dlq: str = "omnisense.dlq"


class StorageSettings(BaseSettings):
    """Cloudflare R2, via the S3-compatible API (ADR-0006)."""

    model_config = _config("R2_")

    endpoint_url: str | None = None
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None
    bucket: str = "omnisense"
    region: str = "auto"
    raw_retention_days: int = Field(
        default=400,
        ge=1,
        description="Matches the Signal retention window so a rebuild from raw "
        "payloads is always possible (`docs/security-and-privacy.md` §6.2).",
    )


class LLMSettings(BaseSettings):
    """The model-agnostic AI layer (Design Doc §15)."""

    model_config = _config("LLM_")

    provider: LLMProvider = Field(default=LLMProvider.ANTHROPIC, alias="LLM_PROVIDER")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        description="Credential for any non-Anthropic backend -- OpenRouter, "
        "OpenAI, LiteLLM. Separate from ANTHROPIC_API_KEY because a deployment "
        "can legitimately hold both: Anthropic direct for chat and an aggregator "
        "for a cheaper fast tier, and collapsing them into one field would make "
        "switching provider silently reuse the wrong credential.",
    )
    base_url: str | None = Field(
        default=None,
        alias="LLM_BASE_URL",
        description="Endpoint override. Required for provider=ollama or litellm; "
        "defaults to OpenRouter for provider=openai, which is where an "
        "unqualified 'openai' setting almost always points in practice.",
    )

    model_planner: str = Field(default="claude-opus-5", alias="LLM_MODEL_PLANNER")
    model_worker: str = Field(default="claude-sonnet-5", alias="LLM_MODEL_WORKER")
    model_fast: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_MODEL_FAST")

    max_output_tokens: int = Field(
        default=16000,
        ge=1,
        alias="LLM_MAX_OUTPUT_TOKENS",
        description="Caps thinking *plus* response text together on the current "
        "Claude generation, where thinking is on by default. 8192 -- the obvious "
        "number, and the old default -- is low enough that a planner call can "
        "spend its whole budget reasoning and return a truncated answer, which "
        "presents as a model-quality problem rather than as the config defect it is.",
    )
    timeout_seconds: int = Field(default=120, ge=1, alias="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, ge=0, alias="LLM_MAX_RETRIES")
    cache_enabled: bool = Field(default=True, alias="LLM_CACHE_ENABLED")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    effort: LLMEffort | None = Field(
        default=None,
        alias="LLM_EFFORT",
        description="Reasoning depth. Unset means the provider's own default; see LLMEffort.",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding generation. See the warning on `EmbeddingProvider`."""

    model_config = _config("EMBEDDING_")

    provider: EmbeddingProvider | None = None
    model: str | None = None
    api_key: SecretStr | None = Field(
        default=None,
        description="Credential for the embedding provider. Deliberately separate "
        "from ANTHROPIC_API_KEY: Anthropic has no embeddings API, so the embedding "
        "vendor is always a different one, and the two rotate independently.",
    )
    base_url: str | None = Field(
        default=None,
        description="Override endpoint. Required for provider=modal or local, "
        "where the URL is whatever infra/modal/ exposes.",
    )
    dimensions: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=64, ge=1)
    max_chars_per_chunk: int = Field(default=2000, ge=100)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _overlap_below_chunk(self) -> Self:
        """Overlap must be smaller than the chunk, or chunking never advances."""
        if self.chunk_overlap_chars >= self.max_chars_per_chunk:
            raise ValueError(
                "EMBEDDING_CHUNK_OVERLAP_CHARS must be smaller than "
                "EMBEDDING_MAX_CHARS_PER_CHUNK, otherwise the splitter makes no "
                "forward progress and emits chunks forever."
            )
        return self


class RetrievalSettings(BaseSettings):
    """Hybrid retrieval tuning (`docs/retrieval.md`). Defaults are starting points."""

    model_config = _config("RETRIEVAL_")

    keyword_candidates: int = Field(default=100, ge=1)
    vector_candidates: int = Field(default=100, ge=1)
    graph_candidates: int = Field(default=50, ge=0)
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant.")
    rerank_enabled: bool = True
    rerank_depth: int = Field(default=50, ge=1)
    rerank_model: str | None = None
    final_top_k: int = Field(default=20, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _depths_are_coherent(self) -> Self:
        """Reranking fewer candidates than you return silently truncates results."""
        if self.rerank_enabled and self.rerank_depth < self.final_top_k:
            raise ValueError(
                f"RETRIEVAL_RERANK_DEPTH ({self.rerank_depth}) is below "
                f"RETRIEVAL_FINAL_TOP_K ({self.final_top_k}); the reranker would "
                "never see enough candidates to fill the result set."
            )
        return self


class AgentSettings(BaseSettings):
    """Orchestration limits (`docs/agent-system.md`)."""

    model_config = _config("")

    max_steps: int = Field(default=50, ge=1, alias="INVESTIGATION_MAX_STEPS")
    timeout_seconds: int = Field(default=1800, ge=1, alias="INVESTIGATION_TIMEOUT_SECONDS")
    max_critic_revisions: int = Field(default=2, ge=0, alias="MAX_CRITIC_REVISIONS")
    max_parallel_agents: int = Field(default=4, ge=1, alias="AGENT_MAX_PARALLEL")
    token_budget_per_investigation: int = Field(
        default=1_000_000, ge=1000, alias="INVESTIGATION_TOKEN_BUDGET"
    )
    checkpoint_enabled: bool = Field(default=True, alias="AGENT_CHECKPOINT_ENABLED")


class ConnectorSettings(BaseSettings):
    """Ingestion-side limits (`docs/connector-spec.md`)."""

    model_config = _config("CONNECTOR_")

    default_rate_limit_per_minute: int = Field(default=60, ge=1)
    max_concurrency: int = Field(default=8, ge=1)
    request_timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=5, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)
    dedup_ttl_seconds: int = Field(default=604_800, ge=0)
    simhash_distance_threshold: int = Field(default=3, ge=0, le=64)

    ingestion_batch_size: int = Field(default=100, ge=1, alias="INGESTION_BATCH_SIZE")


class ObservabilitySettings(BaseSettings):
    """Logs, metrics, traces and LLM tracing (`docs/observability.md`)."""

    model_config = _config("")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="omnisense", alias="LANGSMITH_PROJECT")

    otel_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field(default="omnisense-api", alias="OTEL_SERVICE_NAME")
    otel_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0, alias="OTEL_SAMPLE_RATIO")

    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, ge=1, le=65535, alias="PROMETHEUS_PORT")


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Root settings object. Obtain it via `get_settings()`, never by constructing."""

    model_config = _config("")

    app: AppSettings = Field(default_factory=AppSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    connectors: ConnectorSettings = Field(default_factory=ConnectorSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _production_safety(self) -> Self:
        """Refuse to start a staging/prod process with development defaults.

        Each of these has a real failure mode: a placeholder `SECRET_KEY` means
        anyone can mint a token; a malformed `CREDENTIAL_ENCRYPTION_KEY` means
        connector credentials cannot be written at all; `CORS: *` in production
        makes every authenticated endpoint reachable from any origin.

        Local development is exempt, which is why the defaults elsewhere in this
        file are convenient rather than safe.
        """
        if not self.app.environment.is_production_like:
            return self

        problems: list[str] = []

        if self.security.secret_key.get_secret_value() == _PLACEHOLDER:
            problems.append("SECRET_KEY is still the placeholder value")
        if self.security.credential_encryption_key.get_secret_value() == _PLACEHOLDER:
            problems.append("CREDENTIAL_ENCRYPTION_KEY is still the placeholder value")
        elif not self.security.fernet_key_is_wellformed():
            problems.append(
                "CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key (expected 44 "
                "url-safe base64 characters ending in '='); generate one with: "
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        if "*" in self.app.cors_origin_list:
            problems.append("CORS_ALLOWED_ORIGINS contains '*'")
        if self.app.log_format is not LogFormat.JSON:
            problems.append("LOG_FORMAT should be 'json' outside local development")
        if self.postgres.echo_sql:
            problems.append("POSTGRES_ECHO_SQL is on, which logs query parameters")

        if problems:
            raise ValueError(
                f"refusing to start in environment={self.app.environment.value!r}:\n  - "
                + "\n  - ".join(problems)
            )
        return self

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to log at startup and to expose on /readyz.

        Built by dumping the model in JSON mode, which renders every `SecretStr`
        as its masked form -- so this cannot leak a credential even if a new
        secret field is added later and nobody updates this method.
        """
        return self.model_dump(mode="json")


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that the `.env` file is parsed once and every module observes the
    same object. Tests override it with
    `app.dependency_overrides[get_settings] = ...` or by calling
    `get_settings.cache_clear()` after mutating the environment.
    """
    return Settings()
