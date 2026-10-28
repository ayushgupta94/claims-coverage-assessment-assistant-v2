"""Centralized application configuration.

All runtime configuration comes from environment variables (or a local .env
file) via pydantic-settings. Nothing else in the codebase reads os.environ.

One LLM provider (OpenAI) and one embedding model, used identically in local
dev, CI evals and production -- a single code path is easier to understand
and defend than a stub/real split.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py = <project>/src/app/config.py -> parents[2] = <project>
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Claims Coverage Assessment Assistant"
    environment: Literal["local", "dev", "production"] = "local"
    log_level: str = "INFO"
    # Render (and most PaaS) inject PORT; API_PORT is accepted for local use.
    api_port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "API_PORT"))

    # --- Access: comma-separated codes, sent as X-Access-Code (see api/security.py).
    # Empty means every request is rejected -- fails closed. ---
    access_codes: str = ""
    max_runs_per_day: int = 50

    # --- MCP: the agent calls tools through a real MCP client pointed at this
    # app's own /mcp endpoint over loopback HTTP. Derived from api_port unless set. ---
    mcp_server_url: str | None = None

    # --- Persistence: MongoDB Atlas (vector + full-text search live there too) ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "claims_assistant_v2"

    # --- OpenAI ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2  # the OpenAI SDK retries with exponential backoff
    llm_max_output_tokens: int = 800
    # USD per 1M tokens, used for cost tracking only.
    llm_input_price_per_million: float = 0.15
    llm_output_price_per_million: float = 0.60

    # --- Retrieval ---
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    rag_top_k: int = 4
    rag_candidates: int = 20  # candidates per search before fusion
    max_chunk_chars: int = 1200

    # --- Decision policy ---
    review_confidence_threshold: float = 0.7

    # --- Fraud heuristic thresholds ---
    fraud_high_amount_threshold: float = 500_000.0
    fraud_early_claim_days_threshold: int = 30
    fraud_frequency_lookback_days: int = 365
    fraud_frequency_claim_count_threshold: int = 3

    # --- Optional Langfuse tracing ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @model_validator(mode="after")
    def _derive_mcp_server_url(self) -> "Settings":
        if self.mcp_server_url is None:
            self.mcp_server_url = f"http://127.0.0.1:{self.api_port}/mcp/"
        return self

    @property
    def access_code_list(self) -> list[str]:
        return [code.strip() for code in self.access_codes.split(",") if code.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is parsed exactly once."""
    return Settings()
