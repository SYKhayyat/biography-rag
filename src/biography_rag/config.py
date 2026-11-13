"""All runtime configuration, read from the environment (or a local .env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------
    # Empty -> in-memory store (local dev and tests). Any pgvector-enabled
    # Postgres works: Supabase, AWS RDS, Neon.
    database_url: str = ""

    # --- embeddings --------------------------------------------------------
    embedder: str = "openai"  # "openai" | "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- chat models -------------------------------------------------------
    llm_provider: str = "openai"  # "openai" | "qwen" | "deepseek"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    qwen_api_key: str = ""
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_timeout_seconds: float = 120.0

    # --- retrieval / generation -------------------------------------------
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200
    top_k: int = 6
    candidate_k: int = 24  # per arm, before fusion
    min_similarity: float = 0.0  # drop vector hits below this cosine (0 = off)
    query_concurrency: int = 4
    verify_with_llm: bool = True

    # --- app ---------------------------------------------------------------
    log_dir: str = "logs"
    max_upload_mb: int = 60
    max_files: int = 300
    runs_per_hour_per_client: int = 20
    admin_token: str = ""  # required for destructive endpoints; empty disables them


@lru_cache
def get_settings() -> Settings:
    return Settings()
