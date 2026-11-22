"""All runtime configuration, read from the environment (or a local .env)."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
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
    llm_provider: str = "openai"  # see llm/providers.py for every provider name
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    nvidia_api_key: str = ""
    cheapllm_api_key: str = ""
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
    max_number_density: float | None = 0.15  # skip back-of-book index chunks (numbers per word)
    query_concurrency: int = 4
    verify_with_llm: bool = True
    # The judge that audits the woven story. Empty -> the model that wrote it.
    judge_provider: str = ""
    judge_model: str = ""

    # --- app ---------------------------------------------------------------
    log_dir: str = "logs"
    max_upload_mb: int = 60
    max_files: int = 300
    runs_per_hour_per_client: int = 20
    admin_token: str = ""  # required for destructive endpoints; empty disables them
    site_password: str = ""  # HTTP Basic password for the whole site; empty = open
    job_workers: int = 6  # concurrent background runs (a comparison starts one per model)
    max_compare_targets: int = 6
    db_pool_size: int = 12  # Postgres connections; job workers x query threads share them
    failed_corpus_ttl_hours: float = 24.0  # failed uploads are deleted after this (0 = keep)
    max_events_per_run: int = 400  # run-log rows stored per run; later ones are dropped
    # Read the client address from X-Forwarded-For (proxy hops only). On by default on Render,
    # which sets RENDER=true; off elsewhere, where the header is whatever the client sent.
    trust_forwarded_for: bool = Field(default_factory=lambda: bool(os.environ.get("RENDER")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
