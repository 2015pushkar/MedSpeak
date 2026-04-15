from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MedSpeak API"
    app_env: str = "development"
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    daily_limit: int = 5
    request_char_limit: int = 16_000
    max_file_size_kb: int = 150
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_classifier_model: str = "gpt-4o-mini"
    openai_analyst_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    runtime_judge_enabled: bool = True
    runtime_judge_model: str = "gpt-4o-mini"
    runtime_judge_fail_closed: bool = True
    upstash_redis_rest_url: str | None = Field(default=None)
    upstash_redis_rest_token: str | None = Field(default=None)
    chroma_persist_directory: str = "data/chroma"
    retrieval_top_k: int = 3
    llm_max_retries: int = 3
    llm_retry_base_delay_seconds: float = 0.5

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_kb * 1024

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
