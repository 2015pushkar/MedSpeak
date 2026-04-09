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
    daily_limit: int = 5
    request_char_limit: int = 16_000
    max_file_size_mb: int = 8
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_classifier_model: str = "gpt-4o-mini"
    openai_analyst_model: str = "gpt-4o"
    upstash_redis_rest_url: str | None = Field(default=None)
    upstash_redis_rest_token: str | None = Field(default=None)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

