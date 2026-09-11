"""Application settings loaded from environment variables and an optional .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:4b"  # chosen by the end-to-end evaluation
    embed_model: str = "bge-m3"
    llm_num_ctx: int = Field(8192, ge=2048)
    llm_num_predict: int = Field(512, ge=64)
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_seed: int = 42
    llm_think: bool | None = False
    llm_keep_alive: str = "30m"
    llm_max_concurrency: int = Field(2, ge=1)
    llm_timeout_s: float = Field(120.0, gt=0)
    embed_timeout_s: float = Field(60.0, gt=0)
    embed_batch_size: int = Field(32, ge=1)

    # Retrieval and answering
    retrieval_top_k: int = Field(6, ge=1, le=20)
    retrieval_candidates: int = Field(30, ge=5, le=200)
    expansion_max: int = Field(3, ge=0, le=10)
    context_token_budget: int = Field(3500, ge=500)
    refusal_threshold: float = Field(0.50, ge=0.0, le=1.0)

    # Chunking
    chunk_strategy: Literal["structural", "fixed"] = "structural"
    chunk_max_chars: int = Field(1500, ge=300)
    fixed_chunk_size: int = Field(1000, ge=200)
    fixed_chunk_overlap: int = Field(200, ge=0)

    # Source and index
    source_url: str = "https://lex.uz/uz/docs/-8193120"
    source_html: Path = Path("data/raw/lex_8193120.html")
    source_snapshot_date: str = "2026-09-11"
    index_dir: Path = Path("data/index")
    index_keep_versions: int = Field(2, ge=1)
    auto_ingest: bool = True

    log_level: str = "INFO"

    @field_validator("llm_think", mode="before")
    @classmethod
    def _blank_think_means_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Build settings, turning validation errors into a message that names env variables."""
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            variable = "_".join(str(part) for part in error["loc"]).upper()
            problems.append(f"{variable}: {error['msg']}")
        raise ConfigError("Noto'g'ri sozlama: " + "; ".join(problems)) from exc
