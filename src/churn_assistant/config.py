"""Application settings for the Telecom Customer Churn Insight Assistant."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b-cloud"
    use_fallback_llm: bool = False
    default_session_date: str = "2026-10-15"
    data_pack_path: Path = PROJECT_ROOT / "data" / "sample_data_pack.json"


settings = Settings()
