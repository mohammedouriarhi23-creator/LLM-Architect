from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLMarchitect API"
    app_version: str = "0.1.0"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # LM Studio OpenAI-compatible local server settings
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_model: str = "local-model"

    # Optional API keys for enriched scraping
    huggingface_token: str = ""
    artificial_analysis_api_key: str = ""

    # Target Unsloth base models (Low VRAM 6-8GB 3B class)
    default_base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    alternate_base_model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
