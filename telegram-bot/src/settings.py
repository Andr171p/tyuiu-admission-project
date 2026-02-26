from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

SQLALCHEMY_URL = f"sqlite+aiosqlite:///{BASE_DIR}/db.sqlite3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    chatbot_base_url: str = "http://localhost:8001"
    bot_token: str = "<TOKEN>"


settings = Settings()
