from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

SQLITE_PATH = BASE_DIR / "checkpoint.sqlite"
CHROMA_PATH = BASE_DIR / ".chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    yandex_cloud_folder_id: str = "<FOLDER_ID>"
    yandex_cloud_api_key: str = "<API_KEY>"
    llm_api_base_url: str = "https://llm.api.cloud.yandex.net/v1"
    embeddings_base_url: str = "http://localhost:8001"

    @property
    def gemma_3_27b_it(self) -> str:
        return f"gpt://{self.yandex_cloud_folder_id}/gemma-3-27b-it/latest"

    @property
    def aliceai_llm(self) -> str:
        return f"gpt://{self.yandex_cloud_folder_id}/aliceai-llm"

    @property
    def qwen3_235b(self) -> str:
        return f"gpt://{self.yandex_cloud_folder_id}/qwen3-235b-a22b-fp8/latest"

    @property
    def yandexgpt_rc(self) -> str:
        return f"gpt://{self.yandex_cloud_folder_id}/yandexgpt/rc"


settings = Settings()
