from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    app_env: str =  "development"           # thông tin ứng dụng chạy ở môi trường nào
    database_url: str = "postgresql+psycopg://rag:change_me@postgres:5432/ragdb"    # địa chỉ kết nối database
    cors_origins: str = "http://localhost:3000"     # frontend url, có được gọi backend không

    retriever_top_k: int = 20
    reranker_top_k: int = 5

    chunk_size: int = 500
    chunk_overlap: int = 100


@lru_cache      # dùng lại config khi gọi nhiều lần, tránh tạo nhiều instance
def get_settings() -> Settings:
    return Settings()