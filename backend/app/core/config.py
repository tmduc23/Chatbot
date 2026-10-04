from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    app_env: str =  "development"           # thông tin ứng dụng chạy ở môi trường nào
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://rag:change_me@postgres:5432/ragdb"    # địa chỉ kết nối database
    cors_origins: str = "http://localhost:3000"     # frontend url, có được gọi backend không

    embedding_provider: str = "gemini"      # embedding provider, có thể là "gemini" hoặc "openai"
    embedding_model: str = "gemini-embedding-001"
    gemini_api_key: SecretStr | None = None         # SecretStr là kiểu dữ liệu bảo mật, khi in ra sẽ không hiển thị giá trị thực
    retriever_top_k: int = 20
    reranker_top_k: int = 5

    chunk_size: int = 500
    chunk_overlap: int = 100


@lru_cache      # dùng lại config khi gọi nhiều lần, tránh tạo nhiều instance
def get_settings() -> Settings:
    return Settings()