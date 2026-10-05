"""
MỤC ĐÍCH: Cấu hình tập trung cho toàn bộ backend.

Mọi giá trị đọc từ biến môi trường / file .env, có giá trị mặc định an toàn.
Code khác chỉ gọi get_settings(), không tự đọc os.environ, không hard-code.
"""

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

    # Ingestion (Phase 2)
    storage_dir: str = "storage"   # nơi lưu file gốc, tương đối với /app trong container
    max_upload_mb: int = 10
    chunk_size: int = 256                    # số TOKEN tối đa mỗi chunk
    chunk_overlap: int = 40                  # số token TỐI ĐA lặp lại giữa 2 chunk liền nhau
    tokenizer_encoding: str = "o200k_base"   # bộ tokenizer của tiktoken ("whitespace" = đếm từ, cho test)

    embedding_provider: str = "gemini"      # embedding provider, có thể là "gemini" hoặc "openai"
    embedding_model: str = "gemini-embedding-001"
    gemini_api_key: SecretStr | None = None         # SecretStr là kiểu dữ liệu bảo mật, khi in ra sẽ không hiển thị giá trị thực
    
    retriever_top_k: int = 20
    reranker_top_k: int = 5



@lru_cache      # dùng lại config khi gọi nhiều lần, tránh tạo nhiều instance
def get_settings() -> Settings:
    return Settings()