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
    storage_dir: str = "storage"             # nơi lưu file gốc, tương đối với /app trong container
    max_upload_mb: int = 10
    chunk_size: int = 256                    # số TOKEN tối đa mỗi chunk
    chunk_overlap: int = 40                  # số token TỐI ĐA lặp lại giữa 2 chunk liền nhau
    tokenizer_encoding: str = "o200k_base"   # bộ tokenizer của tiktoken ("whitespace" = đếm từ, cho test)

    embedding_provider: str = "gemini"      # embedding provider, có thể là "gemini" hoặc "openai"
    embedding_model: str = "gemini-embedding-001"
    gemini_api_key: SecretStr | None = None         # SecretStr là kiểu dữ liệu bảo mật, khi in ra sẽ không hiển thị giá trị thực

    # Retrieval
    retrieval_mode: str = "hybrid"            # vector | keyword | hybrid
    retrieval_candidate_k: int = 30           # số kết quả mỗi nhánh (vector, keyword) lấy ra trước
    rrf_k: int = 60                           # hằng số k trong công thức RRF
    retrieval_top_k: int = 20
    reranker_provider: str = "gemini"
    reranker_model: str = "gemini-3.5-flash-lite"
    reranker_top_k: int = 5
    reranker_max_chars: int = 1000             # cắt bớt mỗi chunk trước khi gửi cho reranker

    # Cohere reranker — dùng khi reranker_provider = "cohere" 
    cohere_api_key: SecretStr | None = None 
    cohere_rerank_model: str = "rerank-v4.0-pro"

    # Ngưỡng lọc chunk sau rerank (Phase 6). None = dùng mặc định theo RERANKER_PROVIDER 
    reranker_min_score: float | None = None 
    # Generation (Phase 6) 
    llm_provider: str = "open router"               # gemini | fake | open router
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free" 
    llm_temperature: float = 0.2 
    llm_max_output_tokens: int = 2048 
    context_max_chars: int = 6000              # tổng độ dài context đưa vào LLM

    # OpenRouter (API chuẩn OpenAI) — dùng khi LLM_PROVIDER=openrouter
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Citation + Grounding (Phase 7)
    grounding_check: bool = False               # True = thêm 1 lần gọi LLM mỗi câu để kiểm tra nguồn
    grounding_model: str | None = None          # Nonen = dùng LLM_MODEL

    # API + hội thoại (Phase 9)
    admin_api_key: SecretStr | None = None      # bảo vệ API quản trị (tạo app, upload, xóa ...) qua header X-API-Key
    chat_condense_question: bool = True         # True = viết lại câu hỏi nối tiếp thành câu độc lập (thêm 1 lần gọi LLM)
    chat_history_turns: int = 3                 # số lượt hỏi-đáp gần nhất được đưa vào câu hỏi

    


@lru_cache      # dùng lại config khi gọi nhiều lần, tránh tạo nhiều instance
def get_settings() -> Settings:
    return Settings()