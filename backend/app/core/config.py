"""
MỤC ĐÍCH: Cấu hình tập trung cho toàn bộ backend.

Mọi giá trị đọc từ biến môi trường / file .env, có giá trị mặc định an toàn.
Code khác chỉ gọi get_settings(), không tự đọc os.environ, không hard-code.
"""

from functools import lru_cache
from pydantic import SecretStr, field_validator
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
    chunk_strategy: str = "size"             # size = theo số token | section = theo đoạn văn + mục (paper)      
    chunk_min_tokens: int = 40               # (section) đoạn ngắn hơn thì gộp với đoạn sau, trong cùng mục
    chunk_section_header: bool = True        # (section) thêm dòng "Tiêu đề > Mục" đầu mỗi chunk
    tokenizer_encoding: str = "o200k_base"   # bộ tokenizer của tiktoken ("whitespace" = đếm từ, cho test)

    embedding_provider: str = "gemini"      # embedding provider, có thể là "gemini" hoặc "openai"
    embedding_model: str = "gemini-embedding-001"
    
    gemini_api_key: SecretStr | None = None         # SecretStr là kiểu dữ liệu bảo mật, khi in ra sẽ không hiển thị giá trị thực

    # Retrieval (Phase 3-5)
    retrieval_mode: str = "hybrid"     # vector | keyword | hybrid
    retrieval_candidate_k: int = 30    # số kết quả MỖI nhánh (vector, keyword) lấy ra trước khi fusion
    rrf_k: int = 60                    # hằng số k trong công thức RRF
    rrf_keyword_weight: float = 1.0    # trọng số nhánh keyword trong RRF (vector luôn 1.0); A/B Phase 11
    keyword_ranking: str = "ts_rank"   # ts_rank (Postgres, không IDF) | bm25 (BM25 tính bằng SQL, có IDF)
                                       # | pg_textsearch (BM25 của extension, cần Postgres 17+, migration 0006)
    bm25_k1: float = 1.2               # BM25 SQL: tf bão hoà nhanh hay chậm (pg_textsearch: đặt trong index)
    bm25_b: float = 0.75               # BM25 SQL: phạt chunk dài (0 = không phạt, 1 = phạt tối đa)
    retrieval_top_k: int = 20          # số kết quả sau fusion (đưa vào reranker ở Phase 5)


    
    reranker_provider: str = "gemini"   
    reranker_model: str = "gemini-3.5-flash-lite"
    reranker_top_k: int = 5
    reranker_max_chars: int = 1000             # cắt bớt mỗi chunk trước khi gửi cho reranker

    # Cohere — key chung, hoặc 2 key riêng cho embed / rerank (key riêng được ưu tiên,
    # để trống thì dùng COHERE_API_KEY). Đọc key qua cohere_key("embed" | "rerank").
    cohere_api_key: SecretStr | None = None
    cohere_embed_api_key: SecretStr | None = None
    cohere_rerank_api_key: SecretStr | None = None

    cohere_embedding_model: str ="embed-v4.0"
    cohere_rerank_model: str = "rerank-v4.0-pro"
    

    # Ngưỡng lọc chunk sau rerank (Phase 6). None = dùng mặc định theo RERANKER_PROVIDER 
    reranker_min_score: float | None = None 
    # Generation (Phase 6) 
    llm_provider: str = "openrouter"               # gemini | fake | open router
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free" 
    llm_temperature: float = 0.2 
    llm_max_output_tokens: int = 2048 
    context_max_chars: int = 6000              # tổng độ dài context đưa vào LLM
    answer_prompt: str = "v4"                  # v3 | v4: phiên bản prompt trả lời, đổi để chạy A/B

    # # OpenRouter (API chuẩn OpenAI) — dùng khi LLM_PROVIDER=openrouter
    openrouter_api_key: SecretStr | None = None
    # 3 key riêng theo vai trò (để trống = dùng OPENROUTER_API_KEY). Đọc qua openrouter_key(role)
    openrouter_answer_api_key: SecretStr | None = None       # trả lời + viết lại câu hỏi nối tiếp
    openrouter_grounding_api_key: SecretStr | None = None    # grounding check (Phase 7)
    openrouter_judge_api_key: SecretStr | None = None        # giám khảo faithfulness (Phase 11)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    
    # Groq (API chuẩn OpenAI) — dùng khi LLM_PROVIDER / GROUNDING_PROVIDER / JUDGE_PROVIDER = groq
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # NVIDIA NIM (build.nvidia.com, API chuẩn OpenAI) — dùng khi LLM_PROVIDER / GROUNDING_PROVIDER / JUDGE_PROVIDER = nvidia
    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Citation + Grounding (Phase 7)
    grounding_check: bool = False               # True = thêm 1 lần gọi LLM mỗi câu để kiểm tra nguồn
    grounding_model: str | None = None          # Nonen = dùng LLM_MODEL
    grounding_provider: str | None = None

    # Giám khảo faithfulness (Phase 11). None = dùng GROUNDING_MODEL, rồi LLM_MODEL. --judge-model thắng tất cả
    judge_model: str | None = None

    # Provider của giám khảo, vd. trả lời bằng groq nhưng chấm bằng openrouter. None = dùng LLM_PROVIDER
    judge_provider: str | None = None

    # API + hội thoại (Phase 9)
    admin_api_key: SecretStr | None = None      # bảo vệ API quản trị (tạo app, upload, xóa ...) qua header X-API-Key
    chat_condense_question: bool = True         # True = viết lại câu hỏi nối tiếp thành câu độc lập (thêm 1 lần gọi LLM)
    chat_history_turns: int = 3                 # số lượt hỏi-đáp gần nhất được đưa vào câu hỏi

    # Xử lý nền + timeout 
    background_ingestion: bool = False          # True = ingestion (Phase 2) chạy nền, trả về ngay 202 Accepted. False = chờ xong mới trả về.
    worker_poll_seconds: float = 2.0            # hàng đợi rỗng -> chờ bao lâu rồi hỏi lại
    worker_max_attempts: int = 3                # số lần thử lại khi ingestion thất bại
    worker_retry_base_seconds: float = 30       # chờ trước lần thử lại: 30s, 60s, 120s... (gấp đôi mỗi lần)
    worker_retry_max_seconds: float = 900       # ... nhưng không quá 15 phút
    worker_stale_after_seconds: int = 900       # PROCESSING quá 15 phút = worker đã chết -> cho worker khác lấy lại
    provider_timeout_seconds: float = 60.0      # tối đa 1 lần gọi embedding / rerank
    llm_timeout_seconds: float = 180.0          # tối đa 1 lần gọi LLM (model reasoning miễn phí khá chậm)
        
    # Đăng nhập + phân quyền (Phase 12b)
    auth_enabled: bool = False                 # false = giữ cách cũ của Phase 9 (chỉ X-API-Key cho API quản trị)
    jwt_secret: SecretStr | None = None        # khoá ký token, >= 32 ký tự ngẫu nhiên. BẮT BUỘC khi AUTH_ENABLED=true
    access_token_minutes: int = 60             # token hết hạn sau bấy nhiêu phút -> đăng nhập lại

    # Rate limiting (Phase 12c): giới hạn số request mỗi người, đếm trong Postgres
    rate_limit_enabled: bool = False
    rate_limit_chat_per_minute: int = 10        # mỗi user (đã đăng nhập) hoặc mỗi IP (Khách)
    rate_limit_chat_per_day: int = 200          # bảo vệ quota LLM miễn phí hàng ngày
    rate_limit_login_per_minute: int = 10       # chống dò mật khẩu (brute force), theo IP
    rate_limit_upload_per_minute: int = 20
    trust_forwarded_for: bool = False           # true chỉ khi backend đứng sau reverse proxy tin cậy


    # Cache(phase 12c)
    answer_cache_enabled: bool = False         # cùng câu hỏi + cùng tài liệu + cùng cấu hình -> trả lại câu trả lời cũ
    answer_cache_ttl_seconds: int = 86400      # câu trả lời trong cache sống tối đa 1 ngày
    embedding_cache_size: int = 1000           # số câu hỏi giữ vector trong RAM (0 = tắt)

    # Logging + monitoring + bảo mật (Phase 12d)
    log_format: str = "text"                   # text (dễ đọc khi dev) | json (production: máy đọc, tìm kiếm, lọc được)
    metrics_token: SecretStr | None = None     # Prometheus gửi "Authorization: Bearer <token>" để đọc /metrics
    expose_api_docs: bool | None = None        # /docs, /openapi.json. None = chỉ bật khi APP_ENV=development


    
    @field_validator("chunk_strategy")
    @classmethod
    def _check_chunk_strategy(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"size", "section"}:
            raise ValueError("CHUNK_STRATEGY phải là size hoặc section")
        return value

    # Dòng "OPENROUTER_JUDGE_API_KEY=" hay "JUDGE_MODEL=" (để trống) trong .env nghĩa là "chưa đặt",
    # không phải chuỗi rỗng "". Không có bước này: key rỗng bị gửi lên OpenRouter, model rỗng bị gọi.
    @field_validator(
        "openrouter_api_key", "openrouter_answer_api_key", "openrouter_grounding_api_key", "openrouter_judge_api_key",
        "groq_api_key", "nvidia_api_key", "grounding_model", "grounding_provider", "judge_model", "judge_provider",
        "admin_api_key", "jwt_secret",   # Phase 12b: "JWT_SECRET=" để trống = chưa đặt
        "gemini_api_key", "cohere_api_key", "cohere_embed_api_key", "cohere_rerank_api_key",   # Phase 12d
        "metrics_token", "expose_api_docs", "reranker_min_score",
        mode="before",
    )
    @classmethod
    def _empty_is_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    def openrouter_key(self, role: str) -> str | None:
        """Key OpenRouter cho 1 vai trò: answer | grounding | judge. Key riêng nếu có, không thì key chung."""
        specific = {"answer": self.openrouter_answer_api_key, "grounding": self.openrouter_grounding_api_key,
                    "judge": self.openrouter_judge_api_key}[role]
        for key in (specific, self.openrouter_api_key):
            if key is not None and key.get_secret_value().strip():
                return key.get_secret_value().strip()
        return None

    def cohere_key(self, purpose: str) -> str | None:
        """Key Cohere cho 1 việc: key riêng (COHERE_EMBED_/RERANK_API_KEY) nếu có, không thì key chung."""
        specific = {"embed": self.cohere_embed_api_key, "rerank": self.cohere_rerank_api_key}[purpose]
        for key in (specific, self.cohere_api_key):
            if key is not None and key.get_secret_value().strip():
                return key.get_secret_value().strip()
        return None
    


@lru_cache      # dùng lại config khi gọi nhiều lần, tránh tạo nhiều instance
def get_settings() -> Settings:
    return Settings()