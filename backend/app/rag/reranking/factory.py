"""
MỤC ĐÍCH: Chọn reranker theo cấu hình RERANKER_PROVIDER (gemini | cohere | none | fake)

"none" hữu ích để so sánh: chạy cùng bộ câu hỏi với không có reranker
"""

from app.core.config import Settings, get_settings
from app.rag.reranking.base import PassthroughReranker, Reranker
from app.rag.clients import make_cohere_client, make_gemini_client

PROVIDERS = "gemini | coher | none | fake"

# Ngưỡng lọc mặc định sau rerank (Phase 6). Hai reranker có THANG ĐIỂM KHÁC NHAU:
#   Gemini: 0-10 chia 10 -> mức tròn 0.0, 0.1 ... 1.0; 0.4 = "liên quan một phần" trong prompt
#   Cohere: relevance 0-1 liên tục; chunk không liên quan thường < 0.05
# Đây chỉ là điểm xuất phát: chỉnh lại bằng scripts/tune_threshold.py trên bộ eval của bạn.
DEFAULT_MIN_SCORES = {"gemini": 0.4, "cohere": 0.1}

def get_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    name = s.reranker_provider.lower()
    if name == "none":
        return PassthroughReranker()
    if name == "fake":
        from app.rag.reranking.fake import FakeReranker
        return FakeReranker()
    
    
    if name == "gemini":
        if s.gemini_api_key is None or not s.gemini_api_key.get_secret_value():
            raise ValueError("GEMINI_API_KEY is not set in .env")
        from app.rag.reranking.gemini import GeminiReranker
        return GeminiReranker(api_key=s.gemini_api_key.get_secret_value(),
                              model=s.reranker_model, max_chars=s.reranker_max_chars,
                              client=make_gemini_client(s.gemini_api_key.get_secret_value(),
                                                        s.provider_timeout_seconds))

    if name == "cohere":
        api_key = s.cohere_key("rerank")
        if api_key is None:
            raise ValueError("COHERE_RERANK_API_KEY (or COHERE_API_KEY) is not set in .env")
        from app.rag.reranking.cohere_reranker import CohereReranker
        return CohereReranker(api_key=api_key,
                              model=s.cohere_rerank_model, max_chars=s.reranker_max_chars,
                              client=make_cohere_client(api_key, s.provider_timeout_seconds))


    raise ValueError(f"Unknown RERANKER_PROVIDER '{s.reranker_provider}' (use: {PROVIDERS})")

def get_min_score(settings: Settings | None = None) -> float:
    """RERANKER_MIN_SCORE trong .env nếu có, không thì mặc định theo provider (none/fake -> 0.0)"""
    s = settings or get_settings()
    if s.reranker_min_score is not None:
        return s.reranker_min_score
    return DEFAULT_MIN_SCORES.get(s.reranker_provider.lower(), 0.0)