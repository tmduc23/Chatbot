"""
MỤC ĐÍCH: Chọn embedding provider dựa trên cấu hình EMBEDDING_PROVIDER.

Đây là chỗ DUY NHẤT trong code biết tên các provider cụ thể.
Mọi nơi khác gọi get_embedding_provider() và nhận về một EmbeddingProvider.

Kiểm tra luôn số chiều của provider khớp với cột vector(768) trong DB,
để phát hiện cấu hình sai ngay lúc khởi tạo thay vì lúc ghi DB.
"""

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingError
from app.models import EMBEDDING_DIM
from app.rag.embeddings.base import EmbeddingProvider


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    name = s.embedding_provider.lower()

    if name == "gemini":
        if s.gemini_api_key is None or not s.gemini_api_key.get_secret_value():
            raise EmbeddingError("GEMINI_API_KEY is not set in .env")
        from app.rag.embeddings.gemini import GeminiEmbeddingProvider

        provider: EmbeddingProvider = GeminiEmbeddingProvider(
            api_key=s.gemini_api_key.get_secret_value(),
            model=s.embedding_model,
            dimension=EMBEDDING_DIM
        )

    elif name == "fake":
        from app.rag.embeddings.fake import FakeEmbeddingProvider

        provider = FakeEmbeddingProvider(dimension=EMBEDDING_DIM)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}' (use: gemini | fake)")


    if provider.dimension != EMBEDDING_DIM:
        raise ValueError(f"Provider dimension {provider.dimension} != DB column dimension {EMBEDDING_DIM}")
    return provider