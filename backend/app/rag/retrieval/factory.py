"""
MỤC ĐÍCH: Tạo retriever theo cấu hình RETRIEVAL_MODE (vector | keyword | hybrid).

Nhờ vậy có thể đổi chiến lược tìm kiếm chỉ bằng .env, hoặc truyền `mode`
để so sánh các chiến lược với nhau (scripts/compare_retrieval.py).
"""

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.retrieval.base import Retriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.vector_retriever import VectorRetriever

RETRIEVAL_MODES = ("vector", "keyword", "hybrid")


def build_retriever(
    db: Session,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    mode: str | None = None,
) -> Retriever:
    s = settings or get_settings()
    mode = (mode or s.retrieval_mode).lower()
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}' (use: {', '.join(RETRIEVAL_MODES)})")

    if mode == "keyword":
        return KeywordRetriever(db)

    vector = VectorRetriever(db, embedder or get_embedding_provider(s))
    if mode == "vector":
        return vector
    return HybridRetriever(vector, KeywordRetriever(db), candidate_k=s.retrieval_candidate_k, rrf_k=s.rrf_k)