"""
MỤC ĐÍCH: Kết hợp Vector Search + Keyword Search bằng RRF.

                     Query
              ┌────────┴────────┐
              ▼                 ▼
        VectorRetriever   KeywordRetriever     (mỗi nhánh lấy candidate_k kết quả,
              │                 │               cả hai đều lọc application_id trong SQL)
              └────────┬────────┘
                       ▼
            reciprocal_rank_fusion
                       ▼
                 top_k kết quả

Khả năng chịu lỗi: nếu Gemini API lỗi (hết quota, mất mạng), nhánh vector trả về rỗng
và hệ thống vẫn chạy được bằng keyword search thay vì sập hoàn toàn.
"""
import logging
import time
from app.core.exceptions import EmbeddingError
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval.fusion import reciprocal_rank_fusion


logger = logging.getLogger(__name__)

class HybridRetriever(Retriever):
    def __init__(self, vector: Retriever, keyword: Retriever, candidate_k: int = 30, rrf_k: int = 60):
        self.vector = vector
        self.keyword = keyword
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        try:
            vector_results = self.vector.retrieve(query, application_id, self.candidate_k)
        except EmbeddingError:
            logger.warning("Vector search unavailable, falling back to keyword-only", exc_info=True)
            vector_results = []
        t1 = time.perf_counter()
        keyword_results = self.keyword.retrieve(query, application_id, self.candidate_k)
        t2 = time.perf_counter()

        fused = reciprocal_rank_fusion(
            {"vector": vector_results, "keyword": keyword_results}, k=self.rrf_k
        )[:top_k]

        logger.info(
            "hybrid_search app=%s vector=%d keyword=%d fused=%d vector_ms=%.0f keyword_ms=%.0f",
            application_id, len(vector_results), len(keyword_results), len(fused),
            (t1 - t0) * 1000, (t2 - t1) * 1000,
        )
        return fused