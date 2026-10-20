"""
MỤC ĐÍCH: Ghép Retrieval + Reranking thành MỘT BƯỚC, dùng chung cho script và (Phase 6) RAGService.
    query -- retriever --> top RETRIEVAL_TOP_K (20) -- reranker --> top RERANKER_TOP_K (5)

Ghi log điểm TRƯỚC và SAU rerank cho từng chunk (yêu cầu của spec Phase 5).
Ví dụ: #3: rank 7 -> 1 | retrieval=0.0154 rerank=0.90
Nhờ log này thấy ngay reranker đã "kéo" chunk nào lên, đẩy chunk nào xuống.

Lọc theo ngưỡng (Phase 6) - reranker đóng vai "giám khảo" (judge):
    Reranker luôn trả về top 5, kể cả khi CẢ 5 đều không liên quan (vd. hỏi "mức lương mong muốn?"
    mà CV không ghi). Ngưỡng loại các chunk điểm thấp -> RAGService thấy 0 chunk thì từ chối
    trả lời mà KHÔNG gọi LLM (không tốn token, không cho LLM cơ hội bịa).
    Chunk có judged=False (reranker lỗi rồi fail-open, hoặc RERANKER_PROVIDER=none) KHÔNG bị lọc:
    điểm của chúng là điểm retrieval, khác thang đo -> lọc sẽ loại sạch và từ chối mọi câu hỏi.

Bảo mật: reranker chỉ SẮP XẾP LẠI những gì retriever trả về; retriever đã lọc
application_id trong SQL -> reranker không thể làm lộ dữ liệu của application khác.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranking import RerankedChunk, Reranker, get_min_score, get_reranker
from app.rag.retrieval import RetrievedChunk, Retriever, build_retriever
from app.repositories import application_repository as app_repo

logger = logging.getLogger(__name__)

# application_id -> mô tả bộ tài liệu (hoặc None). Tách thành hàm để test không cần DB.
DomainResolver = Callable[[str], str | None]

"""
# application_domain giúp hướng tới mục tiêu trả lời cụ thể hơn

resolver = application_domain_resolver(db)  # Tạo hàm resolver

resolver("cv")        # Truy vấn database lần 1
resolver("projects")  # Truy vấn database lần 2
resolver("cv")        # Truy vấn database lần 3
"""
def application_domain_resolver(db: Session) -> DomainResolver:
    """Đọc applications.description -> bối cảnh cho prompt của reranker và LLM."""
    def resolve(application_id: str) -> str | None:
        app = app_repo.get(db, application_id)
        return app.description if app is not None else None     # trả về application_description
    return resolve


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    application_id: str
    candidates: list[RetrievedChunk]       # trước rerank: chunk từ hybrid search
    reranked: list[RerankedChunk]          # sau rerank, CHƯA lọc ngưỡng (compare_retrieval dùng cái này)
    retrieval_ms: float
    rerank_ms: float
    relevant: list[RerankedChunk] = field(default_factory=list)   # sau rerank VÀ qua ngưỡng -> đưa vào LLM
    min_score: float = 0.0


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        domain_resolver: DomainResolver | None = None,
        min_score: float = 0.0,
    ):
        if rerank_top_k > retrieval_top_k:
            raise ValueError("RERANKER_TOP_K must be <= RETRIEVAL_TOP_K")
        self.retriever = retriever
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.domain_resolver = domain_resolver
        self.min_score = min_score

    # Nếu hết request thì xử lý bằng cách lấy thứ tự từ xếp hạng hybrid
    def passes(self, r: RerankedChunk) -> bool:
        """Chunk có qua ngưỡng không. Chunk không được reranker chấm (judged=False) luôn qua, để LLM quyết định."""
        return not r.judged or r.score >= self.min_score

    def run(self, query: str, application_id: str) -> RetrievalResult:
        t0 = time.perf_counter()

        # retrive
        candidates = self.retriever.retrieve(query, application_id, self.retrieval_top_k)       # truy xuất các chunk từ hybrid
        t1 = time.perf_counter()
        domain = self.domain_resolver(application_id) if self.domain_resolver else None
        # rerank
        reranked = self.reranker.rerank(query, candidates, self.rerank_top_k, domain=domain)    # rerank bằng LLM
        t2 = time.perf_counter()
        relevant = [r for r in reranked if self.passes(r)]

        logger.info("retrieval_pipeline app=%s candidates=%d kept=%d relevant=%d min_score=%.2f "
                    "retrieval_ms=%.0f rerank_ms=%.0f", application_id, len(candidates), len(reranked),
                    len(relevant), self.min_score, (t1 - t0) * 1000, (t2 - t1) * 1000)
        for new_rank, r in enumerate(reranked, start=1):
            logger.info("  %s#%d: rank %d -> %d | retrieval=%.4f rerank=%.2f %s",
                        r.chunk.filename, r.chunk.chunk_index, r.retrieval_rank, new_rank,
                        r.chunk.score, r.score, "✓" if self.passes(r) else "✗")

        # Có chunk, nhưng không chunk nào được reranker chấm điểm (tất cả judged=False) thì in cảnh báo.
        if reranked and not any(r.judged for r in reranked):
            logger.warning("Chunks were not scored by a reranker; min_score filter skipped")

        return RetrievalResult(query, application_id, candidates, reranked,
                               (t1 - t0) * 1000, (t2 - t1) * 1000, relevant, self.min_score)


# Retrieval + Reranking thành MỘT BƯỚC
def build_retrieval_pipeline(
    db: Session,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    retrieval_mode: str | None = None,
    min_score: float | None = None,
) -> RetrievalPipeline:
    s = settings or get_settings()   # không truyền settings thì đọc cấu hình từ .env
    return RetrievalPipeline(
        retriever=build_retriever(db, s, embedder=embedder, mode=retrieval_mode),
        reranker=reranker or get_reranker(s),
        retrieval_top_k=s.retrieval_top_k,
        rerank_top_k=s.reranker_top_k,
        domain_resolver=application_domain_resolver(db),
        min_score=get_min_score(s) if min_score is None else min_score,
    )
