"""
MỤC ĐÍCH: Tìm các chunk có NGHĨA gần với câu hỏi nhất, bằng pgvector.

    câu hỏi --embed_query --> vector q
    SELECT ... ORDER BY embedding <=> q   (<=> là toán tử cosine distancee của pgvector)
            WHERE application_id = ?      (lọc NGAY trong SQL - yêu cầu bảo mật)
            LIMIT top_k

    score = 1 - cosine_distance =- cosine similarity 
    Lưu ý: score cao KHÔNG có nghĩa chắc chắn đúng — chỉ là "gần nhất trong những gì có".
    Phase 5 (reranker) và Phase 7 (ngưỡng) sẽ xử lý chuyện này.

    Chỉ lấy chunk của document READY: tránh trả về tài liệu đang xử lý dở / bị lỗi.
"""

import logging
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.retrieval.base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    def __init__(self, db: Session, embedder: EmbeddingProvider):
        self.db = db
        self.embedder = embedder

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        # Đo thời gian embedding
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        t1 = time.perf_counter()

        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = (        
            select(Chunk, Document.filename, distance)         # lấy 3 thứ: toàn bộ thông tin Chunk, filename của Document và distance giữa chunk và vector
            .join(Document, Document.id == Chunk.document_id)  # nối 2 bảng
            .where(
                Chunk.application_id == application_id,        # lọc dữ liệu
                Document.status == DocumentStatus.READY,
                Chunk.embedding.is_not(None),                   # chỉ lấy chunk đã có embedding
            )
            .order_by(distance)         # xếp theo distance, top k
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()  # thực hiện để chạy SQL vào PostgreSQL
        t2 = time.perf_counter()

        logger.info(
            "vector_search app=%s top_k=%d results=%d embed_ms=%.0f search_ms=%.0f",
            application_id, top_k, len(rows), (t1 - t0) * 1000, (t2 - t1) * 1000
        )

        return [
            RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page=chunk.page, chunk_index=chunk.chunk_index, text=chunk.text,
                score=1.0 - float(dist)
            )
            for chunk, filename, dist in rows
        ]
        