"""
MỤC ĐÍCH: Tìm các chunk có NGHĨA gần với câu hỏi nhất, bằng pgvector.

    câu hỏi --embed_query --> vector q
    SELECT ... ORDER BY embedding <=> q   (<=> là toán tử cosine distance của pgvector)
            WHERE application_id = ?      (lọc NGAY trong SQL - yêu cầu bảo mật)
            LIMIT top_k

    score = 1 - cosine_distance = cosine similarity 
    Lưu ý: score cao KHÔNG có nghĩa chắc chắn đúng — chỉ là "gần nhất trong những gì có".
    Phase 5 (reranker) và Phase 7 (ngưỡng) sẽ xử lý chuyện này.

    Chỉ lấy chunk của document READY: tránh trả về tài liệu đang xử lý dở / bị lỗi.

    Phase 8 - HNSW + WHERE application_id (pgvector >= 0.8):
        Index HNSW tìm ~40 vector gần nhất TRONG CẢ BẢNG (hnsw.ef_search = 40). RỒI mới áp WHERE.
        Nếu 40 vector đó phần lớn thuộc application KHÁC -> lọc xong còn rất ít, thậm chí 0 kết quả,
        dù application của mình vẫn có chunk. Dữ liệu không bị lộ, nhưng câu trả lời bị "mất".
        hnsw.iterative_scan = strict_order: index quét TIẾP cho tới khi đủ LIMIT kết quả qua WHERE,
        vẫn giữ đúng thứ tự khoảng cách. Đặt bằng set_config(..., chỉ trong transaction này.)
"""

import logging
import time
import uuid
from collections.abc import Sequence

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.retrieval.base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

# "strict_order" | "relaxed_order" | "off". Test dùng "off" để chứng minh lỗi khi không bật
HNSW_ITERATIVE_SCAN = "strict_order"


class VectorRetriever(Retriever):
    def __init__(self, db: Session, embedder: EmbeddingProvider, iterative_scan: str = HNSW_ITERATIVE_SCAN):
        self.db = db
        self.embedder = embedder
        self.iterative_scan = iterative_scan

    def retrieve(self, query: str, application_id: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None) -> list[RetrievedChunk]:
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

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))

        # is_local = True(tham số thứ 3) giống SET LOCAL: chỉ có hiệu lực tới hết transaction hiện tại
        self.db.execute(select(func.set_config("hnsw.iterative_scan", self.iterative_scan, True)))
        rows = self.db.execute(stmt).all()  # thực hiện để chạy SQL vào PostgreSQL
        t2 = time.perf_counter()

        logger.info(
            "vector_search app=%s docs=%s top_k=%d results=%d embed_ms=%.0f search_ms=%.0f",
            application_id, len(document_ids) if document_ids else "all", top_k, len(rows), (t1 - t0) * 1000, (t2 - t1) * 1000,
        )

        return [
            RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page=chunk.page, chunk_index=chunk.chunk_index, text=chunk.text,
                score=1.0 - float(dist)
            )
            for chunk, filename, dist in rows
        ]
        