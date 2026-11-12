"""
MỤC ĐÍCH: Interface chung cho mọi Retriever + kiểu dữ liệu kết quả.

Phase 3 có VectorRetriever.
Phase 4 thêm KeywordRetriever và HybridRetriever
- tất cả trả về cùng kiểu RetrievedChunk, nên phần còn lại cảu hệ thống
(reranker, context builder) không quan tâm kết quả từ đâu.

'application_id' là tham số BẮT BUỘC: không có cách nào gọi retrive()
mà quên lọc theo application

'document_id' (chế độ paper): None = tìm trong CẢ application (như cũ);
[id1, id2] = chỉ tìm trong các tài liệu này (người dùng đã chọn paper để hỏi)
Lọc tài liệu luôn đi KÈM lọc application, không thay thế nói: id của tài liệu thuộc
application khác -> không có kết quả, chứ không bị lộ dữ liệu.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence

# frozen : read-only
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    text: str
    # càng cao càng liên quan. Ý nghĩa tùy retriever:
    # vector -> cosine similarity [-1, 1]
    # keyword -> ts_rank_cd (>=0, không có trần)
    # hybrid -> điểm RRF (vd. 0.016 -0.033)
    # không so sánh điểm giữa các retriever khác nhau
    score: float    # càng cao càng liên quan
    # Hybrid: thứ hạng của chunk trong từng nhánh, vd. {"vector": 1, "keyword": 4} — để debug
    ranks: dict[str, int] = field(default_factory=dict, compare=False)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, application_id: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None) -> list[RetrievedChunk]:
        ...