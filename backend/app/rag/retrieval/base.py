"""
MỤC ĐÍCH: Interface chung cho mọi Retriever + kiểu dữ liệu kết quả.

Phase 3 có VectorRetriever.
Phase 4 thêm KeywordRetriever và HybridRetriever
- tất cả trả về cùng kiểu RetrievedChunk, nên phần còn lại cảu hệ thống
(reranker, context builder) không quan tâm kết quả từ đâu.

'application_id' là tham số BẮT BUỘC: không có cách nào gọi retrive()
mà quên lọc theo application
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

# frozen : read-only
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float    # càng cao càng liên quan


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        ...