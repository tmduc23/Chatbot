"""
MỤC ĐÍCH: Định nghĩa interface chung cho mọi embedding provider.

Business logic (DocumentService, Retriever) chỉ biết đến EmbeddingProvider.
Không biết đang dùng Gemini, OpenAI hay model local.
=> Đổi provider chỉ cần viết 1 class mới + sửa EMBEDDING_PROVIDER trong .env

Hai hàm tách riêng vì nhiều model (gồm Gemini) embed tài liệu và câu hỏi
theo cách hơi khác nhau (task_type) để tối ưu cho tìm kiếm
"""
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    dimension: int 

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed các chunk tài liệu (lúc indexing)"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed câu hỏi của người dùng (lúc search)"""


def l2_normalize(vector: list[float]) -> list[float]:
    """
    Đưa vector về độ dài 1.
    Gemini chỉ tự chuẩn hóa khi dùng 3072 chiều;
    Rút gọn 768 chiều thì phải tự chuẩn hóa điểm cosine giữa các vector so sánh được với nhau
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]