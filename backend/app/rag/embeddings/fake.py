"""
MỤC ĐÍCH: Emebedding proiver Giả, chạy offline - dùng cho test và khi chưa có API key.

Cách hoạt động (bag-of-words hashing): mỗi từ được băm vào một vị trí 
trong vector 768 chiều. Hai đoạn text có chung nhiều từ -> vector giống nhau -> cosine cao
Không "hiểu nghĩa" như model thật (không biết "car" ~ "automobile)
nhưng đủ để test pipeline một cách nhanh, miễn phí và luôn cho cùng một kết quả
"""

import hashlib
import re

from app.rag.embeddings.base import EmbeddingProvider, l2_normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)



class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int):
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            v[index] += 1.0
        return l2_normalize(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)