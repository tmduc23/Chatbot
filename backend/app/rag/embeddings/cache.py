"""
MỤC ĐÍCH (Phase 12c): cache VECTOR của câu hỏi trong RAM (LRU).

Cùng 1 model, cùng 1 câu -> luôn ra cùng 1 vector. Câu hỏi lặp lại (dù cache câu trả lời không trúng,
vd. vừa upload tài liệu mới) vẫn không phải gọi embedding API lần nữa.

Mẫu thiết kế "decorator": bọc 1 EmbeddingProvider bất kỳ, bên ngoài nhìn vẫn là EmbeddingProvider.
Retriever không biết (và không cần biết) có cache.

LRU (Least Recently Used): đầy thì bỏ câu lâu nhất không ai hỏi. Giới hạn kích thước để RAM không
tăng mãi (1536 số float ~ 12 KB / câu -> 1000 câu ~ 12 MB).

Chỉ cache embed_query. embed_documents (lúc nạp tài liệu) mỗi chunk chỉ embed 1 lần, cache vô ích.
"""
import threading
from collections import OrderedDict

from app.rag.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, inner: EmbeddingProvider, max_size: int = 1000):
        self.inner = inner
        self.dimension = inner.dimension
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        # FastAPI chạy route `def` trong nhiều thread cùng lúc -> OrderedDict cần khoá
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)   # vừa dùng -> đưa về cuối (mới nhất)
                self.hits += 1
                return list(self._cache[text])  # trả bản sao: người gọi lỡ sửa list cũng không hỏng cache
        vector = self.inner.embed_query(text)   # gọi API NGOÀI khoá: không bắt thread khác chờ mạng
        with self._lock:
            self.misses += 1
            self._cache[text] = list(vector)
            self._cache.move_to_end(text)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # bỏ phần tử cũ nhất (đầu danh sách)
        return vector
