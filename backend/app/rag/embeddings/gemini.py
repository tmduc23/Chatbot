"""
MỤC ĐÍCH: Embedding provider dùng Google Gemini API (model gemini-embedding-001)\

Ba điểm kĩ thuật chính:
1. task_type: RETRIEVAL_DOCUMENT khi embed chunk, RETRIEVAL_QUERY khi embed câu hỏi
    -> model tối ưu vector cho bài toán tìm kiếm

2. output_dimension = 768 + tự l2-normalize 

3. Batch 100 text/ lần gọi API + retry với backoff khi bị giới hạn tốc độ (HTTP 429)
    -> quan trọng với free tier
"""

import logging
import time
from typing import Any

from app.core.exceptions import EmbeddingError
from app.rag.embeddings.base import EmbeddingProvider, l2_normalize

logger = logging.getLogger(__name__)


# 429: Too many requests
# 500: Internal Server Error
# 503: Service Unavailable
RETRYABLE_STATUS = {429, 500, 503}

# gemini-embedding-2: nhiệm vụ nằm trong chính văn bản thay cho task_type
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


def uses_task_type(model: str) -> bool:
    """Chỉ dòng gemini-embedding-001 (và cũ hơn) nhận task_type; từ gemini-embedding-2 thì không.

    >>> uses_task_type("gemini-embedding-001"), uses_task_type("gemini-embedding-2")
    (True, False)
    """
    return not model.startswith("gemini-embedding-2")


class GeminiEmbeddingProvider(EmbeddingProvider):
    BATCH_SIZE = 100
    MAX_RETRIES = 6

    def __init__(self, api_key: str, model: str, dimension: int, client: Any | None = None):
        if client is None:
            from google import genai # import muộn: chỉ cần thư viện khi thật sự dùng Gemini
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.dimension = dimension


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(self._embed(texts[start:start + self.BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types      # import các kiểu dữ liệu/config của Google GenAI SDK

        contents: list = texts              # MỚI: thứ gửi lên API (mặc định = list chuỗi, như bản 001)

        # Cấu hình config
        if uses_task_type(self.model):
            config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension)
        else:
            prefix = QUERY_PREFIX if task_type == "RETRIEVAL_QUERY" else DOCUMENT_PREFIX
            config = types.EmbedContentConfig(output_dimensionality=self.dimension)
            # MỚI: gemini-embedding-2 coi list chuỗi là các PHẦN của 1 nội dung -> trả về 1 vector chung.
            # Bọc mỗi chuỗi thành 1 Content riêng -> mỗi chuỗi 1 vector.
            contents = [types.Content(parts=[types.Part(text=prefix + t)]) for t in texts]

        for attempt in range(self.MAX_RETRIES):
            try:
                # Gọi Gemini Embedding API, trả về response, thành công thì out vòng for
                response = self._client.models.embed_content(model=self.model, contents=contents, config=config)  # ĐỔI: contents=contents
                break
            except Exception as exc: # SDK ném APIError có thuộc tính .code (HTTP status)
                code = getattr(exc, "code", None)       # lấy thuộc tính của exception, nếu không có trả về None
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:     # retry chỉ backoff khi gặp 429, 500, 503
                    wait = min(60, 2 ** (attempt + 2))  # 4s, 8s, 16s, 32s
                    logger.warning("Gemini embedding HTTP %s retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc

        # Lấy kết quả chuyển value của embedding thành list, sau đó chuẩn hóa
        vectors = [l2_normalize(list(e.values)) for e in response.embeddings]
        if len(vectors) != len(texts) or any(len(v) != self.dimension for v in vectors):
            raise EmbeddingError("Gemini returned unexpected number/size of embeddings")
        return vectors