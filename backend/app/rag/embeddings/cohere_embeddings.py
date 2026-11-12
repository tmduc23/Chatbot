"""
MỤC ĐÍCH: Embedding provider dùng Cohere Embed API (vd. embed-v5.0-pro, embed-v5.0-fast).

Cùng "hợp đồng" EmbeddingProvider với Gemini -> DocumentService, Retriever không phải sửa gì.

Ba điểm kỹ thuật chính (giống Gemini, khác tên gọi):
  1. input_type: "search_document" khi embed chunk, "search_query" khi embed câu hỏi
     (Gemini gọi là task_type RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY).
  2. output_dimension=EMBEDDING_DIM (1536): khớp cột vector(1536) trong DB.
     embed-v4.0: 256/512/1024/1536; embed-v5.0-pro: 256/512/768/1024/1536/2048;
     model v3 bỏ qua output_dimension (cố định 1024/384) -> không dùng được.
  3. Tối đa 96 text/lần gọi + retry khi 429/5xx (key trial bị giới hạn theo phút).

QUAN TRỌNG: vector của Cohere và của Gemini KHÔNG so sánh được với nhau (2 "không gian" khác nhau),
dù cùng số chiều. Đổi provider = phải embed lại TOÀN BỘ chunk (scripts/reembed_all.py).
"""
import logging
import time
from typing import Any

from app.core.exceptions import EmbeddingError
from app.rag.embeddings.base import EmbeddingProvider, l2_normalize

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503}


class CohereEmbeddingProvider(EmbeddingProvider):
    BATCH_SIZE = 96          # giới hạn của Cohere Embed API
    MAX_RETRIES = 5

    def __init__(self, api_key: str, model: str, dimension: int, client: Any | None = None):
        if client is None:
            import cohere  # import muộn: chỉ cần thư viện khi thật sự dùng Cohere
            client = cohere.ClientV2(api_key=api_key)
        self._client = client
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(self._embed(texts[start:start + self.BATCH_SIZE], "search_document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "search_query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.embed(
                    model=self.model, texts=texts, input_type=input_type,
                    embedding_types=["float"], output_dimension=self.dimension,
                )
                break
            except Exception as exc:  # SDK ném ApiError có .status_code
                code = getattr(exc, "status_code", None)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(60, 2 ** (attempt + 2))  # 4s, 8s, 16s, 32s
                    logger.warning("Cohere embedding HTTP %s, retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise EmbeddingError(f"Cohere embedding failed: {exc}") from exc

        # SDK đặt tên trường là float_ (vì "float" trùng tên kiểu dữ liệu của Python)
        raw = getattr(response.embeddings, "float_", None) or []
        vectors = [l2_normalize(list(v)) for v in raw]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Cohere returned {len(vectors)} embeddings for {len(texts)} texts")
        if any(len(v) != self.dimension for v in vectors):
            raise EmbeddingError(
                f"Cohere returned {len(vectors[0])}-dim vectors, DB needs {self.dimension} "
                f"(model '{self.model}' must support output_dimension={self.dimension}, e.g. embed-v4.0 / embed-v5.0-pro)"
            )
        return vectors
