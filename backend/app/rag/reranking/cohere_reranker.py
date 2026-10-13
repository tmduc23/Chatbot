"""
MỤC ĐÍCH: Reranker dùng Cohere Rerank API (mặc didjnh rerank-multilingual-v3.0)

Đây là cross-encoder (model nhỏ chuyên chấm cặp câu hỏi-đoạn văn), chạy trên server của Cohere:
    - không cần torch, không tải model -> image Docker nhẹ
    - hỗ trợ đa ngôn ngữ, có tiếng Việt
    - trả về relevance_sccore, sẵn ở thang 0.0 - 1.0

1 lần gọi API cho cả ~20 chunk:
    có .rerank(model= ..., query= ..., documents=[text, ...], top_n=len(documents))
    -> results = [{"index": 2, relevance_score: 0.97}, {"index": 0, ...}, ...]

Vì sao vẫn cắt chunk theo RERANKER_MAX_CHARS?
    Cohere tính phí theo "search unit": 1 lần gọi với tối đa 100 document, mỗi document ngắn.
    Document quá dài bị chia nhỏ và tính thêm. Cắt ~1000 ký tự mỗi lần rerank = 1 search unit

Giống Gemini: retry khi 429/5xx, fail-open (giữ thứ tự retrieval) khi lỗi
Khác Gemini: không đọc được prompt -> bỏ qua tham số domain 

Ba điểm đáng chú ý:
- **`top_n=len(documents)`**: lấy điểm của **mọi** chunk rồi tự cắt `top_k` bằng `rank_by_scores`. Cách này giữ `retrieval_rank` cho `rerank_demo.py`,
 và để Phase 6 lọc theo ngưỡng điểm.
- **Kết quả của Cohere đã được sắp xếp sẵn và dùng `index`** để chỉ chunk gốc. Code đổi lại về thứ tự ban đầu bằng `by_index`, bỏ qua index lạ, và cho index bị thiếu điểm 0.
 Đây là cùng nguyên tắc "không tin tuyệt đối vào API" như ở Gemini.
- **Chỉ retry 429 và lỗi 5xx.** Lỗi 401 (sai key) có retry cũng vô ích, nên fail-open ngay.
"""

import logging
import time
from typing import Any

from app.core.exceptions import RerankError
from app.rag.reranking.base import PassthroughReranker, RerankedChunk, Reranker, rank_by_scores
from app.rag.retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503}
class CohereReranker(Reranker):
    MAX_RETRIES = 5

    def __init__(
        self, 
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
        max_chars: int = 1000,
        fail_open: bool = True,
        client: Any | None = None
    ):
        if client is None:
            import cohere
            client = cohere.ClientV2(api_key=api_key)
        self._client = client
        self.model = model
        self.max_chars = max_chars
        self.fail_open = fail_open

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int, domain: str | None = None) -> list[RerankedChunk]:
        # domain bị bỏ qua: cross-encoder không nhận hướng dẫn, chỉ nhìn cặp (câu hỏi, chunk)
        if not chunks:
            return []
        t0 = time.perf_counter()
        try: 
            scores = self._score(query, [c.text for c in chunks])
        except RerankError:
            if not self.fail_open:
                raise
            logging.warning("Cohere rerank failed, keeping retrieval order", exc_info=True)
            return PassthroughReranker().rerank(query, chunks, top_k)

        result = rank_by_scores(chunks, scores, top_k)
        logger.info("coher_rerank model=%s candidates=%d kept=%d rerank_ms=%.0f",
                     self.model, len(chunks), len(result), (time.perf_counter() - t0) * 1000)
        return result

    def _score(self, query: str, texts: list[str]) -> list[float]:
        documents = [t[:self.max_chars] for t in texts]
        # top_n = tất cả: lấy điểm của MỌI chunk, tự cắt top_k bằng rank_by_scores như các reranker khác
        response = self._call(query, documents)

        by_index: dict[int, float] = {}
        for item in getattr(response, "results", None) or []:
            if 0 <= item.index < len(texts):    # không tin tuyệt đối vào API
                by_index[item.index] = float(item.relevance_score)
        if not by_index:
            raise RerankError("Cohere rerank returned no results")
        if len(by_index) < len(texts):
            logger.warning("Cohere returned %d/%d scores; missing ones get 0", len(by_index), len(texts))
        return [by_index.get(i, 0.0) for i in range(len(texts))]

    # _call giúp xử lý lỗi  và retry
    def _call(self, query: str, documents: list[str]) -> Any:
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._client.rerank(model=self.model, query=query,
                                           documents=documents, top_n=len(documents))

            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(30, 2 ** (attempt + 1))      # 2s, 4s, 8s
                    logger.warning("Cohere rerank HTTP %s, retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise RerankError(f"Cohere rerank call failed: {exc}") from exc
        raise RerankError("unreachable")
