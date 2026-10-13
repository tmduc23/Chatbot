"""
MỤC ĐÍCH: Reranker dùng LLM(Gemini) chấm điểm mức liên quan của chunk.


Cách làm ("LLM-as-a-reranker", chấm theo lô - 1 lần gọi API cho cả ~20 chunk):
    1. Gửi câu hỏi + các chunk (đã đánh số id, cắt tối đa RERANKER_MAX_CHARS ký tự).
    2. Yêu cầu Gemini trả về JSON [{"id": 0, "score": 8}, ...] theo schema cố định.
        (structured output) với temperature=0 -> kết quả ổn định, parse được.
    3. Đối điểm 0-10 về 0.0-1.0 sắp xếp, giữ top_k

So với cross-encoder chuyên dụng (Cohere Rerank, bge-reranker):
    + không cần thêm API key / model, hiểu tốt tiếng Việt;
    - chậm hơn (~1-3 giây), tốn token, điểm kém "mịn" hơn.
Vì có interface Reranker, sau này đổi sang cross-encoder chỉ cần thêm 1 class.

fail_open = True: nếu Gemini  lỗi/hết quota, trả về thứ tự retrieval ban đầu (kèm cảnh báo)
thay vì làm hỏng cả câu trả lời. Chất lượng giảm nhẹ nhưng hệ thống vẫn chạy.
"""


import logging
import time
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.core.exceptions import RerankError
from app.rag.prompts.rerank import build_rerank_prompt, build_rerank_system_instruction
from app.rag.reranking.base import PassthroughReranker, RerankedChunk, Reranker, rank_by_scores
from app.rag.retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 503}

# BaseModel: lớp nền của Pydantic, cung cấp khả năng kiểm tra và chuyển đổi dữ liệu theo kiểu đã khai báo
class RerankScore(BaseModel):
    id: int
    score: float = Field(ge=0, le=10)       # điểm số là 0 <= score <= 10

# Bộ xác thực 1 danh sách các đối tuowjgn RerankScore
_SCORES = TypeAdapter(list[RerankScore])    # TypeAdapter : công cụ Pydantic dùng để xác thực và chuyển đổi dữ liệu theo kiểu được chỉ định


class GeminiReranker(Reranker):
    MAX_RETRIES = 6

    def __init__(
        self,
        api_key: str,
        model: str,
        max_chars: int = 1000,
        fail_open: bool = True,
        client: Any | None = None,
    ):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.max_chars = max_chars
        self.fail_open = fail_open

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int, domain: str | None = None 
    ) -> list[RerankedChunk]:
        if not chunks:
            return []
        t0 = time.perf_counter()
        try:
            scores = self._score(query, [c.text for c in chunks], domain)
        except RerankError:
            if not self.fail_open:          # fail_open=False thì nems lỗi lên tầng gọi bên ngoài
                raise
            logging.warning("Gemini rerank failed, keep retrieval order", exc_info=True)
            return PassthroughReranker().rerank(query, chunks, top_k)

        result = rank_by_scores(chunks, scores, top_k)           
        logger.info("gemini_rerank model=%s candidates=%d kept=%d reranks_ms=%.0f",
                    self.model, len(chunks), len(result), (time.perf_counter() - t0) * 1000)
        return result

    # Cắt văn bản, xây dựng prompt, gọi Gemini
    def _score(self, query: str, texts: list[str], domain: str | None = None) -> list[float]:
        from google.genai import types

        passages = [t[: self.max_chars] for t in texts]

        # Cấu hình gọi gemini
        # config: là chứa system instruction, temperature và yêu cầu trả về JSON
        config = types.GenerateContentConfig(
            system_instruction=build_rerank_system_instruction(domain),
            temperature=0.0,
            response_mime_type="application/json",                  # yêu cầu đầu ra định dạng JSON
            response_schema=list[RerankScore]                       # yêu cầu đầu ra tuân theo danh sách Rankscore, chỉ định cấu trúc dữ liệu mong muốn, gồm id và score
        )

        response = self._generate(build_rerank_prompt(query, passages), config)

        # Theo dõi lượng token sử dụng
        usage = getattr(response, "usage_metadata", None)                   # thông tin sử dụng token của request
        if usage is not None:   # phục vụ theo dõi chi phí (spec mục 27)
            logger.info("gemini_rerank tokens in=%s out=%s",                # ghi thông tin vào log để theo dõi chi phí
                        getattr(usage, "prompt_token_count", None),         # prompt_token_count: số token đầu vào
                        getattr(usage, "candidates_token_count", None))     # candidates_token_count: số token đầu ra
        try:
            # kiểm tra xem có trả về đúng định dạng RerankScore không, nếu không có trả về chuỗi rỗng 
            parsed = _SCORES.validate_json(response.text or "")             
        except ValidationError as exc:
            raise RerankError(f"Invalid rerank JSON: {exc}") from exc

        by_id = {item.id: item.score for item in parsed if 0 <= item.id < len(texts)}       # trả về dạng id:score (dict)
        if len(by_id) < len(texts):                                     # không tin tuyệt đối vào LLM, nếu Gemini "bịa" ra id không tồn tại, id đó bị bỏ qua, còn id bị thiếu thì nhận điểm 0
            logger.warning("Reranker returned %d/%d scores; missing ones get 0", len(by_id), len(texts))        
        return [by_id.get(i, 0.0) / 10.0 for i in range(len(texts))]

    # Đọc exc.code, chỉ retry mã trong RETRYABLE_STATUS,
    def _generate(self, prompt: str, config: Any) -> Any:
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._client.models.generate_content(model=self.model, contents=prompt, config=config)
            except Exception as exc:
                code = getattr(exc, "code", None)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(30, 2 ** (attempt + 1))      # 2s, 4s, 8s
                    logger.warning("Gemini rerank HTTP %s, retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise RerankError(f"Gemini rerank call failed: {exc}") from exc
        raise RerankError("unreachable")
