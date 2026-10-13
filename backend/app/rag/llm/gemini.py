"""
MỤC ĐÍCH: LLMProvider dùng Gemini để sinh câu trả lời.

Giống GeminiReranker ở phase 5:
    - client = None -> tự tạo genai.Client: test truyền client giả vào (dependency injection);
    - retry khi HTTP 429/5xx (2s, 4s, 8s, 16s, 30s), bỏ cuộc ngay vơi 400/401/403/404.
Khác reranker: KHÔNG fail-open. Reranker lỗi thì vẫn còn thứ tự retrieval để dùng,
còn LLM lỗi thì không có câu trả lời nào để trả -> raise LLMError, tầng API (phase 9) đổi thành HTTP 503.

temperature thấp (0.2) : câu trả lời bám sát context, ít "sáng tạo". Không đặt 0 tuyệt đối
để câu văn bớt cứng; muốn kết quả lặp lại y hệt khi debug thì đặt LLM_TEMPERATURE = 0
"""

import logging
import time
from typing import Any

from app.core.exceptions import LLMError
from app.rag.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 503}

class GeminiLLM(LLMProvider):
    # 6 lần -> chờ tối đa 2+4+8+16+30 = 60s: đủ qua 1 "phút" của giới hạn RPM(free tier ~5 RPM)
    MAX_RETRIES = 6

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        client: Any | None = None,
    ):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)

        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        from google.genai import types

        # config theo SDK yêu cầu
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        # Timer rồi gọi API (_call để có retry)
        t0 = time.perf_counter()
        response = self._call(prompt, config)

        # Nếu response không có .text thì lấy "None" thay vì crash, hoặc chuỗi rỗng
        text = (getattr(response, "text", None) or "").strip()

        # Gemini có thể trả 200 OK nhưng **không có chữ nào**, vì hai lý do:
        # - `SAFETY`: bộ lọc an toàn chặn.
        # - `MAX_TOKENS`: model "thinking" dùng hết 2048 token để suy nghĩ, chưa kịp viết câu trả lời.
        if not text:
            # Bị bộ lọc an toàn chặn, hoặc hết max_output_tokens trước khi viết xong (model có "thinking")
            raise LLMError(f"Gemini returned an empty answer (finish_reason={_finish_reason(response)})")

        # Lấy số token vào ra để tính chi phí
        # Thiếu thông tin trả về None
        usage = getattr(response, "usage_metadata", None)
        result = LLMResponse(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None)
        )

        logger.info("gemini_llm model=%s  tokens in=%s out=%s llm_ms=%.0f", self.model,
                    result.input_tokens, result.output_tokens, (time.perf_counter() - t0) * 1000)
        return result

    # _call() -> gọi API, tự retry khi 429/500/503
    def _call(self, prompt: str, config: Any) -> Any:
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._client.models.generate_content(model=self.model, contents=prompt, config=config)       # thành công -> thoát luôn
            except Exception as exc:
                code = getattr(exc, "code", None)                               # lỗi HTTP của SDK có .code (429, 503,...)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(30, 2 ** (attempt + 1))       # 2s, 4s, 8s, 16s, 30s
                    logger.warning("Gemini LLM HTTP %s, retry in %ss", code, wait)
                    time.sleep(wait)
                    continue                # bỏ qua, qua lên vòng for để tiến hành gọi lại API
                raise LLMError(f"Gemini LLM call failed: {exc}") from exc       # lỗi khác, hoặc hết lượt
        raise LLMError("unreachable")

# Response của Gemini có danh sách "candidates" (các phương án trả lời ta chỉ xin 1)
# Lấy "finish reason của phương án đầu tiên (`STOP`, `MAX_TOKENS`, `SAFETY`...). `or []"
# if candidates để không crash khi danh sách rỗng hoặc None
"""
response.candidates = [
    candidate_1,  # finish_reason = STOP
    candidate_2,  # finish_reason = MAX_TOKENS
]
"""
def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    return str(getattr(candidates[0], "finish_reason", None)) if candidates else None