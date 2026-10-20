"""
MỤC ĐÍCH: LLMProvider cho mọi API theo chuẩn OpenAI (OpenRouter, Groq, OpenAI...).

Nhiều nhà cung cấp dùng chung "giao diện" /chat/completions của OpenAI, chỉ khác base_url và key.
=> 1 class dùng được cho tất cả, dùng thư viện `openai` với base_url của từng bên:
    OpenRouter: https://openrouter.ai/api/v1      model vd. "nvidia/nemotron-3-ultra-550b-a55b:free"
    Groq:       https://api.groq.com/openai/v1    model vd. "openai/gpt-oss-120b"

Giống GeminiLLM ở Phase 6:
  - client=None -> tự tạo OpenAI client; test truyền client giả vào;
  - retry khi HTTP 429/5xx (2s, 4s, 8s, 16s, 30s), bỏ cuộc ngay với 400/401/403/404;
  - không fail-open: LLM lỗi -> LLMError.
Tắt retry có sẵn của thư viện (max_retries=0) để không retry 2 tầng.
"""
import logging
import time
from typing import Any

from app.core.exceptions import LLMError
from app.rag.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503}


class OpenAICompatLLM(LLMProvider):
    MAX_RETRIES = 6

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        client: Any | None = None,
    ):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=180.0)  # model reasoning free chậm (~21 tok/s)
        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        response = self._call(messages)

        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        text = ((choice.message.content if choice else None) or "").strip()
        if not text:
            # Bị chặn, hoặc model "suy nghĩ" (reasoning) dùng hết max_output_tokens trước khi viết xong
            raise LLMError(f"LLM returned an empty answer (finish_reason={getattr(choice, 'finish_reason', None)})")

        usage = getattr(response, "usage", None)
        result = LLMResponse(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
        logger.info("openai_compat_llm model=%s tokens in=%s out=%s llm_ms=%.0f", self.model,
                    result.input_tokens, result.output_tokens, (time.perf_counter() - t0) * 1000)
        return result

    def _call(self, messages: list[dict]) -> Any:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(30, 2 ** (attempt + 1))
                    logger.warning("LLM HTTP %s, retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(f"LLM call failed: {exc}") from exc

            # OpenRouter: provider phía sau lỗi -> HTTP 200 nhưng body là {"error": {...}}, không có choices
            error = getattr(response, "error", None)
            if error and not getattr(response, "choices", None):
                code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
                if code in RETRYABLE_STATUS and attempt < self.MAX_RETRIES - 1:
                    wait = min(30, 2 ** (attempt + 1))
                    logger.warning("LLM provider error %s, retry in %ss: %s", code, wait, error)
                    time.sleep(wait)
                    continue
                raise LLMError(f"LLM provider error: {error}")
            return response
        raise LLMError("unreachable")