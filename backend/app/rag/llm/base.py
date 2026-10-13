"""
MỤC ĐÍCH: "Interface" (hợp đồng) chung cho mọi LLLM ssinh câu trả lời

RAGService chỉ biết LLLMProvider.generate(), không biết bên đưới là Gemini, OpenAI hay model local
(spec mục 2: "LLM: thiết kế abstraction ddeer có thể dùng")

=> Đổi LLM = viết 1 class mới + sửa LLM_PROVIDER trong .env. Giống EmbeddingProvider và Reranker
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None    # để theo dõi chi phí
    output_tokens: int | None = None


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        """prompt: nội dung (context + câu hỏi). system_instruction: luật chơi cho model."""