"""
MỤC ĐÍCH: LLM giả, chạy offline - dùng cho test và để chạy thử pipeline không tốn quota.

Trả về câu trả lời soạn sẵn (mặc định trích [Source 1]) và GHI LẠI mọi lần gọi,
để test kiểm tra RAGService đã gửi gì cho LLM, hoặc KHÔNG gọi LLM khi phải từ chối.
"""

from app.rag.llm.base import LLMResponse, LLMProvider

class FakeLLM(LLMProvider):
    def __init__(self, answer: str = "Fake answer based on the context [Source 1]."):
        self.answer = answer
        self.calls: list[dict] = []

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})
        return LLMResponse(text=self.answer, model="fake")