"""
MỤC ĐÍCH: QuestionCondenser - dùng LLM viết lại câu hỏi nối tiếp thành câu độc lập (Phase 9)

Nguyên tắc FAIL-OPEN: bước này chỉ để "cải thiện" retrieval, KHÔNG được làm hỏng câu hỏi.
    - LLM lỗi (hết quota, mạng ...)     -> dùng nguyên câu hỏi gốc, không báo lỗi cho người dùng
    - LLM trả rỗng / quá dài / lạc đề   -> dùng nguyên câu hỏi gốc
    - Không có lịch sử (câu đầu tiên)   -> không gọi LLM (tiết kiệm 1 request)
"""

import logging
import time
from dataclasses import dataclass

from app.core.exceptions import LLMError
from app.rag.llm import LLMProvider
from app.rag.prompts.condense import build_condense_prompt, CONDENSE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

MAX_STANDALONE_CHARS = 1000             # = MAX_QUESTION_CHARS của RAGService


@dataclass(frozen=True)
class CondensedQuestion:
    question: str                       # câu sẽ đưa vào RAG (đã viết lại, hoặc câu gốc)
    rewritten: bool                     # True = LLM đã viết lại thành câu KHÁC câu gốc
    ms: float


class QuestionCondenser:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def condense(self, question: str, history: list[tuple[str, str]]) -> CondensedQuestion:
        if not history:
            return CondensedQuestion(question, False, 0.0)
        t0 = time.perf_counter()
        try:
            response = self.llm.generate(build_condense_prompt(history, question), CONDENSE_SYSTEM_INSTRUCTION)
        except LLMError as exc:
            logging.warning("condense failed, using original question: %s", exc)
            return CondensedQuestion(question, False, (time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000

        # Model hay thêm nhãn/ngoặc kép: "Standalone question: "..."" -> bóc ra
        text = response.text.strip().strip('"').strip()             # bỏ khoảng trắng -> bỏ dấu " 2 đầu -> bỏ khoảng trắng
        if text.lower().startswith("standalone question:"):         
            text = text.split(":", 1)[1].strip().strip('"').strip() # bỏ dấu : -> bỏ dấu " -> lại bỏ khoảng trắng
        if not text or len(text) > MAX_STANDALONE_CHARS or "\n\n" in text:
            logging.warning("condense return unusable text, using original question")
            return CondensedQuestion(question, False, ms)

        logger.info("condensed question in %.0f ms: %r -> %r", ms, question, text)
        return CondensedQuestion(text, text !=question, ms)