"""
MỤC ĐÍCH: Kiểm tra câu trả lời có BÁM NGUỒN không (grounding / faithfulness), TÙY CHỌN.

    answer + context -> LLM (temperature 0) -> SUPPORTED | PARTIAL | UNSUPPORTED (+ lý do)

Bật bằng GROUNDING_CHECK=true trong .env. Mặc định TẮT vì mỗi câu hỏi tốn thêm 1 lần gọi LLM
(free tier gemini-3.8-flash chỉ 20 request/ngày -> bật lên thì còn 10 câu hỏi/ngày).

Checker lỗi (hết quota, mạng...) hoặc trả về format lạ -> verdict UNKNOWN và GIỮ câu trả lời (fail-open):
câu trả lời đã qua 2 lớp bảo vệ của Phase 6, không nên vứt đi chỉ vì bước kiểm tra phụ bị lỗi.

Đây là "LLM-as-judge": một LLM chấm LLM. Điểm yếu đã biết: cùng họ model có thể dễ dãi với nhau,
và có thể chấm sai -> trong evaluation phải tự đọc lại một mẫu để so (xem scripts/eval_generation.py).
"""

import logging
import time
from dataclasses import dataclass

from app.core.config import  Settings, get_settings
from app.core.exceptions import LLMError
from app.rag.llm import LLMProvider, get_llm
from app.rag.prompts.grounding import (
    GROUNDING_SYSTEM_INSTRUCTION,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    build_grounding_prompt,
)

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"         # checker lỗi hoặc trả lời sai format -> không kết luận được
VERDICTS = {SUPPORTED, PARTIAL, UNSUPPORTED}

@dataclass
class GroundingResult:
    verdict: str        # SUPPORTED | PARTIAL | UNSUPPORTED | UNKNOWN
    reason: str = ""
    ms: float = 0.0


def parse_grounding(text: str) -> GroundingResult:
    """'PARTIAL\\nThe year 2021 is not in the sources.' -> GroundingResult('PARTIAL', 'The year ...')"""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]        # chỉ  giữ lại những dòng không rỗng
    if not lines:
        return GroundingResult(UNKNOWN, "empty grounding reply")
    # lấy phần tử đầu trong line, split sau đó strip, upper
    first = lines[0].split()[0].strip("*.:,").upper()     # chịu được "**Supported**", "PARTIAL:"
    if first not in VERDICTS:
        return GroundingResult(UNKNOWN, f"unexpected grounding reply: {lines[0][:80]}")

    # reason dòng thứ 2 trở đi trong lines, nối vào
    # Trường hợp 2: VERDICTS cùng dòng với reason, tách ra để lấy reason.
    reason = " ".join(lines[1:]) or lines[0][len(lines[0].split()[0]):].strip(" :-")
    return GroundingResult(first, reason)


class GroundingChecker:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # Grounding checker, gọi LLM, build, đánh giá câu trả lời và nguồn xem câu trả lời có sát với nguồn không
    def check(self, answer: str, context: str) -> GroundingResult:
        t0 = time.perf_counter()
        try:
            response = self.llm.generate(build_grounding_prompt(answer, context), GROUNDING_SYSTEM_INSTRUCTION)
        except LLMError as exc:
            logger.warning("grounding check failed, keeping the answer: %s", exc)
            return GroundingResult(UNKNOWN, str(exc), (time.perf_counter() - t0) * 1000)

        result = parse_grounding(response.text)
        return GroundingResult(result.verdict, result.reason, (time.perf_counter() - t0) * 1000)


def get_grounding_checker(settings: Settings | None = None) -> GroundingChecker | None:
    """GROUNDING_CHECK=false -> None (RAGService bỏ qua bước này). Dùng temperature 0: chấm điểm cần ổn định."""
    s = settings or get_settings()
    if not s.grounding_check:
        return None
    judge_settings = s.model_copy(update={
        "llm_model": s.grounding_model or s.llm_model,
        "llm_temperature":  0.0
    })
    return GroundingChecker(get_llm(judge_settings))