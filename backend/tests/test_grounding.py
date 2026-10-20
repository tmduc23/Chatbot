"""
MỤC ĐÍCH: Test grounding checker (Phase 7) bằng LLM giả — không gọi Gemini.
    - đọc đúng 3 mức SUPPORTED / PARTIAL / UNSUPPORTED, chịu được format hơi lệch;
    - trả lời lạ hoặc LLM lỗi -> UNKNOWN (fail-open, không chặn câu trả lời);
    - factory: tắt -> None; bật -> dùng GROUNDING_MODEL và temperature 0.
"""

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.rag.grounding import UNKNOWN, GroundingChecker, get_grounding_checker, parse_grounding
from app.rag.llm import LLMResponse, LLMProvider
from app.rag.llm.fake import FakeLLM


@pytest.mark.parametrize(("reply", "verdict", "reason"), [
    ("SUPPORTED\nOK", "SUPPORTED", "OK"),
    ("**Partial**\nThe year 2021 is not in the sources.", "PARTIAL", "The year 2021 is not in the sources."),
    ("UNSUPPORTED: no source mentions Kubernetes", "UNSUPPORTED", "no source mentions Kubernetes"),
    ("", UNKNOWN, "empty grounding reply"),
    ("I think the answer looks fine", UNKNOWN, "unexpected grounding reply: I think the answer looks fine"),
])
def test_parse_grounding(reply: str, verdict: str, reason: str) -> None:
    result = parse_grounding(reply)
    assert (result.verdict, result.reason) == (verdict, reason)

def test_checker_sends_answer_and_sources() -> None:
    judge = FakeLLM("SUPPORTED\nOK")
    result = GroundingChecker(judge).check("Dùng Docker [Source 1].", "[Source 1]\nDocument: cv.pdf\n\nDocker")
    assert result.verdict == "SUPPORTED"
    prompt = judge.calls[0]["prompt"]
    assert "SOURCES:\n[Source 1]\nDocument: cv.pdf" in prompt
    assert "ANSWER:\nDùng Docker [Source 1]." in prompt
    assert "SUPPORTED, PARTIAL or UNSUPPORTED" in judge.calls[0]["system_instruction"]

class _FailingLLM(LLMProvider):
    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        raise LLMError("429 RESOURCE_EXHAUSTED")

    
def test_checker_fails_open_on_llm_error() -> None:
    result = GroundingChecker(_FailingLLM()).check("answer", "context")
    assert result.verdict == UNKNOWN
    assert "429" in result.reason

# LLM Grounding mà không gọi được sẽ trả về uknown
def test_factory_returns_none_when_disabled() -> None:
    assert get_grounding_checker(Settings(llm_provider="fake", grounding_check=False)) is None

# test sử dụng theo cấu hình llm_grounding
def test_factory_uses_grounding_model_and_zero_temperature() -> None:
    s = Settings(llm_provider="gemini", gemini_api_key="test", llm_model="answer-model",
                 llm_temperature=0.7, grounding_check=True, grounding_model="judge-model")
    checker = get_grounding_checker(s)
    assert checker.llm.model == "judge-model"
    assert checker.llm.temperature == 0.0