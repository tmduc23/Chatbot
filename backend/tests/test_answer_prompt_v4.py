"""Prompt trả lời v3 / v4 chọn bằng ANSWER_PROMPT (A/B câu bẫy). Không gọi API thật."""
import pytest

from app.core.config import Settings
from app.evaluation.report import config_snapshot
from app.rag.llm.fake import FakeLLM
from app.rag.prompts.answer import INSUFFICIENT_MARKER, build_answer_prompt, build_answer_system_instruction
from app.services.rag_service import RAGService
from tests.test_rag_service import _service


def test_v4_adds_exact_detail_rule_and_reminder() -> None:
    v3, v4 = build_answer_system_instruction(version="v3"), build_answer_system_instruction(version="v4")
    assert "7. Before answering" not in v3 and "7. Before answering" in v4
    assert "8. Every sentence" in v4
    p3, p4 = build_answer_prompt("Who?", "ctx", "v3"), build_answer_prompt("Who?", "ctx", "v4")
    assert INSUFFICIENT_MARKER not in p3 and INSUFFICIENT_MARKER in p4
    assert p4.endswith("Answer in English.")          # dòng ngôn ngữ (v3) vẫn ở cuối cùng


def test_unknown_version_fails_fast() -> None:
    with pytest.raises(ValueError, match="ANSWER_PROMPT"):
        build_answer_system_instruction(version="v6")
    with pytest.raises(ValueError, match="ANSWER_PROMPT"):
        RAGService(pipeline=None, llm=FakeLLM(), answer_prompt="V4 ")


def test_service_sends_the_chosen_version() -> None:
    llm = FakeLLM("Docker [Source 1].")
    service = _service(llm)
    service.answer_prompt = "v3"
    service.ask("Docker Kubernetes", "cv")
    assert "7. Before answering" not in llm.calls[0]["system_instruction"]
    service.answer_prompt = "v4"
    service.ask("Docker Kubernetes", "cv")
    assert "7. Before answering" in llm.calls[1]["system_instruction"]


def test_version_is_saved_in_eval_config() -> None:
    assert config_snapshot(Settings(_env_file=None, answer_prompt="v3"))["answer_prompt"] == "v3"