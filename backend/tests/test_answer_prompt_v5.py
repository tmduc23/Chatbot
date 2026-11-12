"""Prompt trả lời v5: giống v4 nhưng nới luật 7 để bớt từ chối nhầm. Không gọi API thật."""
from app.core.config import Settings
from app.evaluation.report import config_snapshot
from app.rag.llm.fake import FakeLLM
from app.rag.prompts.answer import INSUFFICIENT_MARKER, build_answer_prompt, build_answer_system_instruction
from tests.test_rag_service import _service


def test_v5_relaxes_rule_7_but_keeps_rule_8() -> None:
    v4, v5 = build_answer_system_instruction(version="v4"), build_answer_system_instruction(version="v5")
    assert "EXACTLY" in v4 and "EXACTLY" not in v5                  # bỏ đòi hòi khớp từng chi tiết
    assert "in different words" in v5 and "table" in v5             # diễn đạt khác / số trong bảng vẫn tính
    assert f"exactly {INSUFFICIENT_MARKER} only when no source" in v5
    assert "8. Every sentence" in v5                                # luật trích dẫn giữ nguyên


def test_v5_reminder_after_question_and_language_line_last() -> None:
    p4, p5 = build_answer_prompt("Who?", "ctx", "v4"), build_answer_prompt("Who", "ctx", "v5")
    assert "even in other words" in p5 and "even in other words" not in p4
    assert INSUFFICIENT_MARKER in p5
    assert p5.index("Question:") < p5.index("even in other words")      # nhắc NGAY SAU câu hỏi
    assert p5.endswith("Answer in English.")

def test_service_sends_v5_and_eval_config_records_it() -> None:
    llm = FakeLLM("Docker [Source 1].")
    service = _service(llm)
    service.answer_prompt = "v5"
    service.ask("Docker Kubernetes", "cv")
    assert "in different words" in llm.calls[0]["system_instruction"]
    assert config_snapshot(Settings(_env_file=None, answer_prompt="v5"))["answer_prompt"] == "v5"