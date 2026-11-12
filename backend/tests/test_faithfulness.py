"""MỤC ĐÍCH: Test giám khảo faithfulness kiểu RAGAS — LLM giả, không gọi API thật."""
from app.core.config import Settings
from app.core.exceptions import LLMError
from app.evaluation.faithfulness import (
    FaithfulnessJudge, build_faithfulness_judge, build_verify_prompt, parse_claims, parse_verdicts, strip_citations,
)
from app.rag.llm.base import LLMProvider, LLMResponse
from app.evaluation.faithfulness import (
    FaithfulnessJudge, build_faithfulness_judge, build_verify_prompt, judge_model_name, parse_claims, parse_verdicts,
    strip_citations,
)

CONTEXT = "[Source 1] Học vấn: Đại học Bách khoa Hà Nội, ngành Điều khiển và Tự động hóa."


class ScriptedLLM(LLMProvider):
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_instruction))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, model="judge")


def test_strip_citations() -> None:
    assert strip_citations("Đức học Bách khoa [Source 1] và [Sources 2, 3].") == "Đức học Bách khoa và ."
    assert strip_citations("no marker") == "no marker"


def test_parse_claims_accepts_common_list_styles() -> None:
    assert parse_claims("- A\n* B\n1. C\n2) D\nrác không phải ý") == ["A", "B", "C", "D"]
    assert parse_claims("NONE") == [] and parse_claims("none.") == []
    assert parse_claims("\n".join(f"- ý {i}" for i in range(20)), max_claims=3) == ["ý 0", "ý 1", "ý 2"]


def test_parse_verdicts_tolerates_format_and_marks_missing() -> None:
    assert parse_verdicts("1: YES\n2: no\n**3.** Yes", 3) == [True, False, True]
    assert parse_verdicts("1: YES\n5: NO", 3) == [True, None, None]   # thiếu 2, 3; số 5 không tồn tại
    assert parse_verdicts("I think they are fine", 2) == [None, None]


def test_score_is_supported_claims_over_judged_claims() -> None:
    llm = ScriptedLLM("- Đức học Bách khoa Hà Nội\n- Đức học ngành Tự động hóa\n- Đức tốt nghiệp loại giỏi",
                      "1: YES\n2: YES\n3: NO")
    result = FaithfulnessJudge(llm).score("Đức học Bách khoa [Source 1], ngành Tự động hóa, loại giỏi.", CONTEXT)
    assert result.score == 2 / 3
    assert result.supported == [True, True, False]
    assert result.llm_calls == 2 and result.error is None
    assert "[Source" not in llm.calls[0][0]                  # dấu trích dẫn không bị tách thành ý
    assert CONTEXT in llm.calls[1][0] and "3. Đức tốt nghiệp loại giỏi" in llm.calls[1][0]
    assert result.to_dict()["claims"][2] == {"text": "Đức tốt nghiệp loại giỏi", "supported": False}


def test_missing_verdicts_are_left_out_of_the_score() -> None:
    result = FaithfulnessJudge(ScriptedLLM("- A\n- B", "1: YES")).score("A, B", CONTEXT)
    assert result.score == 1.0 and result.supported == [True, None]


def test_no_claims_means_no_score_and_one_call() -> None:
    llm = ScriptedLLM("NONE")
    result = FaithfulnessJudge(llm).score("Xin chào!", CONTEXT)
    assert result.score is None and result.error == "no_claims" and len(llm.calls) == 1


def test_llm_error_gives_no_score_instead_of_zero() -> None:
    result = FaithfulnessJudge(ScriptedLLM("- A", LLMError("HTTP 429"))).score("A", CONTEXT)
    assert result.score is None and "429" in result.error and result.llm_calls == 2
    unparseable = FaithfulnessJudge(ScriptedLLM("- A", "tất cả đều đúng")).score("A", CONTEXT)
    assert unparseable.score is None and unparseable.error.startswith("unparseable")


def test_verify_prompt_numbers_claims() -> None:
    assert build_verify_prompt("ctx", ["x", "y"]) == "CONTEXT:\nctx\n\nCLAIMS:\n1. x\n2. y"


def test_judge_uses_temperature_zero_and_chosen_model(monkeypatch) -> None:
    settings = Settings(llm_provider="fake", llm_model="answer-model", llm_temperature=0.7, grounding_model="g-model")
    seen = {}

    def fake_get_llm(s: Settings, role: str = "answer") -> LLMProvider:
        seen.update(model=s.llm_model, temperature=s.llm_temperature, role=role)
        return ScriptedLLM()

    monkeypatch.setattr("app.evaluation.faithfulness.get_llm", fake_get_llm)
    build_faithfulness_judge(settings)
    assert seen == {"model": "g-model", "temperature": 0.0, "role": "judge"}   # mặc định: GROUNDING_MODEL; key giám khảo
    build_faithfulness_judge(settings, model="judge-model")
    assert seen["model"] == "judge-model"                          # --judge-model thắng
    assert settings.llm_temperature == 0.7                         # không sửa cấu hình của LLM trả lời

    monkeypatch.setattr("app.evaluation.faithfulness.get_llm", fake_get_llm)
    build_faithfulness_judge(settings)
    assert seen == {"model": "g-model", "temperature": 0.0, "role": "judge"}   # mặc định: GROUNDING_MODEL; key giám khảo
    build_faithfulness_judge(settings, model="judge-model")
    assert seen["model"] == "judge-model"                          # --judge-model thắng
    assert settings.llm_temperature == 0.7                         # không sửa cấu hình của LLM trả lời

def test_judge_model_setting_beats_grounding_model() -> None:
    s = Settings(_env_file=None, llm_model="answer-model", grounding_model="g-model", judge_model="j-model")
    assert judge_model_name(s) == "j-model"                        # JUDGE_MODEL thắng GROUNDING_MODEL
    assert judge_model_name(s, "cli-model") == "cli-model"         # --judge-model thắng tất cả
    assert judge_model_name(s.model_copy(update={"judge_model": None})) == "g-model"
    assert judge_model_name(Settings(_env_file=None, llm_model="answer-model", judge_model="")) == "answer-model"


def test_strip_citations_handles_fullwidth_brackets() -> None:
    assert strip_citations("Time 【Source 1】 and space [Source 2].") == "Time and space ."