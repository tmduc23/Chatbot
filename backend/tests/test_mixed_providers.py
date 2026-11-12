"""Trả lời bằng Groq, giám khảo / grounding bằng OpenRouter. Không gọi API thật."""
import pytest

from app.core.config import Settings
from app.evaluation.faithfulness import build_faithfulness_judge, judge_provider_name
from app.rag.grounding import get_grounding_checker
from app.rag.llm import get_llm
from app.rag.llm.openai_compat import OpenAICompatLLM

GROQ_URL, OPENROUTER_URL = "https://api.groq.com/openai/v1", "https://openrouter.ai/api/v1"


def _settings(**kw) -> Settings:
    base = dict(_env_file=None, llm_provider="groq", llm_model="openai/gpt-oss-120b", groq_api_key="k-groq",
                openrouter_judge_api_key="k-judge", openrouter_grounding_api_key="k-ground")
    return Settings(**{**base, **kw})


def _where(llm: OpenAICompatLLM) -> tuple[str, str, str]:
    return str(llm._client.base_url).rstrip("/"), llm._client.api_key, llm.model


def test_answer_uses_groq() -> None:
    assert _where(get_llm(_settings())) == (GROQ_URL, "k-groq", "openai/gpt-oss-120b")


def test_judge_can_use_another_provider() -> None:
    s = _settings(judge_provider="openrouter", judge_model="nvidia/nemotron-3-ultra-550b-a55b:free")
    assert judge_provider_name(s) == "openrouter"
    judge = build_faithfulness_judge(s).llm
    assert _where(judge) == (OPENROUTER_URL, "k-judge", "nvidia/nemotron-3-ultra-550b-a55b:free")
    assert s.llm_provider == "groq"                                      # cấu hình của LLM trả lời không đổi
    assert judge_provider_name(_settings(judge_provider="")) == "groq"   # để trống = dùng LLM_PROVIDER


def test_grounding_can_use_another_provider() -> None:
    s = _settings(grounding_check=True, grounding_provider="openrouter",
                  grounding_model="nvidia/nemotron-3-super-120b-a12b:free")
    assert _where(get_grounding_checker(s).llm) == (OPENROUTER_URL, "k-ground", "nvidia/nemotron-3-super-120b-a12b:free")


def test_groq_without_key_names_the_variable() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm(_settings(groq_api_key=""))