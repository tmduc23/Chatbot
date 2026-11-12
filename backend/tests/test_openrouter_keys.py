"""3 key OpenRouter theo vai trò (answer | grounding | judge). Không gọi API thật."""
import pytest

from app.core.config import Settings
from app.evaluation.faithfulness import build_faithfulness_judge
from app.rag.grounding import get_grounding_checker
from app.rag.llm import get_llm


def _settings(**kw) -> Settings:
    base = dict(_env_file=None, llm_provider="openrouter", llm_model="qwen/qwen3.8-27b:free")
    return Settings(**{**base, **kw})


def _key(llm) -> str:
    return llm._client.api_key   # client OpenAI thật, chưa gửi request nào


def test_each_role_uses_its_own_key() -> None:
    s = _settings(openrouter_answer_api_key="k-answer", openrouter_grounding_api_key="k-ground",
                  openrouter_judge_api_key="k-judge", grounding_check=True, grounding_model="google/gemma-4-31b-it:free")
    assert _key(get_llm(s)) == "k-answer"
    assert _key(get_grounding_checker(s).llm) == "k-ground"
    assert _key(build_faithfulness_judge(s, model="nvidia/nemotron-3-ultra-550b-a55b:free").llm) == "k-judge"


def test_missing_role_key_falls_back_to_shared_key() -> None:
    s = _settings(openrouter_api_key="k-shared", openrouter_judge_api_key="k-judge")
    assert _key(get_llm(s)) == "k-shared"
    assert _key(get_llm(s, role="grounding")) == "k-shared"
    assert _key(get_llm(s, role="judge")) == "k-judge"


def test_empty_values_in_env_mean_not_set() -> None:
    s = _settings(openrouter_api_key="k-shared", openrouter_answer_api_key="", openrouter_judge_api_key="  ")
    assert s.openrouter_answer_api_key is None
    assert s.openrouter_key("answer") == "k-shared" and s.openrouter_key("judge") == "k-shared"


def test_no_key_at_all_names_the_variables() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_JUDGE_API_KEY"):
        get_llm(_settings(), role="judge")