"""LLM_PROVIDER=nvidia (NVIDIA NIM): trả lời bằng NIM, giám khảo bằng Groq. Không gọi API thật."""
import pytest

from app.core.config import Settings
from app.evaluation.faithfulness import build_faithfulness_judge
from app.rag.llm import get_llm
from app.rag.llm.openai_compat import OpenAICompatLLM

NVIDIA_URL, GROQ_URL = "https://integrate.api.nvidia.com/v1", "https://api.groq.com/openai/v1"


def _settings(**kw) -> Settings:
    base = dict(_env_file=None, llm_provider="nvidia", llm_model="nvidia/nemotron-3-ultra",
                nvidia_api_key="k-nvidia", groq_api_key="k-groq")
    return Settings(**{**base, **kw})


def _where(llm: OpenAICompatLLM) -> tuple[str, str, str]:
    return str(llm._client.base_url).rstrip("/"), llm._client.api_key, llm.model


def test_answer_uses_nvidia() -> None:
    assert _where(get_llm(_settings())) == (NVIDIA_URL, "k-nvidia", "nvidia/nemotron-3-ultra")


def test_judge_on_groq_while_answer_on_nvidia() -> None:
    s = _settings(judge_provider="groq", judge_model="openai/gpt-oss-120b")
    assert _where(build_faithfulness_judge(s).llm) == (GROQ_URL, "k-groq", "openai/gpt-oss-120b")


def test_nvidia_without_key_names_the_variable() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_llm(_settings(nvidia_api_key=""))           # dòng NVIDIA_API_KEY= để trống = chưa đặt