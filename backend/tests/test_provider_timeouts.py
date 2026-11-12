"""MỤC ĐÍCH: Phase 12a — mọi client gọi API bên ngoài đều được tạo với TIMEOUT lấy từ cấu hình."""
import pytest

from app.core.config import Settings
from app.rag.clients import make_cohere_client, make_gemini_client
from app.rag.embeddings import get_embedding_provider
from app.rag.llm.factory import get_llm


class _Recorder:
    """Thay class client của SDK: chỉ ghi lại tham số được truyền vào, không gọi mạng."""
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        _Recorder.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _clear() -> None:
    _Recorder.calls = []


def test_gemini_client_timeout_is_in_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai
    monkeypatch.setattr(genai, "Client", _Recorder)
    make_gemini_client("key", timeout_s=12.5)
    assert _Recorder.calls[-1]["http_options"].timeout == 12500


def test_cohere_client_has_timeout_and_no_sdk_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import cohere
    monkeypatch.setattr(cohere, "ClientV2", _Recorder)
    make_cohere_client("key", timeout_s=7)
    assert _Recorder.calls[-1]["timeout"] == 7 and _Recorder.calls[-1]["max_retries"] == 0


def test_embedding_factory_uses_provider_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import cohere
    monkeypatch.setattr(cohere, "ClientV2", _Recorder)
    s = Settings(_env_file=None, embedding_provider="cohere", embedding_model="embed-v5.0-pro",
                 cohere_api_key="k", provider_timeout_seconds=9)
    get_embedding_provider(s)
    assert _Recorder.calls[-1]["timeout"] == 9


def test_llm_factory_uses_llm_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    monkeypatch.setattr(openai, "OpenAI", _Recorder)
    s = Settings(_env_file=None, llm_provider="openrouter", llm_model="x", openrouter_api_key="k",
                 llm_timeout_seconds=42)
    get_llm(s)
    assert _Recorder.calls[-1]["timeout"] == 42 and _Recorder.calls[-1]["max_retries"] == 0


@pytest.mark.parametrize("provider, key_field", [("groq", "groq_api_key"), ("nvidia", "nvidia_api_key")])
def test_groq_and_nvidia_also_use_llm_timeout(monkeypatch: pytest.MonkeyPatch, provider: str, key_field: str) -> None:
    # Trước 12a: 180s viết cứng trong OpenAICompatLLM. Giờ đổi được bằng LLM_TIMEOUT_SECONDS.
    import openai
    monkeypatch.setattr(openai, "OpenAI", _Recorder)
    s = Settings(_env_file=None, llm_provider=provider, llm_model="x", llm_timeout_seconds=33, **{key_field: "k"})
    get_llm(s)
    assert _Recorder.calls[-1]["timeout"] == 33 and _Recorder.calls[-1]["max_retries"] == 0
