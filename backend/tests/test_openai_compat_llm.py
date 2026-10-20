"""
MỤC ĐÍCH: Unit test cho OpenAICompatLLM (OpenRouter/Groq...) — KHÔNG gọi API thật.

Stub client giả lập client.chat.completions.create(...), giống cách test GeminiLLM:
  - gửi đúng model, messages (system + user), temperature, max_tokens;
  - đọc text và số token; retry khi 429; KHÔNG retry khi 401;
  - câu trả lời rỗng -> LLMError;
  - factory chọn đúng class, thiếu key thì báo lỗi rõ.
"""
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.rag.llm import get_llm
from app.rag.llm import openai_compat
from app.rag.llm.openai_compat import OpenAICompatLLM


class _StubCompletions:
    def __init__(self, responses: list):
        self.responses = list(responses)   # mỗi phần tử: SimpleNamespace (response) hoặc Exception
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(text: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def _http_error(code: int) -> Exception:
    err = RuntimeError(f"HTTP {code}")
    err.status_code = code                 # lỗi thật của thư viện openai có thuộc tính .status_code
    return err


def _llm(responses: list, **kwargs) -> tuple[OpenAICompatLLM, _StubCompletions]:
    stub = _StubCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    llm = OpenAICompatLLM(api_key="test", model="test/model:free", base_url="https://example.test/v1",
                          client=client, **kwargs)
    return llm, stub


def test_sends_messages_and_config() -> None:
    llm, stub = _llm([_resp("Có [Source 1].")], temperature=0.1, max_output_tokens=500)
    result = llm.generate("the prompt", system_instruction="the rules")

    assert result.text == "Có [Source 1]."
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    call = stub.calls[0]
    assert call["model"] == "test/model:free"
    assert call["messages"] == [{"role": "system", "content": "the rules"},
                                {"role": "user", "content": "the prompt"}]
    assert (call["temperature"], call["max_tokens"]) == (0.1, 500)


def test_no_system_instruction_sends_only_user_message() -> None:
    llm, stub = _llm([_resp("ok")])
    llm.generate("hi")
    assert stub.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_compat.time, "sleep", lambda s: None)   # không chờ thật
    llm, stub = _llm([_http_error(429), _resp("ok")])
    assert llm.generate("q").text == "ok"
    assert len(stub.calls) == 2


def test_no_retry_on_401() -> None:
    llm, stub = _llm([_http_error(401)])
    with pytest.raises(LLMError):
        llm.generate("q")
    assert len(stub.calls) == 1


def test_empty_answer_raises() -> None:
    llm, _ = _llm([_resp(None, finish_reason="length")])
    with pytest.raises(LLMError, match="length"):
        llm.generate("q")


def test_factory_openrouter() -> None:
    s = Settings(llm_provider="openrouter", openrouter_api_key="k",
                 llm_model="nvidia/nemotron-3-ultra-550b-a55b:free")
    llm = get_llm(s)
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.model == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_factory_openrouter_requires_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        get_llm(Settings(llm_provider="openrouter", openrouter_api_key=None))

def test_retries_on_error_body_with_http_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_compat.time, "sleep", lambda s: None)
    error_body = SimpleNamespace(choices=None, error={"message": "Provider returned error", "code": 429})
    llm, stub = _llm([error_body, _resp("ok")])
    assert llm.generate("q").text == "ok"
    assert len(stub.calls) == 2


def test_error_body_not_retryable_raises() -> None:
    error_body = SimpleNamespace(choices=None, error={"message": "bad request", "code": 400})
    llm, stub = _llm([error_body])
    with pytest.raises(LLMError, match="bad request"):
        llm.generate("q")
    assert len(stub.calls) == 1