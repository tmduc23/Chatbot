"""
MỤC ĐÍCH: Unit test cho tầng LLM - KHÔNG gọi Gemini thật

Dùng stub client giống test reranker ở Phase 5 để kiểm tra:
    - gửi đúng model, prompt, system instruction, temperature, max_output_tokens;
    - đọc text và số token; retry khi 429; không retry khi 401
    - câu trả lời rỗng -> LLMError (không âm thầm trả chuỗi rỗng cho người dùng)
    - factory chọn đúng class, thiếu key hoặc sai tên thì báo lỗi rõ.
"""

from types import SimpleNamespace

import pytest

from app.core.config import get_settings, Settings   # get_settings không dùng tới, xóa đi cũng được
from app.core.exceptions import LLMError
from app.rag.llm import get_llm
from app.rag.llm.fake import FakeLLM
from app.rag.llm.gemini import GeminiLLM


# =====================================================================
# PHẦN 1: ĐỒ GIẢ (stub) - thay cho Gemini thật để test không cần mạng,
#          không tốn quota, chạy nhanh và luôn ra cùng kết quả
# =====================================================================

# MỤC ĐÍCH: Giả lập `client.models` của SDK google-genai.
#   - `responses`: danh sách "kịch bản" trả về theo thứ tự cho từng lần gọi.
#       Phần tử là response -> trả về; là Exception -> raise (giả lập lỗi API).
#   - `calls`: ghi lại MỌI lần gọi (model, prompt, config) để test kiểm tra
#       GeminiLLM đã gửi đúng tham số chưa, và đã gọi bao nhiêu lần (đếm retry).
class _StubModels:
    def __init__(self, responses: list):
        self.responses = list(responses)   # copy để pop() không làm hỏng list gốc
        self.calls: list[dict] = []

    # Cùng tên và tham số với hàm thật `client.models.generate_content(...)`
    def generate_content(self, model: str, contents: str, config: object) -> SimpleNamespace:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.responses.pop(0)        # lấy kịch bản tiếp theo
        if isinstance(item, Exception):
            raise item
        return item


# MỤC ĐÍCH: Tạo nhanh một response giả có hình dạng giống response thật của Gemini:
#   .text, .usage_metadata (số token vào/ra), .candidates[0].finish_reason.
#   text=None dùng để giả lập "Gemini trả về nhưng không có chữ nào".
def _resp(text: str | None, tokens_in: int = 120, tokens_out: int = 30) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_token_count=tokens_in, candidates_token_count=tokens_out)
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[SimpleNamespace(finish_reason="STOP")])


# MỤC ĐÍCH: Tạo một lỗi HTTP giả có thuộc tính .code (429, 401...),
#   giống lỗi thật của google-genai, để GeminiLLM._call() đọc `exc.code`
#   và quyết định retry hay không.
def _http_error(code: int) -> Exception:
    err = RuntimeError(f"HTTP {code}")
    err.code = code                        # lỗi thật của google-genai có thuộc tính .code
    return err


# MỤC ĐÍCH: Lắp GeminiLLM với client giả trong 1 dòng.
#   Trả về cả `llm` (để gọi generate) và `stub` (để kiểm tra calls).
#   `client=SimpleNamespace(models=stub)` -> GeminiLLM gọi self._client.models.generate_content
#   sẽ chạy vào stub. Đây là dependency injection: không truyền client thì GeminiLLM tạo client thật.
#   `**kwargs` để test truyền thêm temperature, max_output_tokens.
def _llm(responses: list, **kwargs) -> tuple[GeminiLLM, _StubModels]:
    stub = _StubModels(responses)
    return GeminiLLM(api_key="test", model="gemini-test", client=SimpleNamespace(models=stub), **kwargs), stub


# =====================================================================
# PHẦN 2: TEST GeminiLLM
# =====================================================================

# MỤC ĐÍCH: Kiểm tra GeminiLLM GỬI ĐÚNG và ĐỌC ĐÚNG.
#   Gửi:  model, prompt, system_instruction, temperature, max_output_tokens đúng như cấu hình.
#   Đọc:  text câu trả lời và số token vào/ra (120, 30) từ response.
#   Bắt lỗi như: quên truyền system_instruction vào config, đọc nhầm tên trường token,
#   hard-code temperature thay vì dùng giá trị truyền vào.
def test_gemini_sends_prompt_and_config() -> None:
    llm, stub = _llm([_resp("Có [Source 1].")], temperature=0.1, max_output_tokens=500)
    result = llm.generate("the prompt", system_instruction="the rules")

    assert result.text == "Có [Source 1]."
    assert (result.input_tokens, result.output_tokens) == (120, 30)
    call = stub.calls[0]
    assert call["model"] == "gemini-test" and call["contents"] == "the prompt"
    assert call["config"].system_instruction == "the rules"
    assert call["config"].temperature == 0.1
    assert call["config"].max_output_tokens == 500


# MỤC ĐÍCH: Lỗi 429 (quá giới hạn request) là lỗi TẠM THỜI -> phải THỬ LẠI.
#   Kịch bản: lần 1 lỗi 429, lần 2 thành công -> generate vẫn trả "ok", và stub bị gọi đúng 2 lần.
#   monkeypatch thay time.sleep bằng hàm không làm gì -> test không phải chờ 2 giây thật.
def test_gemini_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.rag.llm.gemini.time.sleep", lambda s: None)
    llm, stub = _llm([_http_error(429), _resp("ok")])
    assert llm.generate("p").text == "ok"
    assert len(stub.calls) == 2


# MỤC ĐÍCH: Lỗi 401 (sai API key) là lỗi VĨNH VIỄN -> thử lại vô ích, phải raise LLMError NGAY.
#   `len(stub.calls) == 1` chứng minh không retry. Nếu code lỡ retry, stub sẽ bị gọi lần 2
#   (calls == 2) nên test fail, dù lần 2 đó lỗi IndexError vì hết kịch bản.
def test_gemini_does_not_retry_on_401() -> None:
    llm, stub = _llm([_http_error(401)])
    with pytest.raises(LLMError):
        llm.generate("p")
    assert len(stub.calls) == 1


# MỤC ĐÍCH: Gemini trả 200 OK nhưng text rỗng (bị lọc SAFETY, hết MAX_TOKENS...)
#   -> phải raise LLMError có chữ "empty", KHÔNG được âm thầm trả "" cho người dùng.
#   `match="empty"` kiểm tra nội dung thông báo lỗi (regex tìm trong message).
def test_gemini_empty_answer_raises() -> None:
    llm, _ = _llm([_resp(None)])
    with pytest.raises(LLMError, match="empty"):
        llm.generate("p")


# =====================================================================
# PHẦN 3: TEST FakeLLM (dùng trong test RAGService ở các file khác)
# =====================================================================

# MỤC ĐÍCH: FakeLLM trả đúng câu đã cài sẵn ("hello") và GHI LẠI prompt + system_instruction.
#   Test RAGService sau này dựa vào `llm.calls` để kiểm tra:
#   "lớp 1 từ chối thì LLM không được gọi" (calls == []), hoặc prompt có chứa context không.
#   Nên phải chắc FakeLLM ghi lại đúng trước.
def test_fake_llm_records_calls() -> None:
    llm = FakeLLM("hello")
    assert llm.generate("p", "s").text == "hello"
    assert llm.calls == [{"prompt": "p", "system_instruction": "s"}]


# =====================================================================
# PHẦN 4: TEST FACTORY get_llm (đọc LLM_PROVIDER trong .env để chọn class)
# =====================================================================

# MỤC ĐÍCH: LLM_PROVIDER=fake -> get_llm trả về FakeLLM.
#   Truyền Settings(...) trực tiếp nên test không phụ thuộc file .env trên máy.
def test_factory_fake() -> None:
    assert isinstance(get_llm(Settings(llm_provider="fake")), FakeLLM)


# MỤC ĐÍCH: LLM_PROVIDER=gemini mà thiếu GEMINI_API_KEY -> báo ValueError có nhắc "GEMINI_API_KEY"
#   NGAY khi khởi tạo, thay vì để tới lúc người dùng hỏi mới lỗi 401 khó hiểu.
def test_factory_gemini_requires_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_llm(Settings(llm_provider="gemini", gemini_api_key=None))


# MỤC ĐÍCH: Gõ sai tên provider trong .env (ví dụ "magic", "gemnii")
#   -> báo lỗi rõ "Unknown LLM_PROVIDER", không âm thầm chọn mặc định.
def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm(Settings(llm_provider="magic"))
