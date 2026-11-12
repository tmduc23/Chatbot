"""
MỤC ĐÍCH: Phase 9 — test REST API bằng TestClient + database THẬT + embedding/reranker/LLM GIẢ.

TestClient gọi thẳng vào app trong cùng process (không cần chạy uvicorn).
app.dependency_overrides thay provider thật bằng bản giả -> không tốn 1 request Gemini/OpenRouter nào,
nhưng mọi thứ còn lại là THẬT: route, schema, service, repository, Postgres, pgvector.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path


import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import get_chat_service, get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.db.session import SessionLocal
from app.main import app
from app.models import EMBEDDING_DIM, Application, Chunk, ChatSession, Message, RetrievalLog
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.base import LLMProvider, LLMResponse
from app.rag.prompts.condense import CONDENSE_SYSTEM_INSTRUCTION, build_condense_prompt
from app.rag.question_condenser import QuestionCondenser
from app.rag.reranking.fake import FakeReranker


API = "/api/v1"

CV_TEXT = "Duc built a RAG chatbot project using FastAPI and pgvector."
JOB_TEXT = "Duc worked at Acme Corp as a backend engineer using Docker and Postgres."


class ScriptedLLM(LLMProvider):
    """LLM giả biết mình được gọi để VIẾT LẠI câu hỏi hay để TRẢ LỜI (nhìn system instruction)."""

    def __init__(self, answer: str = "Duc built a RAG chatbot [Source 1].", condensed: str | None = None,
                 fail_answer: bool = False, fail_condense: bool = False):
        self.answer, self.condensed = answer, condensed
        self.fail_answer, self.fail_condense = fail_answer, fail_condense
        self.answer_prompts: list[str] = []
        self.condensed_prompts: list[str] = []


    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        if system_instruction == CONDENSE_SYSTEM_INSTRUCTION:
            self.condensed_prompts.append(prompt)
            if self.fail_condense:
                raise LLMError("429 rate limited")
            return LLMResponse(text=self.condensed or "", model="fake")
        self.answer_prompts.append(prompt)
        if self.fail_answer:
            raise LLMError("429 rate limited (secret provider detail)")
        return LLMResponse(text=self.answer, model="fake")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="development", admin_api_key=None, storage_dir=str(tmp_path / "storage"), max_upload_mb=1,
        chunk_size=500, chunk_overlap=50, tokenizer_encoding="whitespace",
        embedding_provider="fake", reranker_provider="fake", reranker_min_score=0.3,
        llm_provider="fake", grounding_check=False, chat_condense_question=True, chat_history_turns=3,
        background_ingestion=False,
        auth_enabled=False,           # Phase 12b: file này test chế độ Phase 9 (X-API-Key); đăng nhập ở test_auth_api.py
        rate_limit_enabled=False, answer_cache_enabled=False,   # Phase 12c: test riêng ở test_rate_limit / test_answer_cache
    )

@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def client(settings: Settings, llm: ScriptedLLM) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: llm
    # raise_server_exceptions=False: lỗi 500 trả về response như server thật, thay vì làm test crash
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def app_ids(db: Session) -> Iterator[list[str]]:
    """Danh sách id application test tạo ra -> xóa hết sau test (CASCADE xóa documents, chunks, chat...)"""
    ids: list[str] = []
    yield ids
    db.rollback()
    db.execute(delete(Application).where(Application.id.in_(ids)))
    db.commit()


def _new_app(client: TestClient, app_ids: list[str], prefix: str = "t-api") -> str:
    app_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    app_ids.append(app_id)
    r = client.post(f"{API}/applications", json={"id": app_id, "name": "Test app"})
    assert r.status_code == 201, r.text
    return app_id


def _upload(client: TestClient, app_id: str, filename: str, content: bytes):
    return client.post(f"{API}/documents/upload", data={"application_id": app_id},
                       files={"file": (filename, content, "text/plain")})


@pytest.fixture
def cv_app(client: TestClient, app_ids: list[str]) -> str:
    app_id = _new_app(client, app_ids)
    assert _upload(client, app_id, "cv.txt", CV_TEXT.encode()).status_code == 201
    assert _upload(client, app_id, "job.txt", JOB_TEXT.encode()).status_code == 201
    return app_id


# ======================= applications =======================

def test_create_get_and_list_application(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    r = client.get(f"{API}/applications/{app_id}")
    assert r.status_code == 200
    assert r.json()["name"] == "Test app"
    assert app_id in[a["id"] for a in client.get(f"{API}/applications").json()]


def test_create_duplicate_application_returns_409(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    r = client.post(f"{API}/applications", json={"id": app_id, "name": "Again"})
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]



@pytest.mark.parametrize("bad_id", ["CV", "cv đức", "-cv", "a", "x" * 51, "../cv"])
def test_invalid_application_id_returns_422(client: TestClient, bad_id: str) -> None:
    assert client.post(f"{API}/applications", json={"id": bad_id, "name":"x"}).status_code == 422


def test_unknown_application_returns_404_with_detail(client: TestClient) -> None:
    r = client.get(f"{API}/applications/does-not-exist")
    assert r.status_code == 404
    assert r.json() == {"detail": "Application 'does-not-exist' not found"}


# ======================= documents =======================

def test_upload_list_and_get_document(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    r = _upload(client, app_id, "cv.txt", CV_TEXT.encode())
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "READY" and doc["filename"] == "cv.txt" and doc["application_id"] == app_id
    assert "storage_path" not in doc and "file_hash" not in doc         # không lộ thông tin server

    listed = client.get(f"{API}/documents", params={"application_id": app_id}).json()
    assert [d["id"] for d in listed] == [doc["id"]]
    detail = client.get(f"{API}/documents/{doc['id']}", params={"application_id": app_id}).json()
    assert detail["chunk_count"] >= 1


def test_upload_duplicate_returns_409(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    assert _upload(client, app_id, "cv.txt", CV_TEXT.encode()).status_code == 201
    assert _upload(client, app_id, "cv-copy.txt", CV_TEXT.encode()).status_code == 409


def test_upload_unsupported_type_returns_415(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    assert _upload(client, app_id, "virus.exe", b"MZ...").status_code == 415

def test_upload_to_unknown_application_returns_404(client: TestClient) -> None:
    assert _upload(client, "no-such-app", "cv.txt", CV_TEXT.encode()).status_code == 404


def test_upload_too_large_returns_413(client: TestClient, app_ids: list[str], settings: Settings) -> None:
    app_id = _new_app(client, app_ids)
    big = b"word " * (settings.max_upload_mb * 1024 * 1024 // 5 + 10)   # > 1MB
    assert _upload(client, app_id, "big.txt", big).status_code == 413
    assert client.get(f"{API}/documents", params={"application_id": app_id}).json() == []


def test_upload_strips_path_from_filename(client: TestClient, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    r = _upload(client, app_id, "../../etc/evil.txt", CV_TEXT.encode())
    assert r.status_code == 201
    assert r.json()["filename"] == "evil.txt"


def test_document_of_other_application_is_404(client: TestClient, app_ids: list[str], cv_app: str) -> None:
    other = _new_app(client, app_ids)
    doc_id = client.get(f"{API}/documents", params={"application_id": cv_app}).json()[0]["id"]
    assert client.get(f"{API}/documents/{doc_id}", params={"application_id": other}).status_code == 404
    assert client.delete(f"{API}/documents/{doc_id}", params={"application_id": other}).status_code == 404
    # Vẫn còn nguyên ở app của nó
    assert client.get(f"{API}/documents/{doc_id}", params={"application_id": cv_app}).status_code == 200

def test_delete_document_removes_row_chunks_and_file(
    client: TestClient, app_ids: list[str], db: Session, settings: Settings,
) -> None:
    app_id = _new_app(client, app_ids)
    doc_id = _upload(client, app_id, "cv.txt", CV_TEXT.encode()).json()["id"]
    stored = list(Path(settings.storage_dir, app_id).iterdir())
    assert len(stored) == 1

    r = client.delete(f"{API}/documents/{doc_id}", params={"application_id": app_id})
    assert r.status_code == 204
    assert client.get(f"{API}/documents/{doc_id}", params={"application_id": app_id}).status_code == 404
    assert db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == uuid.UUID(doc_id))) == 0
    assert not stored[0].exists()



# ======================= chat =======================

def test_chat_new_session_returns_answer_sources_and_saves_turn(
    client: TestClient, cv_app: str, db: Session,
) -> None:
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": None,
                                         "query": "Which project did Duc build?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Duc built a RAG chatbot [Source 1]."
    assert body["abstained"] is False
    assert body["sources"][0]["number"] == 1 and body["sources"][0]["filename"] in {"cv.txt", "job.txt"}
    meta = body["metadata"]
    assert meta["retrieval_count"] >= meta["relevant_count"] >= 1 and meta["standalone_question"] is None

    session_id = uuid.UUID(body["session_id"])
    messages = db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.seq)).all()
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Which project did Duc build?"), ("assistant", body["answer"])]
    assert messages[1].sources == body["sources"] and messages[1].id == uuid.UUID(body["message_id"])
    log = db.scalars(select(RetrievalLog).where(RetrievalLog.session_id == session_id)).one()
    assert log.message_id == messages[1].id and log.application_id == cv_app and log.retrieved_chunk_ids


def test_follow_up_question_is_condensed_with_history(client: TestClient, cv_app: str, llm: ScriptedLLM) -> None:
    first = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Which project did Duc build?"})
    assert llm.condensed_prompts == []                  # câu đầu tiên không có lịch sử -> không gọi LLM viết lại

    llm.condensed = "Which technologies did Duc use in the RAG chatbot project?"
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": first.json()["session_id"],
                                         "query": "Which technologies did he use there?"})
    assert r.status_code == 200
    assert r.json()["session_id"] == first.json()["session_id"]
    assert r.json()["metadata"]["standalone_question"] == llm.condensed
    assert "Which project did Duc build?" in llm.condensed_prompts[0]       # lịch sử được gửi kèm
    assert llm.condensed in llm.answer_prompts[-1]                          # RAG nhận câu đã viết lại


def test_condense_failure_falls_back_to_original_question(
    client: TestClient, cv_app: str, llm: ScriptedLLM,
) -> None:
    first = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Which project did Duc build"})
    llm.fail_condense = True
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": first.json()["session_id"],
                                         "query": "Which database did Duc use?"})
    assert r.status_code == 200
    assert r.json()["metadata"]["standalone_question"] is None
    assert "Which database did Duc use?" in llm.answer_prompts[-1]

def test_condense_disabled_by_setting(client: TestClient, cv_app: str, llm: ScriptedLLM, settings: Settings) -> None:
    settings.chat_condense_question = False
    first = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Which project did Duc build?"})
    client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": first.json()["session_id"],
                                     "query": "Which database did Duc use?"})
    assert llm.condensed_prompts == []


def test_session_of_other_application_is_404_and_nothing_saved(
    client: TestClient, app_ids: list[str], cv_app: str, db: Session,
) -> None:
    other = _new_app(client, app_ids)
    session_id = client.post(f"{API}/chat", json={"application_id": cv_app,
                                                  "query": "Which project did Duc build?"}).json()["session_id"]
    r = client.post(f"{API}/chat", json={"application_id": other, "session_id": session_id, "query": "Hello?"})
    assert r.status_code == 404
    assert client.get(f"{API}/chat/sessions/{session_id}", params={"application_id": other}).status_code == 404
    assert db.scalar(select(func.count(Message.id)).where(Message.session_id == uuid.UUID(session_id))) == 2


def test_unknown_session_returns_404(client: TestClient, cv_app: str) -> None:
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": str(uuid.uuid4()), "query": "Hi?"})
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_chat_unknown_application_returns_404(client: TestClient) -> None:
    assert client.post(f"{API}/chat", json={"application_id": "no-such-app", "query": "Hi?"}).status_code == 404


@pytest.mark.parametrize("payload", [
    {"application_id": "cv", "query": "   "},          # chỉ có khoảng trắng -> validate_question
    {"application_id": "cv", "query": ""},             # rỗng -> Pydantic min_length
    {"application_id": "cv", "query": "x" * 1001},     # quá dài -> Pydantic max_length
    {"application_id": "cv"},                          # thiếu query
    {"application_id": "cv", "query": "Hi", "session_id": "not-a-uuid"},
])

def test_invalid_chat_request_returns_422(client: TestClient, cv_app: str, payload: dict) -> None:
    payload = {**payload, "application_id": cv_app}
    assert client.post(f"{API}/chat", json=payload).status_code == 422


def test_llm_error_returns_503_without_details_and_saves_nothing(
    client: TestClient, cv_app: str, llm: ScriptedLLM, db: Session,
) -> None:
    # Bật "công tắc hỏng": lần tới RAG gọi LLM để TRẢ LỜI, ScriptedLLM sẽ ném
    #   LLMError("429 rate limited (secret provider detail)")
    #   - "429 rate limited": giả lập Gemini báo hết quota (HTTP 429 Too Many Requests)
    #   - "secret provider detail": giả lập thông tin nội bộ của provider (tên model, URL, tài khoản...)
    #     2 chữ "429" và "secret" là DẤU HIỆU để lát nữa kiểm tra chúng có bị lộ ra ngoài không
    llm.fail_answer = True

    # Gửi câu hỏi đầu tiên (không có session_id -> không có lịch sử -> không gọi condense)
    # Luồng chạy: ChatService.chat -> rag.ask -> LLM ném LLMError
    #   -> ChatService KHÔNG bắt lỗi -> văng ra ngay, các dòng sau rag.ask KHÔNG chạy
    #      (không create_session, không add_message, không commit)
    #   -> app_error_handler (errors.py): LLMError thuộc UPSTREAM_ERRORS
    #      -> ghi lỗi gốc vào LOG server, trả cho client 503 + câu thông báo chung chung
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Which project did Duc build?"})

    # 1) Mã lỗi đúng: 503 Service Unavailable = "dịch vụ AI tạm thời lỗi, thử lại sau"
    #    (không phải 500 lỗi code, cũng không phải 4xx lỗi của người dùng)
    assert r.status_code == 503

    # 2) Không lộ thông tin nội bộ: client chỉ được thấy
    #    "AI service is temporarily unavailable. Please try again in a moment."
    #    Nếu errors.py lỡ trả str(exc) thì detail = "429 rate limited (secret provider detail)" -> assert FAIL
    assert "secret" not in r.json()["detail"] and "429" not in r.json()["detail"]

    # 3) Không lưu dữ liệu rác: đếm số phiên chat của app này trong DB THẬT -> phải bằng 0
    #    Đúng vì ChatService tạo phiên SAU khi gọi LLM: LLM lỗi -> chưa kịp tạo phiên
    #    (Nếu tạo phiên TRƯỚC khi gọi LLM, ở đây sẽ ra 1 phiên rỗng không có tin nhắn nào)
    assert db.scalar(
        select(func.count(ChatSession.id))            # SELECT count(id) FROM chat_sessions
        .where(ChatSession.application_id == cv_app)  # WHERE application_id = <app của test>
    ) == 0


def test_unrelated_question_abstains_without_calling_llm(client: TestClient, cv_app: str, llm: ScriptedLLM) -> None:
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Weather forecast tomorow?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is True and body["sources"] == []
    assert body["metadata"]["abstain_reason"] == "no_relevant_chunks"
    assert llm.answer_prompts == []

def test_list_sessions_and_get_session_detail(client: TestClient, cv_app: str) -> None:
    llm_q1, llm_q2 = "Which project did Duc build?", "Where did Duc work?"
    s1 = client.post(f"{API}/chat", json={"application_id": cv_app, "query": llm_q1}).json()["session_id"]
    client.post(f"{API}/chat", json={"application_id": cv_app, "session_id": s1, "query": llm_q2})
    s2 = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Docker?"}).json()["session_id"]

    sessions = client.get(f"{API}/chat/sessions", params={"application_id": cv_app}).json()
    assert [s["id"] for s in sessions] == [s2, s1]        # mới nhất trước
    assert sessions[1]["message_count"] == 4 and sessions[1]["first_question"] == llm_q1    

    detail = client.get(f"{API}/chat/sessions/{s1}", params={"application_id": cv_app}).json()
    assert [(m["role"], m["content"]) for m in detail["messages"]][::2] == [("user", llm_q1), ("user", llm_q2)]
    assert detail["messages"][1]["sources"] and detail["messages"][0]["sources"] == []



# ======================= admin key + lỗi chung =======================
def test_admin_key_required_when_configured(client: TestClient, settings: Settings, app_ids: list[str]) -> None:
    settings.admin_api_key = SecretStr("s3cret-key")
    app_id = f"t-api-{uuid.uuid4().hex[:8]}"
    app_ids.append(app_id)
    body = {"id": app_id, "name": "x"}
    assert client.post(f"{API}/applications", json=body).status_code == 401
    assert client.post(f"{API}/applications", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get(f"{API}/documents", params={"application_id": app_id}).status_code == 401
    assert client.post(f"{API}/applications", json=body, headers={"X-API-Key": "s3cret-key"}).status_code == 201
    # API công khai không cần key
    assert client.get(f"{API}/applications/{app_id}").status_code == 200


def test_admin_api_locked_outside_development_without_key(client: TestClient, settings: Settings) -> None:
    settings.app_env = "production"
    r = client.post(f"{API}/applications", json={"id": "t-api-prod", "name": "x"})
    assert r.status_code == 403

# Mọi lỗi không thuộc AppError đều phải trả 500
def test_unexpected_error_returns_500_without_stack_trace(client: TestClient) -> None:
    class Broken:
        def chat(self, *args, **kwargs):
            raise RuntimeError("password=hunter2 at /app/secret.py line 42")

    app.dependency_overrides[get_chat_service] = lambda: Broken()
    r = client.post(f"{API}/chat", json={"application_id": "cv", "query": "Hi?"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}


# ======================= QuestionCondenser (không cần DB) =======================

HISTORY = [("user", "Which project did Duc build?"), ("assistant", "A RAG chatbot [Source 1].")]


@pytest.mark.parametrize("raw, expected", [
    ('"Which stack did the RAG chatbot use?"', "Which stack did the RAG chatbot use?"),
    ("Standalone question: Which stack did the RAG chatbot use?", "Which stack did the RAG chatbot use?"),
    ("", "What stack?"),                    # rỗng -> giữ câu gốc
    ("x" * 1001, "What stack?"),            # quá dài -> giữ câu gốc
])
def test_condenser_cleans_or_falls_back(raw: str, expected: str) -> None:
    result = QuestionCondenser(ScriptedLLM(condensed=raw)).condense("What stack?", HISTORY)
    assert result.question == expected
    assert result.rewritten == (expected != "What stack?")

def test_condense_prompt_contains_history_and_truncates_long_turns() -> None:
    prompt = build_condense_prompt([("user", "Q1"), ("assistant", "A" * 2000)], "Q2")
    assert "User: Q1" in prompt and "Follow-up question: Q2" in prompt
    assert "A" * 500 + "..." in prompt and "A" * 501 not in prompt


# ======================= Phase 10b: đoạn trích của nguồn =======================

def test_chat_sources_include_excerpt_and_session_keeps_it(client: TestClient, cv_app: str) -> None:
    body = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Which project did Duc build?"}).json()
    src = body["sources"][0]
    assert src["text"] in {CV_TEXT, JOB_TEXT}            # đúng nội dung chunk, không phải tên file

    detail = client.get(f"{API}/chat/sessions/{body['session_id']}", params={"application_id": cv_app}).json()
    assert detail["messages"][1]["sources"][0]["text"] == src["text"]   # mở lại phiên vẫn còn đoạn trích


def test_message_saved_before_phase10b_has_null_text(client: TestClient, cv_app: str, db: Session) -> None:
    from app.repositories import chat_repository as chat_repo

    session = chat_repo.create_session(db, cv_app)
    old_source = {"number": 1, "document_id": str(uuid.uuid4()), "filename": "cv.txt", "page": None,
                  "chunk_id": str(uuid.uuid4())}                      # dạng cũ: chưa có "text"
    chat_repo.add_message(db, session, "user", "Old question?")
    chat_repo.add_message(db, session, "assistant", "Old answer [Source 1].", sources=[old_source])
    db.commit()

    r = client.get(f"{API}/chat/sessions/{session.id}", params={"application_id": cv_app})
    assert r.status_code == 200, r.text                                # không vỡ vì thiếu field
    assert r.json()["messages"][1]["sources"][0]["text"] is None

