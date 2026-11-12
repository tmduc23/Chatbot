"""
MỤC ĐÍCH: Phase 12c — cache câu trả lời (Postgres) và cache vector câu hỏi (RAM).

Điều quan trọng nhất cần chứng minh KHÔNG phải "cache trúng", mà là "cache KHÔNG trả câu trả lời cũ
khi nó đã sai": upload/xoá tài liệu, đổi model, đổi ngưỡng, hết hạn, khác application.
"""
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.api.deps import get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.db.session import SessionLocal
from app.main import app
from app.models import EMBEDDING_DIM, AnswerCacheEntry, Application, Message, RetrievalLog
from app.rag.embeddings.cache import CachedEmbeddingProvider
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.base import LLMResponse
from app.rag.llm.fake import FakeLLM
from app.rag.prompts.condense import CONDENSE_SYSTEM_INSTRUCTION
from app.rag.reranking.fake import FakeReranker
from app.services.answer_cache import AnswerCache, normalize_question

API = "/api/v1"
Q = "Which project did Duc build?"


class CountingLLM(FakeLLM):
    """calls = các lần gọi để TRẢ LỜI. Lần gọi để viết lại câu hỏi (condense) đếm riêng."""

    def __init__(self):
        super().__init__(answer="Duc built a RAG chatbot [Source 1].")
        self.fail = False
        self.condensed = Q
        self.condense_calls = 0

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        if system_instruction == CONDENSE_SYSTEM_INSTRUCTION:
            self.condense_calls += 1
            return LLMResponse(text=self.condensed, model="fake")
        if self.fail:
            raise LLMError("429 rate limited")
        return super().generate(prompt, system_instruction)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="development", admin_api_key=None, storage_dir=str(tmp_path / "storage"),
        tokenizer_encoding="whitespace", chunk_size=500, chunk_overlap=50,
        embedding_provider="fake", reranker_provider="fake", reranker_min_score=0.3, llm_provider="fake",
        grounding_check=False, chat_condense_question=False, background_ingestion=False,
        auth_enabled=False, rate_limit_enabled=False,
        answer_cache_enabled=True, answer_cache_ttl_seconds=3600,
    )


@pytest.fixture
def llm() -> CountingLLM:
    return CountingLLM()


@pytest.fixture
def client(settings: Settings, llm: CountingLLM) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: llm
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def app_ids() -> Iterator[list[str]]:
    ids: list[str] = []
    yield ids
    with SessionLocal() as db:
        db.execute(delete(Application).where(Application.id.in_(ids)))   # CASCADE xoá cả answer_cache
        db.commit()


def _new_app(client: TestClient, app_ids: list[str], text: bytes = b"Duc built a RAG chatbot project.") -> str:
    app_id = f"t-cache-{uuid.uuid4().hex[:8]}"
    app_ids.append(app_id)
    assert client.post(f"{API}/applications", json={"id": app_id, "name": "Cache"}).status_code == 201
    _upload(client, app_id, "cv.txt", text)
    return app_id


def _upload(client: TestClient, app_id: str, name: str, content: bytes) -> str:
    r = client.post(f"{API}/documents/upload", data={"application_id": app_id},
                    files={"file": (name, content, "text/plain")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _ask(client: TestClient, app_id: str, query: str = Q, session_id: str | None = None) -> dict:
    r = client.post(f"{API}/chat", json={"application_id": app_id, "query": query, "session_id": session_id})
    assert r.status_code == 200, r.text
    return r.json()


# ======================= trúng cache =======================

def test_same_question_calls_llm_once_and_still_saves_history(client: TestClient, llm: CountingLLM,
                                                               app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    first, second = _ask(client, app_id), _ask(client, app_id)
    assert len(llm.calls) == 1
    assert first["metadata"]["cached"] is False and second["metadata"]["cached"] is True
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]
    assert second["metadata"]["relevant_count"] == first["metadata"]["relevant_count"]
    # trả từ cache vẫn ghi đủ lịch sử + retrieval log như 1 lượt bình thường
    with SessionLocal() as db:
        sid = uuid.UUID(second["session_id"])
        roles = db.scalars(select(Message.role).where(Message.session_id == sid).order_by(Message.seq)).all()
        log = db.scalars(select(RetrievalLog).where(RetrievalLog.session_id == sid)).one()
        hits = db.scalar(select(AnswerCacheEntry.hits).where(AnswerCacheEntry.application_id == app_id))
    assert roles == ["user", "assistant"] and log.retrieved_chunk_ids and hits == 1


@pytest.mark.parametrize("variant", ["which project did duc build", "  Which   project did Duc build?? ",
                                     "WHICH PROJECT DID DUC BUILD."])
def test_normalized_variants_share_cache(client: TestClient, llm: CountingLLM, app_ids: list[str],
                                         variant: str) -> None:
    app_id = _new_app(client, app_ids)
    _ask(client, app_id)
    assert _ask(client, app_id, variant)["metadata"]["cached"] is True
    assert len(llm.calls) == 1


def test_normalize_question_unicode_forms() -> None:
    composed, decomposed = "Bạn ấy học ở đâu?", "Bạn ấy học ở đâu?"
    assert composed != decomposed
    assert normalize_question(composed) == normalize_question(decomposed) == "bạn ấy học ở đâu"


def test_follow_up_uses_rewritten_question_as_key(client: TestClient, llm: CountingLLM, settings: Settings,
                                                  app_ids: list[str]) -> None:
    """Câu nối tiếp được viết lại thành đúng câu đã có trong cache -> trúng, dù chữ người dùng gõ khác hẳn."""
    settings.chat_condense_question = True
    app_id = _new_app(client, app_ids)
    first = _ask(client, app_id)                                   # câu đầu: không có lịch sử, không condense
    follow = _ask(client, app_id, "And what about that project?", first["session_id"])
    assert llm.condense_calls == 1 and follow["metadata"]["standalone_question"] == Q
    assert follow["metadata"]["cached"] is True and len(llm.calls) == 1


# ======================= KHÔNG được trúng cache =======================

def test_new_document_invalidates(client: TestClient, llm: CountingLLM, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    _ask(client, app_id)
    _upload(client, app_id, "job.txt", b"Duc worked at Acme as a backend engineer.")
    assert _ask(client, app_id)["metadata"]["cached"] is False
    assert len(llm.calls) == 2


def test_deleted_document_invalidates(client: TestClient, llm: CountingLLM, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    extra = _upload(client, app_id, "job.txt", b"Duc worked at Acme as a backend engineer.")
    _ask(client, app_id)
    assert client.delete(f"{API}/documents/{extra}", params={"application_id": app_id}).status_code == 204
    assert _ask(client, app_id)["metadata"]["cached"] is False


# Mỗi setting 2 giá trị: lấy giá trị ĐẦU TIÊN khác với giá trị đang dùng. .env (Docker đưa vào biến môi trường)
# có thể đã đặt sẵn đúng giá trị đó (vd. ANSWER_PROMPT=v5) -> "đổi" mà không đổi gì thì test vô nghĩa.
@pytest.mark.parametrize("field, values", [("llm_model", ("another-model", "model-2")),
                                           ("reranker_min_score", (0.5, 0.6)),
                                           ("context_max_chars", (1234, 4321)),
                                           # cấu hình bạn thêm sau tháng 10: đổi là câu trả lời có thể đổi
                                           ("answer_prompt", ("v5", "v3")), ("retrieval_mode", ("vector", "keyword")),
                                           ("keyword_ranking", ("bm25", "ts_rank")),
                                           ("cohere_embedding_model", ("x", "y"))])
def test_config_change_invalidates(client: TestClient, settings: Settings, app_ids: list[str],
                                   field: str, values: tuple) -> None:
    app_id = _new_app(client, app_ids)
    _ask(client, app_id)
    value = next(v for v in values if v != getattr(settings, field))
    setattr(settings, field, value)
    assert _ask(client, app_id)["metadata"]["cached"] is False


def _ask_paper(client: TestClient, app_id: str, document_ids: list[str]) -> dict:
    r = client.post(f"{API}/chat", json={"application_id": app_id, "query": Q, "document_ids": document_ids})
    assert r.status_code == 200, r.text
    return r.json()


def test_paper_mode_each_paper_has_its_own_cache(client: TestClient, llm: CountingLLM, app_ids: list[str]) -> None:
    """Cùng câu hỏi ở 2 paper khác nhau KHÔNG được dùng chung câu trả lời (khoá có document_ids)."""
    app_id = _new_app(client, app_ids)
    paper_a = _upload(client, app_id, "a.txt", b"Duc built project Alpha, a retrieval model.")
    paper_b = _upload(client, app_id, "b.txt", b"Duc built project Beta for question answering.")
    assert _ask_paper(client, app_id, [paper_a])["metadata"]["cached"] is False
    assert _ask_paper(client, app_id, [paper_b])["metadata"]["cached"] is False
    assert _ask(client, app_id)["metadata"]["cached"] is False          # cả application = phạm vi khác
    assert len(llm.calls) == 3
    assert _ask_paper(client, app_id, [paper_a])["metadata"]["cached"] is True
    # thứ tự chọn không quan trọng: [A, B] và [B, A] là cùng 1 phạm vi
    _ask_paper(client, app_id, [paper_a, paper_b])
    assert _ask_paper(client, app_id, [paper_b, paper_a])["metadata"]["cached"] is True


def test_other_application_does_not_share(client: TestClient, app_ids: list[str]) -> None:
    a, b = _new_app(client, app_ids), _new_app(client, app_ids)
    _ask(client, a)
    assert _ask(client, b)["metadata"]["cached"] is False


def test_expired_entry_is_not_used_and_gets_cleaned(settings: Settings, client: TestClient,
                                                    app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    _ask(client, app_id)
    with SessionLocal() as db:
        old_key = db.scalar(select(AnswerCacheEntry.key).where(AnswerCacheEntry.application_id == app_id))
        cache = AnswerCache(db, settings)
        later = datetime.now(timezone.utc) + timedelta(seconds=settings.answer_cache_ttl_seconds + 1)
        assert cache.get(old_key, now=later) is None
        # put lúc "later" dọn dòng hết hạn của application này
        snap = cache.get(old_key)
        cache.put("0" * 64, app_id, "other question", snap, now=later)
        db.commit()
        keys = db.scalars(select(AnswerCacheEntry.key).where(AnswerCacheEntry.application_id == app_id)).all()
    assert keys == ["0" * 64]


def test_llm_error_is_not_cached(client: TestClient, llm: CountingLLM, app_ids: list[str]) -> None:
    app_id = _new_app(client, app_ids)
    llm.fail = True
    r = client.post(f"{API}/chat", json={"application_id": app_id, "query": Q})
    assert r.status_code == 503
    llm.fail = False
    assert _ask(client, app_id)["metadata"]["cached"] is False
    with SessionLocal() as db:
        assert db.scalar(select(AnswerCacheEntry.hits).where(AnswerCacheEntry.application_id == app_id)) == 0


def test_abstention_is_cached_too(client: TestClient, llm: CountingLLM, app_ids: list[str]) -> None:
    """Câu hỏi không liên quan: lần 2 không tốn cả embedding + rerank."""
    app_id = _new_app(client, app_ids)
    first, second = _ask(client, app_id, "xyz qqq"), _ask(client, app_id, "xyz qqq")
    assert first["abstained"] and second["abstained"] and second["metadata"]["cached"] is True
    assert llm.calls == []


def test_cache_disabled(client: TestClient, llm: CountingLLM, settings: Settings, app_ids: list[str]) -> None:
    settings.answer_cache_enabled = False
    app_id = _new_app(client, app_ids)
    _ask(client, app_id)
    assert _ask(client, app_id)["metadata"]["cached"] is False
    assert len(llm.calls) == 2


# ======================= cache vector câu hỏi (RAM) =======================

class CountingEmbedder(FakeEmbeddingProvider):
    def __init__(self):
        super().__init__(8)
        self.query_calls = 0
        self.doc_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return super().embed_documents(texts)


def test_embedding_cache_hits_and_lru_eviction() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbeddingProvider(inner, max_size=2)
    assert cached.dimension == 8
    v1 = cached.embed_query("a")
    v1.append(999.0)                                  # người gọi sửa list -> không làm hỏng cache
    assert cached.embed_query("a") == inner.embed_query("a")
    inner.query_calls -= 1                            # bỏ lần gọi kiểm tra ở dòng trên
    assert inner.query_calls == 1 and cached.hits == 1

    cached.embed_query("b")
    cached.embed_query("a")                           # "a" vừa dùng -> "b" thành cũ nhất
    cached.embed_query("c")                           # đầy (2) -> bỏ "b"
    calls = inner.query_calls
    cached.embed_query("a")
    assert inner.query_calls == calls                 # "a" còn
    cached.embed_query("b")
    assert inner.query_calls == calls + 1             # "b" đã bị bỏ -> gọi lại API

    cached.embed_documents(["x", "y"])
    cached.embed_documents(["x", "y"])
    assert inner.doc_calls == 2                       # tài liệu: không cache
