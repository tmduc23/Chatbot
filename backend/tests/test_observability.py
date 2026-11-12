"""
MỤC ĐÍCH: Phase 12d — structured logging (JSON, request_id, che secret), metrics (/metrics),
và các điểm của security review kiểm được bằng code (header bảo mật, /docs, CORS, cảnh báo cấu hình).
"""
import io
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from pydantic import SecretStr
from sqlalchemy import delete

from app.api.deps import get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.core.config_check import security_warnings
from app.core.exceptions import LLMError
from app.core.logging import JsonFormatter, build_handler, redact, secret_values
from app.db.session import SessionLocal
from app.main import app, create_app
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.base import LLMProvider, LLMResponse
from app.rag.reranking.fake import FakeReranker

API = "/api/v1"
GEMINI_KEY = "AIzaFakeGeminiKeyForTests1234567890"
JWT_SECRET = "jwt-secret-for-observability-tests-" + "x" * 20


class SwitchLLM(LLMProvider):
    def __init__(self):
        self.fail = False

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        if self.fail:
            raise LLMError(f"401 invalid key {GEMINI_KEY}")   # lỗi của provider có chứa key -> phải bị che
        return LLMResponse(text="Duc built a RAG chatbot [Source 1].", model="fake", input_tokens=120,
                           output_tokens=30)


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        app_env="development", admin_api_key=None, storage_dir=str(tmp_path / "storage"),
        tokenizer_encoding="whitespace", chunk_size=500, chunk_overlap=50,
        embedding_provider="fake", reranker_provider="fake", reranker_min_score=0.3, llm_provider="fake",
        grounding_check=False, chat_condense_question=False, background_ingestion=False,
        auth_enabled=False, rate_limit_enabled=False, answer_cache_enabled=False,
        gemini_api_key=SecretStr(GEMINI_KEY), jwt_secret=SecretStr(JWT_SECRET), log_format="json",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


@pytest.fixture
def llm() -> SwitchLLM:
    return SwitchLLM()


@pytest.fixture
def client(settings: Settings, llm: SwitchLLM) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: llm
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def logs(settings: Settings) -> Iterator[io.StringIO]:
    """Gắn thêm 1 handler JSON (đúng loại production dùng) ghi vào bộ nhớ để test đọc lại."""
    stream = io.StringIO()
    handler = build_handler(settings, stream)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield stream
    root.removeHandler(handler)
    root.setLevel(old_level)


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@pytest.fixture
def app_id(client: TestClient) -> Iterator[str]:
    app_id = f"t-obs-{uuid.uuid4().hex[:8]}"
    assert client.post(f"{API}/applications", json={"id": app_id, "name": "Obs"}).status_code == 201
    r = client.post(f"{API}/documents/upload", data={"application_id": app_id},
                    files={"file": ("cv.txt", b"Duc built a RAG chatbot project.", "text/plain")})
    assert r.status_code == 201
    try:
        yield app_id
    finally:
        with SessionLocal() as db:
            db.execute(delete(Application).where(Application.id == app_id))
            db.commit()


def _metric(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ======================= logging =======================

def test_json_formatter_has_fields_extra_and_exception() -> None:
    formatter = JsonFormatter(secrets=[])
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("t").makeRecord("t", logging.ERROR, __file__, 1, "failed %s", ("job",),
                                                   exc_info=sys.exc_info(),
                                                   extra={"status": 500, "duration_ms": 12.5})
    data = json.loads(formatter.format(record))
    assert data["level"] == "ERROR" and data["logger"] == "t" and data["message"] == "failed job"
    assert data["status"] == 500 and data["duration_ms"] == 12.5
    assert "ValueError: boom" in data["exception"] and data["ts"].endswith("+00:00")


def test_secret_values_finds_every_secretstr(tmp_path: Path) -> None:
    s = _settings(tmp_path, cohere_rerank_api_key=SecretStr("cohere-rerank-secret-123"),
                  openrouter_api_key=SecretStr("sk-or-v1-abcdefgh12345678"), admin_api_key=SecretStr("short"))
    found = secret_values(s)
    assert {GEMINI_KEY, JWT_SECRET, "cohere-rerank-secret-123", "sk-or-v1-abcdefgh12345678"} <= set(found)
    assert "short" not in found          # < 8 ký tự: không che (tránh thay nhầm chữ thường)


@pytest.mark.parametrize("text, leaked", [
    (f"calling https://api.example.com?key={GEMINI_KEY}", GEMINI_KEY),
    ("headers={'Authorization': 'Bearer abc.def-123'}", "abc.def-123"),
    ("token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl rejected", "eyJzdWIiOiIxIn0"),
    ("X-API-Key: my-admin-key-value", "my-admin-key-value"),
    ('{"x-api-key": "my-admin-key-value"}', "my-admin-key-value"),
])
def test_redact(text: str, leaked: str) -> None:
    out = redact(text, [GEMINI_KEY])
    assert leaked not in out and "***" in out


# ======================= request id + log của request =======================

def test_every_response_has_request_id(client: TestClient) -> None:
    r1, r2 = client.get("/health"), client.get("/health")
    assert len(r1.headers["X-Request-ID"]) == 16 and r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]
    assert client.get("/health", headers={"X-Request-ID": "frontend-abc-123"}).headers["X-Request-ID"] == \
        "frontend-abc-123"                                           # mã hợp lệ từ client: giữ nguyên
    for bad in ("short", "has space and\nnewline", "x" * 100):
        assert client.get("/health", headers={"X-Request-ID": bad}).headers["X-Request-ID"] != bad


def test_logs_of_one_request_share_its_request_id(client: TestClient, logs: io.StringIO) -> None:
    r = client.get(f"{API}/applications/t-obs-does-not-exist")
    assert r.status_code == 404
    rid = r.headers["X-Request-ID"]
    mine = [line for line in _lines(logs) if line["request_id"] == rid]
    # 1 dòng từ error handler (404) + 1 dòng "request" của middleware, cùng mã
    assert {line["logger"] for line in mine} >= {"app.api.errors", "app.request"}
    access = next(line for line in mine if line["logger"] == "app.request")
    assert access["route"] == "/api/v1/applications/{application_id}"     # mẫu route, không phải id thật
    assert access["status"] == 404 and access["method"] == "GET" and access["user"] == "anonymous"
    assert "t-obs-does-not-exist" not in access["route"]


def test_upstream_error_is_logged_without_the_key(client: TestClient, app_id: str, llm: SwitchLLM,
                                                  logs: io.StringIO) -> None:
    llm.fail = True
    before = _metric("upstream_errors_total", {"error": "LLMError"})
    r = client.post(f"{API}/chat", json={"application_id": app_id, "query": "What did Duc build?"})
    assert r.status_code == 503
    assert _metric("upstream_errors_total", {"error": "LLMError"}) == before + 1
    text = logs.getvalue()
    assert "invalid key" in text and GEMINI_KEY not in text       # log có lỗi, nhưng key đã bị che


# ======================= metrics =======================

def test_metrics_endpoint_uses_route_templates(client: TestClient) -> None:
    doc_id = uuid.uuid4()
    client.get(f"{API}/documents/{doc_id}", params={"application_id": "t-obs-none"})
    body = client.get("/metrics").text
    assert 'route="/api/v1/documents/{document_id}"' in body
    assert str(doc_id) not in body                                 # id thật không thành nhãn
    assert "http_request_duration_seconds_bucket" in body and 'route="/metrics"' not in body


def test_chat_updates_rag_metrics_and_documents_gauge(client: TestClient, app_id: str) -> None:
    labels = {"application": app_id, "outcome": "answered", "cached": "false"}
    total_before = _metric("rag_stage_duration_seconds_count", {"stage": "total"})
    tokens_before = _metric("llm_tokens_total", {"direction": "output"})
    assert client.post(f"{API}/chat", json={"application_id": app_id, "query": "What did Duc build?"}).status_code == 200
    assert _metric("chat_answers_total", labels) == 1
    assert _metric("rag_stage_duration_seconds_count", {"stage": "total"}) == total_before + 1
    assert _metric("llm_tokens_total", {"direction": "output"}) == tokens_before + 30
    client.get("/metrics")
    assert _metric("documents", {"status": "READY"}) >= 1


def test_metrics_access_outside_development(tmp_path: Path) -> None:
    prod = _settings(tmp_path, app_env="production", log_format="json")
    prod_app = create_app(prod)
    prod_app.dependency_overrides[get_settings] = lambda: prod
    with TestClient(prod_app) as c:
        assert c.get("/metrics").status_code == 403                # quên đặt token -> KHÓA
        prod.metrics_token = SecretStr("metrics-token-1234567890")
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        r = c.get("/metrics", headers={"Authorization": "Bearer metrics-token-1234567890"})
        assert r.status_code == 200 and "http_requests_total" in r.text


# ======================= security review =======================

def test_api_responses_have_security_headers(client: TestClient) -> None:
    r = client.get(f"{API}/applications")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in r.headers["Content-Security-Policy"]
    # /docs (Swagger) cần tải script -> không gắn CSP chặt ở ngoài /api
    assert "Content-Security-Policy" not in client.get("/docs").headers


def test_docs_hidden_in_production(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, app_env="production"))) as c:
        assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    with TestClient(create_app(_settings(tmp_path, app_env="production", expose_api_docs=True))) as c:
        assert c.get("/docs").status_code == 200                   # bật lại có chủ ý được
    with TestClient(create_app(_settings(tmp_path))) as c:
        assert c.get("/openapi.json").status_code == 200           # development: bật


def test_cors_allows_only_the_frontend_origin(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, cors_origins="http://localhost:3000"))) as c:
        preflight = {"Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "authorization,content-type"}
        ok = c.options(f"{API}/chat", headers={"Origin": "http://localhost:3000", **preflight})
        assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://localhost:3000"
        evil = c.options(f"{API}/chat", headers={"Origin": "https://evil.example", **preflight})
        assert "access-control-allow-origin" not in evil.headers
        put = c.options(f"{API}/chat", headers={"Origin": "http://localhost:3000",
                                                "Access-Control-Request-Method": "PUT"})
        assert put.status_code == 400                              # method không dùng tới -> không cho
        r = c.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "X-Request-ID" in r.headers["access-control-expose-headers"]


def test_security_warnings(tmp_path: Path) -> None:
    assert security_warnings(_settings(tmp_path)) == []                         # development: im lặng
    loose = security_warnings(_settings(tmp_path, app_env="production", log_format="text",
                                        cors_origins="*", database_url="postgresql+psycopg://rag:change_me@db/x"))
    joined = " | ".join(loose)
    for needle in ("AUTH_ENABLED=false", "RATE_LIMIT_ENABLED=false", "change_me", "CORS_ORIGINS", "LOG_FORMAT",
                   "METRICS_TOKEN", "ADMIN_API_KEY"):
        assert needle in joined
    strict = _settings(tmp_path, app_env="production", auth_enabled=True, rate_limit_enabled=True,
                       admin_api_key=SecretStr("admin-key-1234567890"), metrics_token=SecretStr("m" * 20),
                       database_url="postgresql+psycopg://rag:Str0ng-Pass@db/ragdb", log_format="json",
                       cors_origins="https://cv.example.com")   # Phase 13: production chỉ https
    assert security_warnings(strict) == []


def test_empty_env_values_mean_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ADMIN_API_KEY=` để trống trong .env = chưa đặt (không phải key rỗng); bool để trống không làm sập."""
    for name in ("ADMIN_API_KEY", "JWT_SECRET", "METRICS_TOKEN", "EXPOSE_API_DOCS", "RERANKER_MIN_SCORE"):
        monkeypatch.setenv(name, "")
    s = Settings()
    assert (s.admin_api_key, s.jwt_secret, s.metrics_token, s.expose_api_docs, s.reranker_min_score) == \
        (None, None, None, None, None)


def test_redaction_covers_keys_added_after_phase_12() -> None:
    """Key bạn thêm sau này (Groq, NVIDIA, key OpenRouter theo vai trò, Cohere riêng) cũng bị che trong log."""
    s = Settings(_env_file=None, groq_api_key=SecretStr("gsk_groq_secret_123456"),
                 nvidia_api_key=SecretStr("nvapi-nvidia-secret-123456"),
                 openrouter_judge_api_key=SecretStr("sk-or-judge-secret-123456"),
                 cohere_rerank_api_key=SecretStr("cohere-rerank-secret-123456"))
    line = "keys: gsk_groq_secret_123456 nvapi-nvidia-secret-123456 sk-or-judge-secret-123456 cohere-rerank-secret-123456"
    out = redact(line, secret_values(s))
    for raw in ("gsk_groq", "nvapi-nvidia", "sk-or-judge", "cohere-rerank"):
        assert raw not in out
