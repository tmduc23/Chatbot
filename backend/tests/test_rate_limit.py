"""
MỤC ĐÍCH: Phase 12c — rate limiting.

Phần 1: RateLimiter với Postgres thật (đếm đúng, sang cửa sổ mới thì reset, nhiều thread không đếm sót).
Phần 2: qua API (429 + Retry-After, đếm theo user / IP, X-API-Key không bị giới hạn, chống dò mật khẩu).
Mỗi test dùng key / IP ngẫu nhiên riêng -> không ảnh hưởng nhau và không ảnh hưởng dữ liệu thật.
"""
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, func, select

from app.api.deps import get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.main import app
from app.models import EMBEDDING_DIM, Application, RateLimitCounter, User
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.fake import FakeLLM
from app.rag.reranking.fake import FakeReranker
from app.services.auth_service import AuthService
from app.services.rate_limiter import RateLimiter, window_start

API = "/api/v1"
ADMIN_KEY = "rate-limit-test-key-0123456789"
T0 = datetime(2030, 1, 1, 10, 0, 17, tzinfo=timezone.utc)


@pytest.fixture
def key_prefix() -> Iterator[str]:
    prefix = f"t-rl-{uuid.uuid4().hex[:8]}"
    yield prefix
    with SessionLocal() as db:
        db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like(f"{prefix}%")))
        db.commit()


# ======================= RateLimiter =======================

def test_window_start_rounds_down() -> None:
    assert window_start(T0, 60) == datetime(2030, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert window_start(T0, 86400) == datetime(2030, 1, 1, tzinfo=timezone.utc)


def test_blocks_after_limit_and_reports_retry_after(key_prefix: str) -> None:
    with SessionLocal() as db:
        limiter = RateLimiter(db)
        results = [limiter.hit(key_prefix, limit=3, window_seconds=60, now=T0) for _ in range(5)]
    assert [r.allowed for r in results] == [True, True, True, False, False]
    assert [r.count for r in results] == [1, 2, 3, 4, 5]
    assert results[-1].retry_after == 43          # 10:00:17 -> cửa sổ mới lúc 10:01:00


def test_new_window_resets_count_and_keys_are_independent(key_prefix: str) -> None:
    with SessionLocal() as db:
        limiter = RateLimiter(db)
        for _ in range(3):
            limiter.hit(f"{key_prefix}:a", 2, 60, now=T0)
        assert not limiter.hit(f"{key_prefix}:a", 2, 60, now=T0).allowed
        assert limiter.hit(f"{key_prefix}:b", 2, 60, now=T0).allowed                        # key khác
        assert limiter.hit(f"{key_prefix}:a", 2, 60, now=T0 + timedelta(seconds=43)).count == 1   # cửa sổ mới


def test_concurrent_hits_are_counted_exactly(key_prefix: str) -> None:
    """8 thread x 10 lần, mỗi thread 1 connection riêng: upsert nguyên tử -> đúng 80, không đếm sót."""
    def worker() -> None:
        with SessionLocal() as db:
            limiter = RateLimiter(db)
            for _ in range(10):
                limiter.hit(key_prefix, 1000, 60, now=T0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with SessionLocal() as db:
        assert db.scalar(select(RateLimitCounter.count).where(RateLimitCounter.key == key_prefix)) == 80


def test_old_windows_are_cleaned_up(key_prefix: str) -> None:
    with SessionLocal() as db:
        limiter = RateLimiter(db)
        limiter.hit(f"{key_prefix}:old", 5, 60, now=T0)
        limiter.hit(f"{key_prefix}:new", 5, 60, now=T0 + timedelta(days=3))   # mở cửa sổ mới -> dọn rác
        keys = db.scalars(select(RateLimitCounter.key).where(RateLimitCounter.key.like(f"{key_prefix}%"))).all()
    assert keys == [f"{key_prefix}:new"]


# ======================= qua API =======================

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="production", admin_api_key=SecretStr(ADMIN_KEY), storage_dir=str(tmp_path / "storage"),
        tokenizer_encoding="whitespace", chunk_size=500, chunk_overlap=50,
        embedding_provider="fake", reranker_provider="fake", reranker_min_score=0.3, llm_provider="fake",
        grounding_check=False, chat_condense_question=False, background_ingestion=False,
        auth_enabled=False, answer_cache_enabled=False,
        rate_limit_enabled=True, rate_limit_chat_per_minute=3, rate_limit_chat_per_day=100,
        rate_limit_login_per_minute=3, rate_limit_upload_per_minute=20, trust_forwarded_for=True,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLM()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def app_id(client: TestClient) -> Iterator[str]:
    app_id = f"t-rl-{uuid.uuid4().hex[:8]}"
    key = {"X-API-Key": ADMIN_KEY}
    assert client.post(f"{API}/applications", headers=key, json={"id": app_id, "name": "RL"}).status_code == 201
    client.post(f"{API}/documents/upload", headers=key, data={"application_id": app_id},
                files={"file": ("cv.txt", b"Duc built a RAG chatbot.", "text/plain")})
    try:
        yield app_id
    finally:
        with SessionLocal() as db:
            db.execute(delete(Application).where(Application.id == app_id))
            db.commit()


@pytest.fixture
def ip() -> Iterator[str]:
    """IP giả ngẫu nhiên cho mỗi test (qua X-Forwarded-For, settings.trust_forwarded_for=True)."""
    ip = f"10.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}"
    yield ip
    with SessionLocal() as db:
        db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like(f"%:ip:{ip}:%")))
        db.commit()


def _chat(client: TestClient, app_id: str, headers: dict):
    return client.post(f"{API}/chat", headers=headers, json={"application_id": app_id, "query": "What did Duc build?"})


def test_chat_returns_429_with_retry_after(client: TestClient, app_id: str, ip: str) -> None:
    headers = {"X-Forwarded-For": ip}
    codes = [_chat(client, app_id, headers).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = _chat(client, app_id, headers)
    assert r.status_code == 429 and 1 <= int(r.headers["Retry-After"]) <= 60
    assert "Too many requests" in r.json()["detail"]
    # IP khác vẫn hỏi được bình thường
    assert _chat(client, app_id, {"X-Forwarded-For": ip + "9"}).status_code == 200
    with SessionLocal() as db:
        db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like(f"%:ip:{ip}9:%")))
        db.commit()


def test_daily_limit(client: TestClient, app_id: str, ip: str, settings: Settings) -> None:
    settings.rate_limit_chat_per_minute, settings.rate_limit_chat_per_day = 100, 2
    codes = [_chat(client, app_id, {"X-Forwarded-For": ip}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_forwarded_for_ignored_unless_trusted(client: TestClient, app_id: str, settings: Settings) -> None:
    """Không đứng sau proxy mà tin X-Forwarded-For -> kẻ spam đổi IP giả mỗi request là né được."""
    settings.trust_forwarded_for = False
    try:
        codes = [_chat(client, app_id, {"X-Forwarded-For": f"10.9.9.{i}"}).status_code for i in range(4)]
        assert codes == [200, 200, 200, 429]      # cả 4 request bị tính cho cùng 1 IP thật ("testclient")
    finally:
        with SessionLocal() as db:
            db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like("%:ip:testclient:%")))
            db.commit()


def test_admin_api_key_is_not_limited(client: TestClient, app_id: str, ip: str) -> None:
    headers = {"X-Forwarded-For": ip, "X-API-Key": ADMIN_KEY}
    codes = [_chat(client, app_id, headers).status_code for _ in range(5)]
    assert codes == [200] * 5, codes


def test_disabled_means_no_limit(client: TestClient, app_id: str, ip: str, settings: Settings) -> None:
    settings.rate_limit_enabled = False
    assert all(_chat(client, app_id, {"X-Forwarded-For": ip}).status_code == 200 for _ in range(5))


def test_login_brute_force_is_blocked_even_with_right_password(
    client: TestClient, ip: str, settings: Settings,
) -> None:
    settings.auth_enabled, settings.jwt_secret = True, SecretStr("rl-jwt-secret-" + "x" * 40)
    email = f"t-rl-{uuid.uuid4().hex[:6]}@test.local"
    with SessionLocal() as db:
        AuthService(db, settings).create_user(email, "the-right-password")
    try:
        login = lambda pw: client.post(f"{API}/auth/login", headers={"X-Forwarded-For": ip},  # noqa: E731
                                       json={"email": email, "password": pw})
        assert [login("guess-%d" % i).status_code for i in range(3)] == [401, 401, 401]
        assert login("the-right-password").status_code == 429     # hết lượt: đúng mật khẩu cũng phải chờ
    finally:
        with SessionLocal() as db:
            db.execute(delete(User).where(User.email == email))
            db.commit()


def test_logged_in_users_have_their_own_bucket(client: TestClient, app_id: str, ip: str, settings: Settings) -> None:
    """Đếm theo user: 2 người cùng 1 IP (cùng công ty, cùng wifi) không dùng chung giới hạn."""
    settings.auth_enabled, settings.jwt_secret = True, SecretStr("rl-jwt-secret-" + "x" * 40)
    tag = uuid.uuid4().hex[:6]
    emails = [f"t-rl-a-{tag}@test.local", f"t-rl-b-{tag}@test.local"]
    with SessionLocal() as db:
        auth = AuthService(db, settings)
        for e in emails:
            auth.create_user(e, "a-long-password")
            auth.grant(e, app_id, "viewer")
        user_ids = [auth.require_user(e).id for e in emails]
    try:
        tokens = [client.post(f"{API}/auth/login", json={"email": e, "password": "a-long-password"},
                              headers={"X-Forwarded-For": ip + "1"}).json()["access_token"] for e in emails]
        a, b = ({"Authorization": f"Bearer {t}", "X-Forwarded-For": ip} for t in tokens)
        assert [_chat(client, app_id, a).status_code for _ in range(4)] == [200, 200, 200, 429]
        assert _chat(client, app_id, b).status_code == 200        # cùng IP, khác user -> chưa bị chặn
    finally:
        with SessionLocal() as db:
            db.execute(delete(User).where(User.email.in_(emails)))
            for uid in user_ids:
                db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like(f"%:user:{uid}:%")))
            db.execute(delete(RateLimitCounter).where(RateLimitCounter.key.like(f"%:ip:{ip}1:%")))
            db.commit()
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(User).where(User.email.in_(emails))) == 0
