"""
MỤC ĐÍCH: Phase 12b — đăng nhập + phân quyền theo application qua REST API (AUTH_ENABLED=true).

Mỗi test dựng 1 "công ty nhỏ" (fixture `world`):
    pub   : application CÔNG KHAI (như CV cho nhà tuyển dụng)
    priv  : application RIÊNG TƯ, có 1 tài liệu
    other : application RIÊNG TƯ khác (không ai trong test là thành viên)
    editor: editor của priv      viewer: viewer của priv
    admin : superadmin           outsider: đã đăng nhập, không thuộc app nào
Kiểm tra spec mục 25: user chỉ thấy đúng application mình được phép, KHÔNG tin application_id từ client.
"""
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete

from app.api.deps import get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.main import app
from app.models import EMBEDDING_DIM, Application, User
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.fake import FakeLLM
from app.rag.reranking.fake import FakeReranker
from app.services.auth_service import AuthService

API = "/api/v1"
ADMIN_KEY = "service-key-for-tests-0123456789"
PASSWORD = "a-long-test-password"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="production", auth_enabled=True, jwt_secret=SecretStr("jwt-secret-for-tests-" + "x" * 32),
        admin_api_key=SecretStr(ADMIN_KEY), access_token_minutes=5,
        storage_dir=str(tmp_path / "storage"), tokenizer_encoding="whitespace", chunk_size=500, chunk_overlap=50,
        embedding_provider="fake", reranker_provider="fake", reranker_min_score=0.3, llm_provider="fake",
        grounding_check=False, chat_condense_question=False, background_ingestion=False,
        rate_limit_enabled=False, answer_cache_enabled=False,   # Phase 12c: test riêng
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLM(answer="Duc built a RAG chatbot [Source 1].")
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@dataclass
class World:
    pub: str
    priv: str
    other: str
    emails: dict[str, str]          # vai trò -> email
    doc_id: str

    def headers(self, client: TestClient, who: str) -> dict[str, str]:
        if who == "anonymous":
            return {}
        if who == "service":
            return {"X-API-Key": ADMIN_KEY}
        r = client.post(f"{API}/auth/login", json={"email": self.emails[who], "password": PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def world(client: TestClient, settings: Settings) -> Iterator[World]:
    tag = uuid.uuid4().hex[:6]
    pub, priv, other = f"t-auth-pub-{tag}", f"t-auth-priv-{tag}", f"t-auth-other-{tag}"
    emails = {who: f"{who}-{tag}@test.local" for who in ("admin", "editor", "viewer", "outsider")}
    key = {"X-API-Key": ADMIN_KEY}
    try:
        for app_id, public in ((pub, True), (priv, False), (other, False)):
            r = client.post(f"{API}/applications", headers=key,
                            json={"id": app_id, "name": app_id, "is_public": public})
            assert r.status_code == 201, r.text
        for app_id in (pub, priv):
            r = client.post(f"{API}/documents/upload", headers=key, data={"application_id": app_id},
                            files={"file": ("cv.txt", b"Duc built a RAG chatbot using FastAPI.", "text/plain")})
            assert r.status_code == 201, r.text
        doc_id = r.json()["id"]   # tài liệu của priv
        with SessionLocal() as db:
            auth = AuthService(db, settings)
            auth.create_user(emails["admin"], PASSWORD, is_superadmin=True)
            for who in ("editor", "viewer", "outsider"):
                auth.create_user(emails[who], PASSWORD)
            auth.grant(emails["editor"], priv, "editor")
            auth.grant(emails["viewer"], priv, "viewer")
        yield World(pub=pub, priv=priv, other=other, emails=emails, doc_id=doc_id)
    finally:
        with SessionLocal() as db:
            db.execute(delete(Application).where(Application.id.in_([pub, priv, other])))
            db.execute(delete(User).where(User.email.in_(list(emails.values()))))
            db.commit()


def _chat(client: TestClient, headers: dict, app_id: str, session_id: str | None = None):
    return client.post(f"{API}/chat", headers=headers,
                       json={"application_id": app_id, "session_id": session_id, "query": "What did Duc build?"})


# ======================= đăng nhập =======================

def test_login_returns_token_and_me_shows_roles(client: TestClient, world: World) -> None:
    r = client.post(f"{API}/auth/login", json={"email": world.emails["editor"].upper(), "password": PASSWORD})
    assert r.status_code == 200
    assert r.json()["token_type"] == "bearer" and r.json()["expires_in"] == 300
    me = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert me["email"] == world.emails["editor"]           # email không phân biệt hoa thường
    assert me["memberships"] == [{"application_id": world.priv, "role": "editor"}]
    assert me["is_superadmin"] is False


def test_wrong_password_and_unknown_email_get_same_401(client: TestClient, world: World) -> None:
    wrong = client.post(f"{API}/auth/login", json={"email": world.emails["viewer"], "password": "wrong-password"})
    unknown = client.post(f"{API}/auth/login", json={"email": "nobody@test.local", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid email or password"}   # không lộ email nào có thật
    assert wrong.headers["WWW-Authenticate"] == "Bearer"


def test_bad_token_is_401_not_anonymous(client: TestClient, world: World) -> None:
    # Gửi token sai thì báo lỗi, KHÔNG lặng lẽ coi như khách (khách vẫn chat được app công khai)
    r = _chat(client, {"Authorization": "Bearer not-a-real-token"}, world.pub)
    assert r.status_code == 401
    assert client.get(f"{API}/applications", headers={"X-API-Key": "wrong"}).status_code == 401


def test_deactivated_user_is_rejected_immediately(client: TestClient, world: World, settings: Settings) -> None:
    headers = world.headers(client, "viewer")
    assert _chat(client, headers, world.priv).status_code == 200
    with SessionLocal() as db:
        AuthService(db, settings).set_active(world.emails["viewer"], False)
    assert _chat(client, headers, world.priv).status_code == 401          # token cũ còn hạn vẫn bị chặn
    r = client.post(f"{API}/auth/login", json={"email": world.emails["viewer"], "password": PASSWORD})
    assert r.status_code == 401


def test_revoked_role_takes_effect_without_new_login(client: TestClient, world: World, settings: Settings) -> None:
    headers = world.headers(client, "editor")
    assert client.get(f"{API}/documents", headers=headers, params={"application_id": world.priv}).status_code == 200
    with SessionLocal() as db:
        AuthService(db, settings).revoke(world.emails["editor"], world.priv)
    assert client.get(f"{API}/documents", headers=headers, params={"application_id": world.priv}).status_code == 404


def test_auth_routes_404_when_auth_disabled(client: TestClient, world: World, settings: Settings) -> None:
    settings.auth_enabled = False
    r = client.post(f"{API}/auth/login", json={"email": world.emails["admin"], "password": PASSWORD})
    assert r.status_code == 404


# ======================= application: ai thấy gì =======================

@pytest.mark.parametrize("who, sees", [
    ("anonymous", {"pub"}),
    ("outsider", {"pub"}),
    ("viewer", {"pub", "priv"}),
    ("admin", {"pub", "priv", "other"}),
    ("service", {"pub", "priv", "other"}),
])
def test_list_applications_shows_only_visible(client: TestClient, world: World, who: str, sees: set[str]) -> None:
    names = {"pub": world.pub, "priv": world.priv, "other": world.other}
    listed = {a["id"] for a in client.get(f"{API}/applications", headers=world.headers(client, who)).json()}
    assert {k for k, v in names.items() if v in listed} == sees


def test_private_application_hidden_as_404(client: TestClient, world: World) -> None:
    outsider = world.headers(client, "outsider")
    hidden = client.get(f"{API}/applications/{world.priv}", headers=outsider)
    missing = client.get(f"{API}/applications/t-auth-does-not-exist", headers=outsider)
    assert hidden.status_code == missing.status_code == 404      # không phân biệt "riêng tư" với "không có"
    assert client.get(f"{API}/applications/{world.priv}").status_code == 401          # khách: hãy đăng nhập
    assert client.get(f"{API}/applications/{world.pub}").status_code == 200


def test_only_superadmin_creates_application(client: TestClient, world: World) -> None:
    body = {"id": f"t-auth-new-{uuid.uuid4().hex[:6]}", "name": "New"}
    assert client.post(f"{API}/applications", json=body).status_code == 401
    assert client.post(f"{API}/applications", json=body, headers=world.headers(client, "editor")).status_code == 403
    r = client.post(f"{API}/applications", json=body, headers=world.headers(client, "admin"))
    assert r.status_code == 201 and r.json()["is_public"] is False                # mặc định riêng tư
    with SessionLocal() as db:
        db.execute(delete(Application).where(Application.id == body["id"]))
        db.commit()


# ======================= chat =======================

@pytest.mark.parametrize("who, app_key, expected", [
    ("anonymous", "pub", 200), ("anonymous", "priv", 401),
    ("outsider", "pub", 200), ("outsider", "priv", 404),
    ("viewer", "priv", 200), ("editor", "priv", 200), ("admin", "other", 200),
    ("viewer", "other", 404),
])
def test_chat_permissions(client: TestClient, world: World, who: str, app_key: str, expected: int) -> None:
    app_id = {"pub": world.pub, "priv": world.priv, "other": world.other}[app_key]
    assert _chat(client, world.headers(client, who), app_id).status_code == expected


def test_user_session_is_private_to_its_owner(client: TestClient, world: World) -> None:
    viewer, editor, outsider = (world.headers(client, w) for w in ("viewer", "editor", "outsider"))
    session_id = _chat(client, viewer, world.priv).json()["session_id"]
    url, params = f"{API}/chat/sessions/{session_id}", {"application_id": world.priv}

    assert client.get(url, headers=viewer, params=params).status_code == 200       # chủ phiên
    assert client.get(url, headers=editor, params=params).status_code == 200       # editor xem để hỗ trợ
    assert _chat(client, editor, world.priv, session_id).status_code == 404        # nhưng không chat tiếp thay
    assert _chat(client, viewer, world.priv, session_id).status_code == 200


def test_anonymous_session_on_public_app_works_like_share_link(client: TestClient, world: World) -> None:
    session_id = _chat(client, {}, world.pub).json()["session_id"]
    assert _chat(client, {}, world.pub, session_id).status_code == 200
    r = client.get(f"{API}/chat/sessions/{session_id}", params={"application_id": world.pub})
    assert r.status_code == 200 and len(r.json()["messages"]) == 4
    # user đã đăng nhập tạo phiên trên app công khai -> khách không đọc được
    mine = _chat(client, world.headers(client, "outsider"), world.pub).json()["session_id"]
    assert client.get(f"{API}/chat/sessions/{mine}", params={"application_id": world.pub}).status_code == 404


def test_list_sessions_needs_editor(client: TestClient, world: World) -> None:
    params = {"application_id": world.priv}
    assert client.get(f"{API}/chat/sessions", headers=world.headers(client, "viewer"), params=params).status_code == 403
    assert client.get(f"{API}/chat/sessions", headers=world.headers(client, "editor"), params=params).status_code == 200


# ======================= tài liệu =======================

@pytest.mark.parametrize("who, app_key, expected", [
    ("anonymous", "priv", 401), ("outsider", "priv", 404), ("outsider", "pub", 403),
    ("viewer", "priv", 403), ("editor", "priv", 200), ("editor", "other", 404),
    ("admin", "other", 200), ("service", "priv", 200),
])
def test_list_documents_permissions(client: TestClient, world: World, who: str, app_key: str, expected: int) -> None:
    app_id = {"pub": world.pub, "priv": world.priv, "other": world.other}[app_key]
    r = client.get(f"{API}/documents", headers=world.headers(client, who), params={"application_id": app_id})
    assert r.status_code == expected


def test_every_document_route_checks_role(client: TestClient, world: World) -> None:
    """Chặn lỗi "quên kiểm quyền" khi thêm route mới: viewer gọi MỌI route tài liệu đều bị 403."""
    viewer = world.headers(client, "viewer")
    params = {"application_id": world.priv}
    calls = [
        client.post(f"{API}/documents/upload", headers=viewer, data=params,
                    files={"file": ("x.txt", b"viewer must not upload this", "text/plain")}),
        client.get(f"{API}/documents", headers=viewer, params=params),
        client.get(f"{API}/documents/{world.doc_id}", headers=viewer, params=params),
        client.delete(f"{API}/documents/{world.doc_id}", headers=viewer, params=params),
    ]
    assert [c.status_code for c in calls] == [403, 403, 403, 403]
    # Đếm route tài liệu trong OpenAPI: ai thêm route mới mà quên thêm vào danh sách trên -> test đỏ
    doc_ops = [(path, method) for path, ops in app.openapi()["paths"].items()
               if path.startswith(f"{API}/documents") for method in ops]
    assert len(doc_ops) == len(calls)
    # tài liệu vẫn còn nguyên
    editor = world.headers(client, "editor")
    assert client.get(f"{API}/documents/{world.doc_id}", headers=editor, params=params).status_code == 200


def test_editor_cannot_reach_other_app_by_changing_application_id(client: TestClient, world: World) -> None:
    """Đổi application_id trong request thành app khác -> không được, dù document_id đúng."""
    editor = world.headers(client, "editor")
    r = client.get(f"{API}/documents/{world.doc_id}", headers=editor, params={"application_id": world.other})
    assert r.status_code == 404
    r = client.post(f"{API}/documents/upload", headers=editor, data={"application_id": world.other},
                    files={"file": ("x.txt", b"sneaky upload", "text/plain")})
    assert r.status_code == 404


# ======================= chế độ paper (document_ids) + phân quyền =======================

def _paper_chat(client: TestClient, headers: dict, app_id: str, document_id: str):
    return client.post(f"{API}/chat", headers=headers,
                       json={"application_id": app_id, "query": "What did Duc build?", "document_ids": [document_id]})


def test_paper_mode_follows_application_permissions(client: TestClient, world: World) -> None:
    """Chọn 1 paper không mở được cửa sau: quyền vẫn xét theo application, không theo document_id."""
    assert _paper_chat(client, world.headers(client, "viewer"), world.priv, world.doc_id).status_code == 200
    assert _paper_chat(client, world.headers(client, "outsider"), world.priv, world.doc_id).status_code == 404
    assert _paper_chat(client, {}, world.priv, world.doc_id).status_code == 401


def test_paper_mode_rejects_document_of_another_application(client: TestClient, world: World) -> None:
    """Viewer của app công khai gửi document_id của app riêng tư -> 404, không đọc được nội dung."""
    r = _paper_chat(client, world.headers(client, "outsider"), world.pub, world.doc_id)
    assert r.status_code == 404
