"""MỤC ĐÍCH: Phase 12a — upload qua API khi BACKGROUND_INGESTION=true: trả 202 + PENDING ngay."""
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.deps import get_embedder
from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.main import app
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider

API = "/api/v1"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(app_env="development", admin_api_key=None, storage_dir=str(tmp_path / "storage"),
                    tokenizer_encoding="whitespace", embedding_provider="fake", background_ingestion=True,
                    auth_enabled=False, rate_limit_enabled=False)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def app_id(client: TestClient) -> Iterator[str]:
    app_id = f"t-bg-{uuid.uuid4().hex[:8]}"
    assert client.post(f"{API}/applications", json={"id": app_id, "name": "Background"}).status_code == 201
    try:
        yield app_id
    finally:
        with SessionLocal() as db:
            db.execute(delete(Application).where(Application.id == app_id))
            db.commit()


def _upload(client: TestClient, app_id: str, name: str, content: bytes):
    return client.post(f"{API}/documents/upload", data={"application_id": app_id},
                       files={"file": (name, content, "text/plain")})


def test_upload_returns_202_and_pending(client: TestClient, app_id: str) -> None:
    r = _upload(client, app_id, "cv.txt", b"Duc knows Docker and FastAPI.")
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "PENDING" and r.json()["attempts"] == 0
    # client hỏi lại trạng thái bằng GET /documents/{id}
    detail = client.get(f"{API}/documents/{r.json()['id']}", params={"application_id": app_id})
    assert detail.status_code == 200


def test_upload_same_file_while_queued_returns_409(client: TestClient, app_id: str) -> None:
    assert _upload(client, app_id, "cv.txt", b"Duc knows Docker.").status_code == 202
    assert _upload(client, app_id, "cv-again.txt", b"Duc knows Docker.").status_code == 409
