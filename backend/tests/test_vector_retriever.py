"""
MỤC ĐÍCH: Integration test cho vector search với databse thật + FakeEmbedding Provider

# Kiểm tra 4 điều:
    1. Chunk liên quan nhất đứng đầu
    2. Kết quảu sắp xếp điểm giảm dần và tôn trọng top_k
    3. Cách ly: không bao giờ trả về chunk của application khác,
        kể cả khi chunk đó khớp với câu hỏi hơn
    4. Câu hỏi rỗng -> không trả gì
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.retrieval import VectorRetriever
from app.services.document_service import DocumentService


EMBEDDER =  FakeEmbeddingProvider(EMBEDDING_DIM)

DOCS_A = {
    "docker.txt": "Experienced with Docker, Docker Compose and Kubernetes deployments.",
    "python.txt": "Built REST APIs with Python, FastAPI and SQLAlchemy.",
    "hobby.txt": "Hobbies: playing guitar and listening to jazz music.",
}
DOCS_B = {
    "docker_b.txt": "Docker Compose deployments. Docker Compose deployments. Docker Compose deployments.",}

@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def apps(db: Session, tmp_path: Path) -> Iterator[tuple[str, str]]:
    app_a, app_b = f"t_a_{uuid.uuid4().hex[:6]}", f"t_b_{uuid.uuid4().hex[:6]}"
    db.add_all([Application(id=app_a, name="A"), Application(id=app_b, name="B")])
    db.commit()

    service = DocumentService(db, settings=Settings(chunk_size=500, chunk_overlap=50),
                              storage_dir=tmp_path / "storage", embedder=EMBEDDER)
    for app_id, docs in ((app_a, DOCS_A), (app_b, DOCS_B)):
        for name, content in docs.items():
            f = tmp_path / f"{app_id}_{name}"
            f.write_text(content, encoding="utf-8")
            service.ingest_file(app_id, f, filename=name)

    yield app_a, app_b
    db.rollback()
    db.execute(delete(Application).where(Application.id.in_([app_a, app_b])))
    db.commit()


def test_most_relevant_chunk_ranks_first(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    results = VectorRetriever(db, EMBEDDER).retrieve("Docker Compose deployments", app_a, top_k=3)
    assert results[0].filename == "docker.txt"

def test_scores_sorted_and_top_k_respected(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    results = VectorRetriever(db, EMBEDDER).retrieve("Python FastAPI", app_a, top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_never_returns_other_application(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    # docker_b.txt (app B) khớp câu hỏi này còn tốt hơn docker.txt — nhưng không được xuất hiện
    results = VectorRetriever(db, EMBEDDER).retrieve("Docker Compose deployments", app_a, top_k=10)
    assert {r.filename for r in results} == set(DOCS_A)

def test_empty_query_returns_nothing(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    assert VectorRetriever(db, EMBEDDER).retrieve("   ", app_a, top_k=5) == []