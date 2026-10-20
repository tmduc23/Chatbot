"""
MỤC ĐÍCH: Integration test cho KeywordRetriever và HybridRetriever với database thật

Kiểm tra:
    - migration 0002 đã tạo cột search_vector và Postgres tự điền nó
    - keyword search tìm dudssng từ kháo tìm kiếm (vd. "pgvector")
    - Cách ly application ở cả keyword lẫn hybrid
    - Câu hỏi toàn stop-word hoặc không khớp -> không trả gì
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application, Chunk
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.retrieval import KeywordRetriever, build_retriever
from app.services.document_service import DocumentService

EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
SETTINGS = Settings(chunk_size=500, chunk_overlap=50, retrieval_candidate_k=10, rrf_k=60)

DOCS_A = {
    "rag.txt": "Built a RAG chatbot using PostgreSQL with pgvector and hybrid retrieval.",
    "devops.txt": "Deployed services with Docker Compose and GitHub Actions CI/CD.",
    "edu.txt": "Bachelor of Computer Science, Hanoi University of Science and Technology.",
}
DOCS_B = {"secret.txt": "pgvector pgvector pgvector internal secret notes."}


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
    try:
        service = DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / "s", embedder=EMBEDDER)
        for app_id, docs in ((app_a, DOCS_A), (app_b, DOCS_B)):
            for name, content in docs.items():
                f = tmp_path / f"{app_id}_{name}"
                f.write_text(content, encoding="utf-8")
                service.ingest_file(app_id, f, filename=name)
        yield app_a, app_b
    finally:
        # kể cả khi nạp lỗi giữa chừng cũng dọn sạch, không để rác trong DB
        db.rollback()
        db.execute(delete(Application).where(Application.id.in_([app_a, app_b])))
        db.commit()

# postgres tự điền search_vector và có từ pgvector
def test_search_vector_is_generated_by_postgres(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    vectors = db.scalars(select(Chunk.search_vector).where(Chunk.application_id  == app_a)).all()
    assert vectors and all(v for v in vectors)  # vectors không rỗng và mỗi search vector đều có giá trị
    assert any("pgvector" in v for v in vectors)

# Không lọt vào kết quả app A
def test_keyword_finds_rare_exact_term(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    results = KeywordRetriever(db).retrieve("Have you used pgvector?", app_a, top_k=5)
    assert [r.filename for r in results] == ["rag.txt"]
    assert results[0].score > 0

def test_keyword_isolation(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    results = KeywordRetriever(db).retrieve("pgvector secret", app_a, top_k=10)
    assert "secret.txt" not in {r.filename for r in results}

# không cos hoặc toàn stop_word
def test_keyword_no_match_or_only_stopwords(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    assert KeywordRetriever(db).retrieve("blockchain", app_a, top_k=5) == []
    assert KeywordRetriever(db).retrieve("what is it?", app_a, top_k=5) == []


def test_hybrid_combines_and_isolates(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    retriever = build_retriever(db, SETTINGS, embedder=EMBEDDER, mode="hybrid")
    results = retriever.retrieve("pgvector Docker", app_a, top_k=5)
    names = [r.filename for r in results]
    assert set(names[:2]) == {"rag.txt", "devops.txt"}
    assert "secret.txt" not in names
    assert all(r.ranks for r in results)