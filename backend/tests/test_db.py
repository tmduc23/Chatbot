"""Integration test — cần Postgres đang chạy."""
import uuid
from collections.abc import Iterator
import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application, Chunk, Document


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()          # không để lại dữ liệu test
        session.close()


def _one_hot(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def test_pgvector_installed(db: Session) -> None:
    assert db.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar() == "vector"


def test_tables_exist(db: Session) -> None:
    rows = set(db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")).scalars())
    assert {"applications", "users", "documents", "chunks",
            "chat_sessions", "messages", "retrieval_logs"} <= rows


def test_vector_search_is_filtered_by_application(db: Session) -> None:
    for app_id, dim in (("t_cv", 0), ("t_projects", 1)):
        db.add(Application(id=app_id, name=app_id))
        db.flush()
        doc = Document(application_id=app_id, filename=f"{app_id}.pdf", file_type="pdf",
                       storage_path="n/a", file_hash=uuid.uuid4().hex)
        db.add(doc)
        db.flush()
        db.add(Chunk(document_id=doc.id, application_id=app_id, chunk_index=0,
                     text=f"chunk of {app_id}", embedding=_one_hot(dim)))
    db.flush()

    # Query giống hệt chunk của t_projects, nhưng search trong t_cv
    results = db.scalars(
        select(Chunk)
        .where(Chunk.application_id == "t_cv")             # filter NGAY trong SQL
        .order_by(Chunk.embedding.cosine_distance(_one_hot(1)))         # cosine_distance cung cấp bởi pg_vector
        .limit(5)
    ).all()

    assert [c.application_id for c in results] == ["t_cv"]