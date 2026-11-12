"""
MỤC ĐÍCH: Test index BM25 RIÊNG cho từng application (pg_textsearch).

Kiểm tra:
  - tên index hợp lệ, không trùng giữa 2 app; literal SQL an toàn với dấu nháy (không cần DB);
  - có index riêng -> retriever tự dùng, điểm tính theo thống kê CỦA APP (khác index chung);
  - chạy nhiều lần trong 1 session vẫn đúng (psycopg tự prepare câu lệnh sau 5 lần);
  - tạo index TRƯỚC khi nạp: chunk nạp sau vẫn vào index;
  - xoá index riêng -> quay về index chung, không lỗi.

Database chưa có index ix_chunks_bm25 (Postgres 16, chưa có extension) -> các test dùng DB tự BỎ QUA.
"""
import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.retrieval import KeywordRetriever
from app.rag.retrieval.bm25_index import create_app_index, drop_app_index, read_index_options
from app.rag.retrieval.keyword_retriever import (
    _PGTS_SQL, PGTS_INDEX, app_index_name, app_index_sql, sql_literal,
)
from app.services.document_service import DocumentService

EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
SETTINGS = Settings(chunk_size=500, chunk_overlap=50)

# App A: "model" có trong cả 4 file (phổ biến TRONG A), "booktest" chỉ 1 file (hiếm TRONG A).
DOCS_A = {
    "common.txt": "The model model model is a strong model for every model task.",
    "rare.txt": "We release BookTest, a new dataset, and train a model on it.",
    "other1.txt": "Another model trained on news articles.",
    "other2.txt": "A baseline model with logistic regression.",
}
# App B: "booktest" ở khắp nơi -> trên TOÀN BẢNG, booktest không còn hiếm như trong A.
DOCS_B = {f"b{i}.txt": f"BookTest corpus copy number {i}." for i in range(30)}
QUERY = "model BookTest"


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    has_index = session.execute(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :name"), {"name": PGTS_INDEX}
    ).first()
    if has_index is None:
        session.close()
        pytest.skip("chưa có index ix_chunks_bm25 (cần Postgres 17+ có pg_textsearch, migration 0006)")
    yield session
    session.close()


def _ingest(db: Session, tmp_path: Path, app_id: str, docs: dict[str, str]) -> None:
    service = DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / "s", embedder=EMBEDDER)
    for name, content in docs.items():
        f = tmp_path / f"{app_id}_{name}"
        f.write_text(content, encoding="utf-8")
        service.ingest_file(app_id, f, filename=name)


@pytest.fixture
def apps(db: Session, tmp_path: Path) -> Iterator[tuple[str, str]]:
    app_a, app_b = f"t_idx_a_{uuid.uuid4().hex[:6]}", f"t_idx_b_{uuid.uuid4().hex[:6]}"
    db.add_all([Application(id=app_a, name="A"), Application(id=app_b, name="B")])
    db.commit()
    _ingest(db, tmp_path, app_b, DOCS_B)
    yield app_a, app_b
    db.rollback()
    for app_id in (app_a, app_b):
        drop_app_index(db, app_id)
    db.execute(delete(Application).where(Application.id.in_([app_a, app_b])))
    db.commit()


# ---------- không cần DB ----------

def test_app_index_name_is_valid_and_unique() -> None:
    a, b = app_index_name("qasper-test"), app_index_name("qasper_test")
    assert a != b                                       # chỉ khác dấu vẫn ra 2 tên khác nhau
    assert a.startswith("ix_chunks_bm25_qasper_test_")
    long_name = app_index_name("x" * 50)
    for name in (a, b, long_name, app_index_name("Ứng-dụng CV")):
        assert re.fullmatch(r"[a-z0-9_]+", name) and len(name) <= 63


def test_sql_literal_escapes_quotes() -> None:
    assert sql_literal("cv") == "'cv'"
    assert sql_literal("o'brien") == "'o''brien'"
    sql = app_index_sql("o'brien")
    assert "application_id = 'o''brien'" in sql and app_index_name("o'brien") in sql
    assert ":app" in _PGTS_SQL and PGTS_INDEX in _PGTS_SQL   # câu cho index chung giữ tham số :app


# ---------- cần Postgres 17 + pg_textsearch ----------

def test_own_index_uses_app_statistics(db: Session, tmp_path: Path, apps: tuple[str, str]) -> None:
    app_a, app_b = apps
    _ingest(db, tmp_path, app_a, DOCS_A)
    shared = KeywordRetriever(db, ranking="pg_textsearch").retrieve(QUERY, app_a, top_k=5)

    create_app_index(db, app_a)
    db.commit()
    retriever = KeywordRetriever(db, ranking="pg_textsearch")
    assert retriever.has_own_index(app_a) and not retriever.has_own_index(app_b)
    own = retriever.retrieve(QUERY, app_a, top_k=5)

    assert own[0].filename == "rare.txt"                     # trong A, booktest hiếm -> thắng
    assert {r.filename for r in own} == set(DOCS_A)          # vẫn chỉ chunk của A
    score = lambda results: {r.filename: round(r.score, 4) for r in results}   # noqa: E731
    assert score(own) != score(shared)                       # thống kê của A khác thống kê toàn bảng
    for _ in range(8):                                       # app B (chưa có index riêng) vẫn dùng index chung
        assert len(retriever.retrieve(QUERY, app_b, top_k=50)) == len(DOCS_B)


def test_own_index_survives_prepared_statements(db: Session, tmp_path: Path, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    _ingest(db, tmp_path, app_a, DOCS_A)
    create_app_index(db, app_a)
    db.commit()
    retriever = KeywordRetriever(db, ranking="pg_textsearch")
    first = [r.chunk_id for r in retriever.retrieve(QUERY, app_a, top_k=5)]
    for _ in range(8):                                       # psycopg prepare sau 5 lần chạy cùng câu
        assert [r.chunk_id for r in retriever.retrieve(QUERY, app_a, top_k=5)] == first


def test_index_created_before_ingest_picks_up_new_chunks(db: Session, tmp_path: Path,
                                                         apps: tuple[str, str]) -> None:
    app_a, _ = apps
    create_app_index(db, app_a)
    create_app_index(db, app_a)                              # gọi lại -> không lỗi (IF NOT EXISTS)
    db.commit()
    _ingest(db, tmp_path, app_a, DOCS_A)
    results = KeywordRetriever(db, ranking="pg_textsearch").retrieve(QUERY, app_a, top_k=5)
    assert results[0].filename == "rare.txt"


def test_drop_falls_back_to_shared_index(db: Session, tmp_path: Path, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    _ingest(db, tmp_path, app_a, DOCS_A)
    create_app_index(db, app_a)
    drop_app_index(db, app_a)
    db.commit()
    retriever = KeywordRetriever(db, ranking="pg_textsearch")
    assert not retriever.has_own_index(app_a)
    assert {r.filename for r in retriever.retrieve(QUERY, app_a, top_k=5)} == set(DOCS_A)


def test_index_options_copied_from_shared_index(db: Session) -> None:
    options = read_index_options(db)
    assert set(options) == {"text_config", "k1", "b"}
    with pytest.raises(ValueError):
        create_app_index(db, "x", {"text_config": "english'; DROP TABLE chunks; --", "k1": "1.2", "b": "0.75"})
