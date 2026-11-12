"""
MỤC ĐÍCH: Integration test cho KEYWORD_RANKING=pg_textsearch (BM25 của extension pg_textsearch).

Kiểm tra:
  - từ hiếm thắng từ phổ biến (IDF), điểm trả về DƯƠNG, giảm dần;
  - chỉ trả chunk của application được hỏi (app khác không lọt vào);
  - không khớp / toàn stop-word -> [];
  - factory truyền đúng KEYWORD_RANKING=pg_textsearch.

Database chưa có index ix_chunks_bm25 (Postgres 16, chưa có extension) -> các test dùng DB tự BỎ QUA.
"""
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
from app.rag.retrieval import KeywordRetriever, build_retriever
from app.rag.retrieval.keyword_retriever import PGTS_INDEX
from app.services.document_service import DocumentService

EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
SETTINGS = Settings(chunk_size=500, chunk_overlap=50)

# "model" có trong cả 4 file (phổ biến), "booktest" chỉ có trong 1 file (hiếm)
DOCS_A = {
    "common.txt": "The model model model is a strong model for every model task.",
    "rare.txt": "We release BookTest, a new dataset, and train a model on it.",
    "other1.txt": "Another model trained on news articles.",
    "other2.txt": "A baseline model with logistic regression.",
}
DOCS_B = {"leak.txt": "BookTest BookTest BookTest secret copy in another application."}


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
    app_a, app_b = f"t_pg_a_{uuid.uuid4().hex[:6]}", f"t_pg_b_{uuid.uuid4().hex[:6]}"
    db.add_all([Application(id=app_a, name="A"), Application(id=app_b, name="B")])
    db.commit()
    _ingest(db, tmp_path, app_a, DOCS_A)
    _ingest(db, tmp_path, app_b, DOCS_B)
    yield app_a, app_b
    db.rollback()
    db.execute(delete(Application).where(Application.id.in_([app_a, app_b])))
    db.commit()


def test_pg_textsearch_rare_term_beats_common_term(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    results = KeywordRetriever(db, ranking="pg_textsearch").retrieve("model BookTest", app_a, top_k=5)
    assert results[0].filename == "rare.txt"            # "booktest" hiếm -> idf cao
    assert len(results) == 4                            # cả 4 file đều có "model" (truy vấn OR)
    assert results[0].score > results[1].score > 0      # đã đổi dấu: điểm dương, giảm dần


def test_pg_textsearch_only_returns_own_application(db: Session, apps: tuple[str, str]) -> None:
    app_a, app_b = apps
    in_a = KeywordRetriever(db, ranking="pg_textsearch").retrieve("BookTest secret", app_a, top_k=10)
    assert "leak.txt" not in {r.filename for r in in_a}
    in_b = KeywordRetriever(db, ranking="pg_textsearch").retrieve("BookTest secret", app_b, top_k=10)
    assert [r.filename for r in in_b] == ["leak.txt"]


def test_pg_textsearch_no_match_or_only_stopwords(db: Session, apps: tuple[str, str]) -> None:
    app_a, _ = apps
    assert KeywordRetriever(db, ranking="pg_textsearch").retrieve("blockchain", app_a, top_k=5) == []
    assert KeywordRetriever(db, ranking="pg_textsearch").retrieve("what is it?", app_a, top_k=5) == []


def test_factory_passes_pg_textsearch_ranking() -> None:
    hybrid = build_retriever(None, Settings(keyword_ranking="pg_textsearch"), embedder=EMBEDDER, mode="hybrid")
    assert hybrid.keyword.ranking == "pg_textsearch"
