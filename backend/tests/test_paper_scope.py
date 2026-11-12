"""
MỤC ĐÍCH: Test chế độ paper: chỉ tìm trong các document được chọn (document_ids).

Kiểm tra:
  - vector, keyword (ts_rank / bm25 / pg_textsearch), hybrid, pipeline: chỉ trả chunk của document đã chọn;
  - document của application KHÁC -> rỗng (lọc document không mở cửa sang app khác);
  - không truyền document_ids -> giữ nguyên hành vi cũ (tìm cả app);
  - eval: plan_scope bỏ tiền tố tên paper, bỏ qua câu holdout, evaluate_retrieval/generation dùng nó.
"""
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.evaluation.dataset import EvalCase
from app.evaluation.generation_eval import evaluate_generation
from app.evaluation.ground_truth import build_ground_truth
from app.evaluation.retrieval_eval import evaluate_retrieval
from app.evaluation.scope import ScopedQuestion, plan_scope
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.fake import FakeLLM
from app.rag.reranking.fake import FakeReranker
from app.rag.retrieval import HybridRetriever, KeywordRetriever, VectorRetriever
from app.rag.retrieval.keyword_retriever import PGTS_INDEX
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.services.document_service import DocumentService
from app.services.rag_service import build_rag_service

EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
SETTINGS = Settings(_env_file=None, chunk_size=500, chunk_overlap=50, tokenizer_encoding="whitespace")
QUERY = "BookTest dataset examples"
# 2 paper gần như giống hệt nhau: không lọc thì câu hỏi khớp cả 2
DOCS = {
    "p1.txt": "We evaluate the model on the BookTest dataset with 10k examples.",
    "p2.txt": "We evaluate the model on the BookTest dataset with 50k examples.",
}


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


def _make_app(db: Session, tmp_path: Path) -> tuple[str, dict[str, uuid.UUID]]:
    app_id = f"t_scope_{uuid.uuid4().hex[:8]}"
    db.add(Application(id=app_id, name="scope test"))
    db.commit()
    service = DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / app_id, embedder=EMBEDDER)
    ids = {}
    for name, content in DOCS.items():
        f = tmp_path / f"{app_id}-{name}"
        f.write_text(content, encoding="utf-8")
        ids[name] = service.ingest_file(app_id, f, filename=name).id
    return app_id, ids


@pytest.fixture
def papers(db: Session, tmp_path: Path) -> Iterator[tuple[str, dict[str, uuid.UUID], str, dict[str, uuid.UUID]]]:
    """2 application, mỗi app có p1 và p2 -> (app, ids, app khác, ids của app khác)."""
    app_a, ids_a = _make_app(db, tmp_path)
    app_b, ids_b = _make_app(db, tmp_path)
    yield app_a, ids_a, app_b, ids_b
    db.rollback()
    db.execute(delete(Application).where(Application.id.in_([app_a, app_b])))
    db.commit()


def _has_pgts(db: Session) -> bool:
    return db.execute(text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": PGTS_INDEX}).first() is not None


def _retrievers(db: Session) -> dict[str, object]:
    out = {
        "vector": VectorRetriever(db, EMBEDDER),
        "ts_rank": KeywordRetriever(db, ranking="ts_rank"),
        "bm25": KeywordRetriever(db, ranking="bm25"),
    }
    if _has_pgts(db):   # Postgres 16 / chưa có pg_textsearch: bỏ nhánh này, các nhánh khác vẫn test
        out["pg_textsearch"] = KeywordRetriever(db, ranking="pg_textsearch")
    out["hybrid"] = HybridRetriever(VectorRetriever(db, EMBEDDER), KeywordRetriever(db, ranking="bm25"))
    return out


# ---------------- retriever ----------------

def test_without_filter_both_papers_are_found(db: Session, papers) -> None:
    app_a, ids, _, _ = papers
    for name, r in _retrievers(db).items():
        found = {c.document_id for c in r.retrieve(QUERY, app_a, 5)}
        assert found == set(ids.values()), name   # hành vi cũ giữ nguyên


def test_filter_returns_only_chosen_paper(db: Session, papers) -> None:
    app_a, ids, _, _ = papers
    for name, r in _retrievers(db).items():
        results = r.retrieve(QUERY, app_a, 5, document_ids=[ids["p2.txt"]])
        assert results, name
        assert {c.document_id for c in results} == {ids["p2.txt"]}, name


def test_document_of_other_application_returns_nothing(db: Session, papers) -> None:
    app_a, _, _, ids_b = papers
    for name, r in _retrievers(db).items():
        # id có thật nhưng thuộc app B: lọc theo app vẫn chạy trước -> không lộ dữ liệu app khác
        assert r.retrieve(QUERY, app_a, 5, document_ids=[ids_b["p1.txt"]]) == [], name


def test_pipeline_passes_filter_to_retriever(db: Session, papers) -> None:
    app_a, ids, _, _ = papers
    pipeline = RetrievalPipeline(VectorRetriever(db, EMBEDDER), FakeReranker(), retrieval_top_k=5,
                                 rerank_top_k=5, min_score=0.0)
    result = pipeline.run(QUERY, app_a, document_ids=[ids["p1.txt"]])
    assert {r.chunk.document_id for r in result.reranked} == {ids["p1.txt"]}


# ---------------- eval ----------------

CASES = [
    EvalCase(id="q1", question='In the paper "Paper Two": How many examples?', answerable=True,
             expected_texts=["50k examples"], expected_documents=["p2.txt"]),
    EvalCase(id="q2", question="How many examples does p1 have?", answerable=True,
             expected_texts=["10k examples"], expected_documents=["p1.txt"]),
    EvalCase(id="h1", question='In the paper "Not here": What GPU?', answerable=False),   # holdout: không có paper
    EvalCase(id="t1", question='In the paper "P2": What GPU?', answerable=False),         # bẫy thật: paper p2 có
    EvalCase(id="x1", question="Typo file?", answerable=True, expected_texts=["model"],
             expected_documents=["missing.txt"]),                                            # tên file không có
]


def test_plan_scope_strips_title_and_skips_holdout(db: Session, papers) -> None:
    app_a, ids, _, _ = papers
    plan = plan_scope(db, app_a, CASES, "paper")
    assert plan.skipped == ["h1", "x1"]
    assert plan.questions["q1"].question == "How many examples?"          # bỏ tiền tố tên paper
    assert plan.questions["q1"].document_ids == [ids["p2.txt"]]
    assert plan.questions["q2"].question == CASES[1].question               # không có tiền tố -> giữ nguyên
    assert [c.id for c in plan.keep(CASES)] == ["q1", "q2", "t1"]


def test_real_trap_finds_its_paper_by_title(db: Session, papers) -> None:
    app_a, ids, _, _ = papers
    plan = plan_scope(db, app_a, CASES, "paper")
    assert plan.by_title == ["t1"]                                          # không có expected_documents
    assert plan.questions["t1"] == ScopedQuestion("What GPU?", [ids["p2.txt"]])


def test_plan_scope_all_keeps_everything_unchanged(db: Session, papers) -> None:
    app_a, _, _, _ = papers
    plan = plan_scope(db, app_a, CASES, "all")
    assert plan.skipped == [] and len(plan.questions) == len(CASES)
    assert all(q.document_ids is None and q.kwargs() == {} for q in plan.questions.values())
    with pytest.raises(ValueError):
        plan_scope(db, app_a, CASES, "corpus")


def test_plan_scope_never_maps_to_other_application(db: Session, papers) -> None:
    app_a, ids_a, _, ids_b = papers
    plan = plan_scope(db, app_a, CASES, "paper")
    assert set(plan.questions["q1"].document_ids) <= set(ids_a.values())
    assert not set(plan.questions["q1"].document_ids) & set(ids_b.values())


def test_retrieval_eval_paper_scope(db: Session, papers) -> None:
    app_a, _, _, _ = papers
    report = evaluate_retrieval(db, SETTINGS, app_a, CASES, embedder=EMBEDDER, reranker=FakeReranker(),
                                ks=[1], rerank_sources=("vector",), scope="paper")
    assert report["scope"] == {"mode": "paper", "skipped": ["h1", "x1"], "by_title": ["t1"]}
    assert [c["id"] for c in report["cases"]] == ["q1", "q2"]
    # chỉ còn đúng 1 chunk (paper của câu hỏi) -> mọi cách tìm đều xếp chunk đúng ở hạng 1
    for mode, m in report["modes"].items():
        assert m["metrics"]["hit@1"] == 1.0, mode


def test_generation_eval_paper_scope(db: Session, papers) -> None:
    app_a, _, _, _ = papers
    llm = FakeLLM("50k examples [Source 1].")
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(), llm=llm, min_score=0.0)
    cases = CASES[:1] + CASES[2:3]
    plan = plan_scope(db, app_a, cases, "paper")
    report = evaluate_generation(service, app_a, cases, build_ground_truth(db, app_a, cases), plan=plan)
    assert report["scope"]["skipped"] == ["h1"] and report["counts"]["cases"] == 1
    row = report["cases"][0]
    assert row["sources"] and all(s["filename"] == "p2.txt" for s in row["sources"])
    assert "Not here" not in llm.calls[0]["prompt"] and "Paper Two" not in llm.calls[0]["prompt"]


def test_filter_works_with_per_app_bm25_index(db: Session, papers) -> None:
    """Có index BM25 riêng của app (create_bm25_index.py --app): câu SQL khác, lọc vẫn phải đúng."""
    if not _has_pgts(db):
        pytest.skip("cần Postgres 17+ có pg_textsearch")
    from app.rag.retrieval.bm25_index import create_app_index, drop_app_index
    app_a, ids, _, _ = papers
    create_app_index(db, app_a)
    try:
        r = KeywordRetriever(db, ranking="pg_textsearch")
        assert {c.document_id for c in r.retrieve(QUERY, app_a, 5)} == set(ids.values())
        results = r.retrieve(QUERY, app_a, 5, document_ids=[ids["p1.txt"]])
        assert results and {c.document_id for c in results} == {ids["p1.txt"]}
    finally:
        drop_app_index(db, app_a)
