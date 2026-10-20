"""
MỤC ĐÍCH: Phase 8 0 kiểm tra CÁCH LY dữ liệu giữa các application, với database thật.

Bối cảnh test: 2 application "alpha" và "beta" có tài liệu GIỐNG HỆT NHAU về cấu trúc,
chỉ khác 1 "mật mã" (ALPHA-111 / BETA-222). Nếu dữ liệu rò rỉ giữa 2 app, mật mã của app kia 
sẽ xuất hiện trong kết quả -> test bắt được ngay.

Kiểm tra 4 tầng:
    1. Database: ràng buộc 2 cột (migration 0003) chặn chunk / log "lệch aplication"
    2. Repository: đọc/xóa comment, phiên chat, tin nhắn bằng id của app khác -> không thấy gì/
    3. Retrieval + RAG end-to-end: hỏi cùng 1 câu ở 2 app -> mỗi app chỉ trả nguồn của chính nó.
    4. pgvector: HNSW + WHERE application_id vẫn đủ kết quả nhờ hnsw.iterative_scan
"""


import inspect
import random
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest 
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ApplicationNotFoundError
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application, Chunk, Document, RetrievalLog
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.fake import FakeLLM
from app.rag.reranking.fake import FakeReranker
from app.rag.retrieval import VectorRetriever, build_retriever
from app.repositories import application_repository as app_repo
from app.repositories import chat_repository as chat_repo
from app.repositories import document_repository as doc_repo
from app.services.document_service import DocumentService
from app.services.rag_service import build_rag_service


EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
SETTINGS = Settings(chunk_size=500, chunk_overlap=50, retrieval_candidate_k=10,
                    reranker_provider="fake", llm_provider="fake", grounding_check=False)

SHARED = "Company handbook: all employees use Docker Compose for local development"


def _secret_doc(code: str) -> str:
    return f"Project secret code is {code}. The secret code protects the deployment pipeline"


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def apps(db: Session, tmp_path: Path) -> Iterator[tuple[str, str]]:
    alpha, beta = f"t_alpha_{uuid.uuid4().hex[:6]}", f"t_beta_{uuid.uuid4().hex[:6]}"
    db.add_all([Application(id=alpha, name="Alpha"), Application(id=beta, name="Beta")])
    db.commit()
    try:
        service = DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / "storage", embedder=EMBEDDER)
        for app_id, code in ((alpha, "ALPHA-111"), (beta, "BETA-222")):
            for name, content in (("secret.txt", _secret_doc(code)), ("handbook.txt", SHARED)):
                f = tmp_path / f"{app_id}_{name}"
                f.write_text(content, encoding="utf-8")
                service.ingest_file(app_id, f, filename=name)
        yield alpha, beta
    finally:
        # try/finally: kể cả ki nạp lỗi giữa chừng cũng dọn sạch, không để rác trong DB
        db.rollback()
        db.execute(delete(Application).where(Application.id.in_([alpha, beta])))
        db.commit()



def _doc(db: Session, application_id: str, filename: str) -> Document:
    return next(d for d in doc_repo.list_documents(db, application_id) if d.filename == filename)


# ------- 1. Database: ràng buộc 2 cột -----
def test_same_file_can_be_ingested_into_two_application(db: Session, apps: tuple[str, str]) -> None:
    """
    handbook.txt giống hệt nhau (cùng SHA-256) nhưng ở 2 app khác nhau -> hợp lệ
    Chống trùng (uq_document_app_hash) chỉ áp dụng TRONG 1 application.
    """
    alpha, beta = apps
    assert _doc(db, alpha, "handbook.txt").file_hash == _doc(db, beta, "handbook.txt").file_hash


def test_chunk_cannot_belong_document_of_other_application(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    doc_alpha = _doc(db, alpha, "secret.txt")
    db.add(Chunk(document_id=doc_alpha.id, application_id=beta, chunk_index=99, text="lạc app"))
    with pytest.raises(IntegrityError, match="fk_chunks_document_same_app"):
        db.flush()
    db.rollback()


def test_retrieval_log_must_match_its_chat_session(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    session_alpha = chat_repo.create_session(db, alpha)
    with pytest.raises(IntegrityError, match="fk_retrieval_logs_session_same_app"):
        chat_repo.add_retrieval_log(db, application_id=beta, query="q", chunk_ids=[], scores=[],
                                    session_id=session_alpha.id)
    db.rollback()

def test_retrieval_log_without_session_is_allowed(db: Session, apps: tuple[str, str]) -> None:
    """Scripts (ask.py, eval) không có phiên chat -> session_id NULL vẫn ghi log được"""
    alpha, _ = apps
    log = chat_repo.add_retrieval_log(db, application_id=alpha, query="q", chunk_ids=["x"], scores=[0.5])
    assert db.get(RetrievalLog, log.id).application_id == alpha
    db.rollback()



# ---------- 2. Repository: id của app khác -> không thấy gì ----------
def test_list_documents_only_returns_own_application(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    docs = doc_repo.list_documents(db, alpha)
    assert sorted(d.filename for d in docs) == ["handbook.txt", "secret.txt"]
    assert {d.application_id for d in docs} == {alpha}

def test_get_document_with_wrong_application_returns_none(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    doc_alpha = _doc(db, alpha, "secret.txt")
    assert doc_repo.get_document(db, alpha, doc_alpha.id) is not None
    assert doc_repo.get_document(db, beta, doc_alpha.id) is None

def test_delete_document_with_wrong_application_deletes_nothing(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    doc_alpha = _doc(db, alpha, "secret.txt")
    assert doc_repo.delete_document(db, beta, doc_alpha.id) is False
    assert doc_repo.get_document(db, alpha, doc_alpha.id) is not None

    assert doc_repo.delete_document(db, alpha, doc_alpha.id) is True
    db.commit()

    # chunks xoá theo nhờ ON DELETE CASCADE của khóa ngoại 2 cột
    assert db.scalars(select(Chunk).where(Chunk.document_id == doc_alpha.id)).first() is None


def test_chat_session_and_messages_are_scoped(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    session = chat_repo.create_session(db, alpha)
    chat_repo.add_message(db, session, "user", "Mật mã là gì?")
    chat_repo.add_message(db, session, "assistant", "ALPHA-111 [Source 1]")
    db.commit()

    assert chat_repo.get_session(db, alpha, session.id) is not None
    assert chat_repo.get_session(db, beta, session.id) is None              # đoán được id cũng không đọc được
    assert [m.role for m in chat_repo.list_messages(db, alpha, session.id)] == ["user", "assistant"]
    assert chat_repo.list_messages(db, beta, session.id) == []

def test_require_application(db: Session, apps: tuple[str, str]) -> None:
    alpha, _ = apps
    assert app_repo.require(db, alpha).id == alpha
    with pytest.raises(ApplicationNotFoundError):
        app_repo.require(db, "khong_ton_tai")


@pytest.mark.parametrize("module", [doc_repo, chat_repo])
def test_every_read_function_takes_application_id(module) -> None:
    """
    Quy ước của dự án: hàm đọc/xóa nào trong repository cũng phải nhận application_id.
    Ngoại lệ hợp lệ: hàm GHI nhận object đã được kiểm tra (add_chunks, add_message, touch_session).
    """
    allowed_without = {"add_chunks", "add_message", "touch_session"}
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__ or name.startswith("_") or name in allowed_without:
            continue
        assert "application_id" in inspect.signature(fn).parameters, f"{module.__name__}.{name}"


# ---------- 3. Retrieval + RAG end-to-end ----------
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_retrieval_returns_only_own_secret(db: Session, apps: tuple[str, str], mode: str) -> None:
    alpha, beta = apps
    retriever = build_retriever(db, SETTINGS, embedder=EMBEDDER, mode=mode)
    for app_id, own, other in ((alpha, "ALPHA-111", "BETA-222"), (beta, "BETA-222", "ALPHA-111")):
        results = retriever.retrieve("secret code BETA-222 ALPHA-111", app_id, top_k=10)
        texts = " ".join(r.text for r in results)
        assert own in texts
        assert other not in texts

def test_same_question_in_two_applications_cites_only_own_documents(db: Session, apps: tuple[str, str]) -> None:
    alpha, beta = apps
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(), llm=FakeLLM())
    own_doc_ids = {app_id: {d.id for d in doc_repo.list_documents(db, app_id)} for app_id in apps}

    for app_id in apps:
        result = service.ask("What is the project secret code?", app_id)
        assert not result.abstained
        assert result.sources, "FakeLLM trích [Source 1] nên phải có ít nhất 1 nguồn"
        assert {s.document_id for s in result.sources} <= own_doc_ids[app_id]
        # context gửi cho LLM cũng không được chứa dữ liệu app kia
        other = "BETA-222" if app_id == alpha else "ALPHA-111"
        assert other not in service.llm.calls[-1]["prompt"]

def test_unknown_application_raises_before_calling_any_model(db: Session) -> None:
    llm = FakeLLM()
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(), llm=llm)
    with pytest.raises(ApplicationNotFoundError):
        service.ask("What is the secret code?", "khong_ton_tai")
    assert llm.calls == []


# ---------- 4. pgvector: HNSW + WHERE application_id ----------
def test_pgvector_supports_iterative_scan(db: Session) -> None:
    """Cần pgvector >= 0.8. Fail ở đây -> xem mục Common Errors của guide Phase 8."""
    version = db.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar_one()
    assert tuple(int(x) for x in version.split(".")[:2]) >= (0, 8), f"pgvector {version} < 0.8"
    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    db.rollback()

def test_vector_retriever_turns_on_iterative_scan(db: Session, apps: tuple[str, str]) -> None:
    alpha, _ = apps
    VectorRetriever(db, EMBEDDER).retrieve("secret code", alpha, top_k=3)
    # set_config(..., true) có hiệu lực tới hết transaction -> đọc lại ngay trong transaction đó
    assert db.execute(text("SHOW hnsw.iterative_scan")).scalar_one() == "strict_order"
    db.rollback()

def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


@pytest.fixture
def hnsw_demo(db: Session) -> Iterator[str]:
    """Bảng TẠM (TEMP) giống chunks thu nhỏ: app nhỏ 3 dòng (vector ngẫu nhiên, xa câu hỏi) +
    app lớn 300 dòng (đều RẤT gần câu hỏi), có index HNSW riêng.
    Dùng bảng tạm thay vì bảng chunks thật: mỗi lần test có index mới tinh, và không làm
    "bẩn" index HNSW của dữ liệu thật (thêm/xoá hàng nghìn dòng liên tục làm đồ thị HNSW xấu đi)."""
    rng = random.Random(42)
    db.execute(text(f"CREATE TEMP TABLE hnsw_demo (app text, embedding vector({EMBEDDING_DIM}))"))
    rows = [{"app": "small", "v": str(_unit([rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]))} for _ in range(3)]
    rows += [{"app": "big", "v": str(_unit([5.0] + [rng.gauss(0, 0.3) for _ in range(EMBEDDING_DIM - 1)]))}
             for _ in range(300)]
    db.execute(text("INSERT INTO hnsw_demo VALUES (:app, CAST(:v AS vector))"), rows)
    db.execute(text("CREATE INDEX ON hnsw_demo USING hnsw (embedding vector_cosine_ops)"))
    db.execute(text("SET LOCAL enable_seqscan = off"))   # bảng nhỏ: ép planner dùng index như khi bảng lớn
    yield str([1.0] + [0.0] * (EMBEDDING_DIM - 1))   # câu hỏi: trục x
    db.rollback()   # bảng tạm tạo trong transaction -> rollback là mất


def _search_small_app(db: Session, query: str, iterative_scan: str) -> int:
    db.execute(text("SELECT set_config('hnsw.iterative_scan', :mode, true)"), {"mode": iterative_scan})
    return len(db.execute(text(
        "SELECT 1 FROM hnsw_demo WHERE app = 'small' "
        "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 3"), {"q": query}).all())

def test_hnsw_filter_without_iterative_scan_loses_results(db: Session, hnsw_demo: str) -> None:
    """Chứng minh vấn đề: 40 ứng viên đầu của HNSW đều thuộc app lớn -> lọc xong app nhỏ còn < 3."""
    assert _search_small_app(db, hnsw_demo, "off") < 3

def test_hnsw_filter_with_iterative_scan_keeps_results(db: Session, hnsw_demo: str) -> None:
    assert _search_small_app(db, hnsw_demo, "strict_order") == 3