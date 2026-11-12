"""
MỤC ĐÍCH: Phase 12a — test worker xử lý tài liệu nền, với database THẬT (cần SKIP LOCKED thật).

Mọi document của test được hẹn xử lý ở TƯƠNG LAI XA (FUTURE) và test gọi worker với now=FUTURE:
nếu container `worker` thật đang chạy, nó (dùng giờ hiện tại) sẽ không đụng vào dữ liệu của test.
"""
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import DuplicateDocumentError, EmbeddingError
from app.db.session import SessionLocal
from app.models import EMBEDDING_DIM, Application, Chunk, Document, DocumentStatus
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.services.document_service import DocumentService
from app.workers.document_worker import claim_next, retry_delay, run_forever, run_once

FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)
SETTINGS = Settings(chunk_size=500, chunk_overlap=50, tokenizer_encoding="whitespace",
                    worker_max_attempts=3, worker_retry_base_seconds=30, worker_retry_max_seconds=900,
                    worker_stale_after_seconds=900, worker_poll_seconds=0.01)


class FlakyEmbedder(FakeEmbeddingProvider):
    """Embedding giả lỗi `failures` lần đầu (giống API trả 429), sau đó chạy bình thường."""

    def __init__(self, failures: int):
        super().__init__(EMBEDDING_DIM)
        self.failures = failures
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.failures:
            raise EmbeddingError("429 rate limited")
        return super().embed_documents(texts)


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def app_id(db: Session) -> Iterator[str]:
    app_id = f"t_worker_{uuid.uuid4().hex[:6]}"
    db.add(Application(id=app_id, name="Worker test"))
    db.commit()
    try:
        yield app_id
    finally:
        db.rollback()
        db.execute(delete(Application).where(Application.id == app_id))
        db.commit()


def _service(db: Session, tmp_path: Path, embedder=None) -> DocumentService:
    return DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / "storage",
                           embedder=embedder or FakeEmbeddingProvider(EMBEDDING_DIM))


def _submit(service: DocumentService, tmp_path: Path, app_id: str, name: str, content: str) -> Document:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return service.submit_file(app_id, f, filename=name, available_at=FUTURE)


def _run(db: Session, service: DocumentService, app_id: str, now: datetime = FUTURE) -> Document | None:
    return run_once(db, service, SETTINGS, now=now, application_ids=[app_id])


def _chunk_count(db: Session, doc_id) -> int:
    return db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc_id))


def _reload(db: Session, doc_id) -> Document:
    db.expire_all()
    return db.get(Document, doc_id)


# ---------- submit: nhanh, chưa xử lý gì ----------

def test_submit_creates_pending_document_without_chunks(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker and FastAPI.")
    assert doc.status == DocumentStatus.PENDING and doc.attempts == 0
    assert Path(doc.storage_path).exists()            # file gốc đã lưu để worker đọc
    assert _chunk_count(db, doc.id) == 0


def test_same_file_while_pending_is_duplicate(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")
    with pytest.raises(DuplicateDocumentError):
        _submit(service, tmp_path, app_id, "cv-copy.txt", "Duc knows Docker.")


# ---------- worker: đường thành công ----------

def test_worker_processes_pending_document_to_ready(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker and FastAPI.")

    assert _run(db, service, app_id).id == doc.id
    doc = _reload(db, doc.id)
    assert doc.status == DocumentStatus.READY
    assert doc.attempts == 1 and doc.locked_at is None and doc.error_message is None
    assert _chunk_count(db, doc.id) >= 1
    assert _run(db, service, app_id) is None          # hàng đợi đã rỗng


def test_document_scheduled_later_is_not_picked_yet(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")
    assert _run(db, service, app_id, now=FUTURE - timedelta(seconds=1)) is None


# ---------- worker: lỗi tạm thời / vĩnh viễn ----------

def test_transient_error_is_retried_later_then_succeeds(db: Session, tmp_path: Path, app_id: str) -> None:
    embedder = FlakyEmbedder(failures=1)
    service = _service(db, tmp_path, embedder)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")

    _run(db, service, app_id)                         # lần 1: embedding báo 429
    failed_once = _reload(db, doc.id)
    assert failed_once.status == DocumentStatus.PENDING
    assert failed_once.available_at == FUTURE + timedelta(seconds=30)
    assert "429" in failed_once.error_message
    assert _chunk_count(db, doc.id) == 0              # chunk làm dở đã bị rollback

    assert _run(db, service, app_id, now=FUTURE + timedelta(seconds=10)) is None   # chưa tới giờ thử lại
    _run(db, service, app_id, now=FUTURE + timedelta(seconds=31))                 # lần 2: thành công
    done = _reload(db, doc.id)
    assert done.status == DocumentStatus.READY and done.attempts == 2 and done.error_message is None


def test_transient_error_fails_after_max_attempts(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path, FlakyEmbedder(failures=99))
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")
    now = FUTURE
    for _ in range(SETTINGS.worker_max_attempts):
        _run(db, service, app_id, now=now)
        now += timedelta(hours=1)
    doc = _reload(db, doc.id)
    assert doc.status == DocumentStatus.FAILED and doc.attempts == 3
    assert _run(db, service, app_id, now=now) is None  # FAILED không bao giờ được nhận lại


def test_permanent_error_fails_immediately(db: Session, tmp_path: Path, app_id: str) -> None:
    embedder = FlakyEmbedder(failures=0)
    service = _service(db, tmp_path, embedder)
    doc = _submit(service, tmp_path, app_id, "blank.txt", "   \n\n   ")   # không có chữ -> EmptyDocumentError
    _run(db, service, app_id)
    doc = _reload(db, doc.id)
    assert doc.status == DocumentStatus.FAILED and doc.attempts == 1
    assert "No text extracted" in doc.error_message
    assert embedder.calls == 0                        # không phí 1 lần gọi API nào


def test_failed_document_can_be_uploaded_again(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    _submit(service, tmp_path, app_id, "blank.txt", "   ")
    _run(db, service, app_id)
    again = _submit(service, tmp_path, app_id, "blank.txt", "   ")    # cùng hash, lần trước FAILED -> được
    assert again.status == DocumentStatus.PENDING


def test_retry_delay_doubles_and_is_capped() -> None:
    assert [retry_delay(n, 30, 900).total_seconds() for n in (1, 2, 3, 6)] == [30, 60, 120, 900]


# ---------- nhiều worker cùng lúc ----------

def test_locked_document_is_skipped_by_other_worker(db: Session, tmp_path: Path, app_id: str) -> None:
    """Worker A đang giữ khoá dòng doc1 -> worker B KHÔNG đứng chờ, lấy luôn doc2 (SKIP LOCKED)."""
    service = _service(db, tmp_path)
    doc1 = _submit(service, tmp_path, app_id, "a.txt", "Document A about Docker.")
    doc2 = _submit(service, tmp_path, app_id, "b.txt", "Document B about Python.")

    worker_a = SessionLocal()
    try:
        worker_a.execute(text("SELECT id FROM documents WHERE id = :id FOR UPDATE"), {"id": doc1.id})
        with SessionLocal() as worker_b:
            # quên skip_locked thì B sẽ ĐỨNG CHỜ A mãi -> giới hạn 2s để test báo lỗi thay vì treo
            worker_b.execute(text("SET LOCAL lock_timeout = '2s'"))
            claimed = claim_next(worker_b, SETTINGS, now=FUTURE, application_ids=[app_id])
            assert claimed is not None and claimed.id == doc2.id
    finally:
        worker_a.rollback()
        worker_a.close()


def test_stale_processing_document_is_reclaimed(db: Session, tmp_path: Path, app_id: str) -> None:
    service = _service(db, tmp_path)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")
    claim_next(db, SETTINGS, now=FUTURE, application_ids=[app_id])     # worker 1 nhận việc rồi "chết"

    soon = FUTURE + timedelta(seconds=SETTINGS.worker_stale_after_seconds - 1)
    assert claim_next(db, SETTINGS, now=soon, application_ids=[app_id]) is None   # còn sống, không giành
    later = FUTURE + timedelta(seconds=SETTINGS.worker_stale_after_seconds + 1)
    reclaimed = claim_next(db, SETTINGS, now=later, application_ids=[app_id])
    assert reclaimed is not None and reclaimed.id == doc.id and reclaimed.attempts == 2


def test_slow_worker_result_is_discarded_after_reclaim(db: Session, tmp_path: Path, app_id: str) -> None:
    """Fencing: worker chậm (attempt 1) xong việc SAU KHI worker khác đã nhận lại (attempt 2)."""
    service = _service(db, tmp_path)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")

    class SlowService(DocumentService):
        def process_document(self, d: Document) -> int:
            # trong lúc worker này đang embed, worker khác coi nó đã chết và nhận lại việc
            with SessionLocal() as other:
                other.execute(text("UPDATE documents SET attempts = attempts + 1 WHERE id = :id"), {"id": d.id})
                other.commit()
            return super().process_document(d)

    slow = SlowService(db, settings=SETTINGS, storage_dir=tmp_path / "storage",
                       embedder=FakeEmbeddingProvider(EMBEDDING_DIM))
    _run(db, slow, app_id)
    doc = _reload(db, doc.id)
    assert doc.status == DocumentStatus.PROCESSING and doc.attempts == 2   # vẫn là việc của worker 2
    assert _chunk_count(db, doc.id) == 0                                    # chunk của worker chậm bị bỏ


def test_document_deleted_while_processing_does_not_crash_worker(
    db: Session, tmp_path: Path, app_id: str,
) -> None:
    service = _service(db, tmp_path)
    doc = _submit(service, tmp_path, app_id, "cv.txt", "Duc knows Docker.")

    class DeletingService(DocumentService):
        def process_document(self, d: Document) -> int:
            with SessionLocal() as admin:     # admin xoá document đúng lúc worker đang xử lý
                admin.execute(delete(Document).where(Document.id == d.id))
                admin.commit()
            return super().process_document(d)

    deleting = DeletingService(db, settings=SETTINGS, storage_dir=tmp_path / "storage",
                               embedder=FakeEmbeddingProvider(EMBEDDING_DIM))
    doc_id = doc.id
    assert _run(db, deleting, app_id) is not None      # không ném lỗi ra ngoài
    assert _reload(db, doc_id) is None and _chunk_count(db, doc_id) == 0


def test_run_forever_stops_when_signalled() -> None:
    stop = threading.Event()
    stop.set()                                         # giống nhận SIGTERM trước vòng lặp đầu
    assert run_forever(SessionLocal, lambda db: None, SETTINGS, stop) == 0
