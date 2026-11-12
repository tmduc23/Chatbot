"""
MỤC ĐÍCH (Phase 12a): xử lý tài liệu NỀN, dùng chính bảng 'documents' của Postgres
làm hàng đợi.

    API upload -> lưu file + document PENDING -> trả 202 ngay
    Worker: lặp {nhận 1 document PENDING -> parse/chunk/embed -> READY | thử lại sau |FAILED }

Vì sao không cần Redis/Celery: đã có Postgres. 'SELECT ... FOR UPDATE SKIP LOCKED' cho phép nhiều worker cùng lấy việc
mà KHÔNG BAO GIỜ 2 worker nhận cùng 1 document: dòng đang bị worker A khóa thì worker B bỏ qua (SKIP) và lấy dòng kế tiếp, không phải đứng chờ.

Ba tình huống lỗi và cách xử lý:
  1. Lỗi TẠM THỜI (API embedding 429/503, mất kết nối DB): quay về PENDING, lùi available_at
     (30s, 60s, 120s...), tối đa WORKER_MAX_ATTEMPTS lần rồi mới FAILED.
  2. Lỗi VĨNH VIỄN (file rỗng, file hỏng, sai định dạng): FAILED ngay, thử lại cũng vô ích.
  3. Worker CHẾT giữa chừng (container bị kill): document kẹt ở PROCESSING. Sau
     WORKER_STALE_AFTER_SECONDS, worker khác coi như việc bị bỏ dở và nhận lại.

"Fencing" bằng cột attempts: khi ghi kết quả, chỉ cập nhật nếu attempts VẪN là số lần của mình nhận việc. 
Nếu worker chậm bị coi là chết và worker khác đã nhận lại (attempts tăng), kết quả của worker chậm bị bỏ, không ghi đè và không tạo chunk trùng.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.models import Document, DocumentStatus
from app.services.document_service import DocumentService


logger = logging.getLogger(__name__)

# Lỗi đáng để thử lại: do bên ngoài, thường tự hết sau do ít phút, Mọi lỗi khác = vĩnh viễn

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    EmbeddingError, OperationalError, TimeoutError, ConnectionError,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def retry_delay(attempt: int, base_s: int, max_s: int) -> timedelta:
    """
    Exponential backoff: lần 1 lỗi -> chờ base, lần 2 -> 2*base, lần 3 -> 4*base... tối đa max_s.
    """
    return timedelta(seconds=min(max_s, base_s * 2 ** (attempt - 1)))


def claim_next(
    db: Session, 
    settings: Settings,
    now: datetime | None = None,
    application_ids: list[str] | None = None,
) -> Document | None:
    """
    Nhận 1 document cần xử lý: chuyển sang PROCESSING, attempts + 1, commit. Không có việc -> None.

    'now' và 'application_ids' để test điều khiển thời gian và chỉ nhận document của chính test.
    """
    now = now or utcnow()
    stale_before = now - timedelta(seconds=settings.worker_stale_after_seconds)
    stmt = (
        select(Document)
        .where(or_(
            and_(Document.status == DocumentStatus.PENDING, Document.available_at <= now),
            # worker trước nhận việc rồi im lặng quá lâu -> coi như đã chất
            and_(Document.status == DocumentStatus.PROCESSING, Document.locked_at < stale_before),
        ))
        .order_by(Document.available_at)
        .limit(1)
        .with_for_update(skip_locked=True)      # dòng đang bị worker khác khóa -> bỏ qua, lấy dòng sau
    )
    if application_ids is not None:
        stmt = stmt.where(Document.application_id.in_(application_ids))

    doc = db.scalars(stmt).first()
    if doc is None:
        db.rollback()           # kết thúc transaction rỗng
        return None
    doc.status = DocumentStatus.PROCESSING
    doc.locked_at = now
    doc.attempts += 1
    db.commit()         # nhả khóa ngay: các worker khác thấy PROCESSING, không nhận nữa
    return doc


def _fenced_update(db: Session, doc_id: uuid.UUID, attempt: int, **values) -> bool:
    """UPDATE chỉ khi document vẫn là việc của MÌNH (PROCESSING + đúng attempts). True nếu ghi được."""
    result = db.execute(
        update(Document)
        .where(Document.id == doc_id, Document.status == DocumentStatus.PROCESSING,
               Document.attempts == attempt)
        .values(**values)
    )
    return result.rowcount == 1


def run_once(
    db: Session,
    service: DocumentService,
    settings: Settings,
    now: datetime | None = None,
    application_ids: list[str] | None = None,
) -> Document | None:
    """Xử lý tối đa 1 document. Trả về document đã nhận (hoặc None nếu hàng đợi rỗng)."""
    doc = claim_next(db, settings, now=now, application_ids=application_ids)
    if doc is None:
        return None

    # Lưu id ra biến: sau rollback, SQLAlchemy "quên" khi mọi thuộc tính của doc và phải đọc lại từ DB.
    # Nếu document vừa bị xóa thì đọc lại sẽ lỗi ObjectDeletedError.
    doc_id, attempt = doc.id, doc.attempts
    logger.info("worker picked document_id=%s app=%s attemp=%d", doc_id, doc.application_id, attempt)

    try:
        n_chunks = service.process_document(doc)            # chunk mới chỉ nằm trong transaction, chưa commit
        if not _fenced_update(db, doc_id, attempt, status=DocumentStatus.READY,
                              locked_at=None, error_message=None):
            db.rollback()       # việc đã bị worker khác nhận lại hoặc document đã bị xóa -> bỏ kết quả
            logger.warning("worker lost document_id=%s (deleted or re-claimed), discarding", doc_id)
            return doc
        db.commit()         # chunks + READY cùng 1 commit: không bao giờ có READY mà thiếu chunk
        logger.info("worker document_id=%s READY chunks=%d", doc_id, n_chunks)
    except Exception as exc:
        db.rollback()           # bỏ chunk làm dở
        _record_failure(db, doc_id, attempt, exc, settings, now or utcnow())
    return doc


def _record_failure(db: Session, doc_id: uuid.UUID, attempt: int, exc: Exception,
                    settings: Settings, now: datetime) -> None:
    transient = isinstance(exc, TRANSIENT_ERRORS)
    if transient and attempt < settings.worker_max_attempts:
        delay = retry_delay(attempt, settings.worker_retry_base_seconds, settings.worker_retry_max_seconds)
        values = {"status": DocumentStatus.PENDING, "available_at": now + delay, "locked_at": None,
                  "error_message": f"attempt {attempt} failed, retrying: {exc}"[:1000]}
        logger.warning("worker document_id=%s attempt %d failed (%s), retry in %ss",
                       doc_id, attempt, type(exc).__name__, int(delay.total_seconds()))
    else:
        values = {"status": DocumentStatus.FAILED, "locked_at": None, "error_message": str(exc)[:1000]}
        logger.error("worker document_id=%s FAILED after %d attempt(s): %s", doc_id, attempt, exc)

    try:
        # 0 dòng được ghi = document đã bị xóa / đã bị nhận lại -> không có gì để cập nhật
        _fenced_update(db, doc_id, attempt, **values)
        db.commit()
    except OperationalError:
        db.rollback()       # DB mất kết nối: để nguyên PROCESSING, hết hạn stale sẽ có người nhận lại
        logger.exception("worker could not record failure for document_id=%s", doc_id)


def run_forever(
    session_factory: Callable[[], Session],
    service_factory: Callable[[Session], DocumentService],
    settings: Settings,
    stop: threading.Event,
) -> int:
    """Vòng lặp của worker. Dừng khi 'stop' được set (SIGTERM/Ctrl+C). Trả về số document đã xử lý."""
    processed = 0
    while not stop.is_set():
        try:
            with session_factory() as db:
                doc = run_once(db, service_factory(db), settings)
        except OperationalError:
            logger.exception("worker: database unavailable, waiting")
            doc = None
        if doc is None:
            stop.wait(settings.worker_poll_seconds)         # hàng đợi rỗng: ngủ, nhưng thức dậy ngay khi bị dừng
        else: 
            processed += 1
    return processed


