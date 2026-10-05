"""
Mục đích: integration test toàn bộ luồng với database thật, gồm 5 kịch bản:

nạp thành công;
file trùng;
file rỗng thì chuyển FAILED;
application không tồn tại;
sai định dạng.

MỤC ĐÍCH: Integration test cho toàn bộ luồng ingestion với database thật.

Mỗi test tạo 1 application riêng và XOÁ nó khi xong (documents + chunks bị xoá
theo nhờ ON DELETE CASCADE) -> không để lại rác trong DB.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    ApplicationNotFoundError, DuplicateDocumentError, EmptyDocumentError, UnsupportedFileTypeError,
)
from app.db.session import SessionLocal
from app.models import Application, DocumentStatus
from app.repositories import document_repository as doc_repo
from app.services.document_service import DocumentService, sha256_of_file

CV_TEXT = "\n\n".join(
    f"Project {i}: Built a RAG chatbot with FastAPI, PostgreSQL and pgvector. "
    f"Handled document ingestion, chunking and hybrid retrieval for use case {i}."
    for i in range(10)
)

# Tạo session database
@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()

# Tạo application tạm
@pytest.fixture
def app_id(db: Session) -> Iterator[str]:
    app_id = f"t_{uuid.uuid4().hex[:8]}"
    db.add(Application(id=app_id, name="test"))
    db.commit()
    yield app_id
    db.rollback()
    db.execute(delete(Application).where(Application.id == app_id))
    db.commit()

# Tạo application_service
@pytest.fixture
def service(db: Session, tmp_path: Path) -> DocumentService:
    settings = Settings(chunk_size=60, chunk_overlap=10)
    return DocumentService(db, settings=settings, storage_dir=tmp_path / "storage")

def _write(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f

# Luồng ingest
def test_ingest_txt_creates_ready_document_with_chunks(
        db: Session, service: DocumentService, app_id: str, tmp_path: Path
) -> None:
    doc = service.ingest_file(app_id, _write(tmp_path, "cv.txt", CV_TEXT))

    assert doc.status == DocumentStatus.READY
    assert Path(doc.storage_path).exists()          # file gốc đã được lưu
    chunks = doc_repo.list_chunks(db, app_id, doc.id)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    first = chunks[0]
    assert first.application_id == app_id
    assert 0 < first.metadata_["token_count"] <= 60
    assert first.embedding is None              # Phase 3 mới có embedding
    assert first.metadata_["source"] == "cv.txt"
    assert first.metadata_["document_id"] == str(doc.id)
    assert first.metadata_["application_id"] == app_id

# file trùng
def test_duplicate_file_is_rejected(service: DocumentService, app_id:str, tmp_path: Path) -> None:
    f = _write(tmp_path, "cv.txt", CV_TEXT)
    service.ingest_file(app_id, f)
    with pytest.raises(DuplicateDocumentError):
        service.ingest_file(app_id, f)

# File rỗng
def test_empty_file_is_marked_failed(
    db: Session, service: DocumentService, app_id: str, tmp_path: Path
) -> None:
    f = _write(tmp_path, "empty.txt", "   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        service.ingest_file(app_id, f)
    doc = doc_repo.get_by_hash(db, app_id, sha256_of_file(f))
    assert doc is not None and doc.status == DocumentStatus.FAILED
    assert doc.error_message

# Kiểm tra khi application không tồn tại
def test_unknown_application_is_rejected(service: DocumentService, tmp_path: Path) -> None:
    with pytest.raises(ApplicationNotFoundError):
        service.ingest_file("does_not_exist", _write(tmp_path, "a.txt", "hello"))

# Kiểm tra file không hỗ trợ định dạng
def test_unsupported_extension_is_rejected(service: DocumentService, app_id: str, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        service.ingest_file(app_id, _write(tmp_path, "photo.png", "not really an image"))