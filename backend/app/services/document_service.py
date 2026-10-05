"""
MỤC ĐÍCH: Nghiệp vụ "nạp một tài liệu vào knowledge base" (document ingestion)
Đây là nơi điều phối toàn mộ ở mục 6 của spec:

    Validate file -> SHA-256 -> Check duplicate -> Lưu file gốc
    -> Trích text + Chunk (pipeline) -> Gắn metadata -> Lưu chunks -> READY


Nếu bất kỳ bước xử lý nào lỗi: document chuyển sang FAILED kèm error_message,
để người dùng biết file nào hỏng và vì sao

Phase 2 chạy đồng bộ 
Phase 3 thêm embedding vào đây
Phase 12 chuyển phần xử lý sang background job (lúc đó mới dùng PENDING)
"""

import hashlib
import logging
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ApplicationNotFoundError, DuplicateDocumentError, EmptyDocumentError,
    FileTooLargeError, UnsupportedFileTypeError,
)

from app.ingestion.loaders import SUPPORTED_FILE_TYPES
from app.ingestion.pipeline import parse_document
from app.ingestion.tokenizer import TokenCounter, get_token_counter
from app.models import Chunk, Document, DocumentStatus
from app.repositories import application_repository as app_repo
from app.repositories import document_repository as doc_repo

logger = logging.getLogger(__name__)


# Tính sha256 của file
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()

    # Mở file để đọc dữ liệu nhị phân
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):        # đọc từng 1MB, không load toàn bộ file vào RAM
            h.update(block)
    return h.hexdigest()

class DocumentService:
    def __init__(self, db: Session, settings: Settings | None = None, storage_dir: Path | None = None):
        self.db = db
        self.settings = settings or get_settings()
        self.storage_dir = storage_dir or Path(self.settings.storage_dir)

    @property
    def token_counter(self) -> TokenCounter:
        return get_token_counter(self.settings.tokenizer_encoding)

    def ingest_file(self, application_id: str, source_path: Path, filename: str | None = None) -> Document:
        filename = filename or source_path.name

        # 1. Application phải tồn tại
        if app_repo.get(self.db, application_id) is None:
            raise ApplicationNotFoundError(f"Application '{application_id}' not found")

        # 2. Validate định dạng + dung lượng
        file_type = source_path.suffix.lower().lstrip(".")
        if file_type not in SUPPORTED_FILE_TYPES:
            raise UnsupportedFileTypeError(f"'{filename}': unsupported type '.{file_type}'")
        size_mb = source_path.stat().st_size / (1024 * 1024)
        if size_mb > self.settings.max_upload_mb:
            raise FileTooLargeError(f"'{filename}' is {size_mb:.1f}MB (max {self.settings.max_upload_mb}MB)")
        


        # 3. Hash + kiểm tra trùng (trong cùng application)
        file_hash = sha256_of_file(source_path)
        existing = doc_repo.get_by_hash(self.db, application_id, file_hash)
        if existing is not None:
            if existing.status != DocumentStatus.FAILED:
                raise DuplicateDocumentError(f"'{filename}' already exists (document {existing.id})")       # dừng ngay chứ không chạy câu lệnh sau raise
            self.db.delete(existing)        # cho phép lạp lại file
            self.db.flush()

        # 4. Lưu file gốc: storge/<application_id>/<hash>/<ext>
        # Đặt theo hash -> không bao giờ đè nhầm 2 file trùng tên khác nội dung
        dest = self.storage_dir / application_id / f"{file_hash}.{file_type}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, dest)

        # 5. Tạo bản ghi document ở trạng thái PROCESSING và commit ngay,
        # để nếu bước sau lỗi thì vẫn còn dấu vết trong DB
        doc = doc_repo.create(
            self.db, application_id=application_id, filename=filename, file_type=file_type,
            storage_path=str(dest), file_hash=file_hash, status=DocumentStatus.PROCESSING
        )
        self.db.commit()
        logger.info("Ingesting document_id=%s filename=%s app=%s", doc.id, filename, application_id)

        try:
            # 6. Loader -> Cleaner -> Chunker
            pieces = parse_document(dest, file_type, self.settings.chunk_size,
                        self.settings.chunk_overlap, self.token_counter)
            if not pieces:
                raise EmptyDocumentError(f"No text extracted from '{filename}' (scanned/image-only file?)")

            # 7. Metadata enrichment + lưu chunks (embedding = None, Phase 3 mới xử lý)
            chunks = [
                Chunk(
                    document_id = doc.id,
                    application_id = application_id,
                    chunk_index = p.chunk_index,
                    text = p.text,
                    page = p.page,
                    metadata_ = {
                        "document_id": str(doc.id),
                        "application_id": application_id,
                        "source": filename,
                        "file_type": file_type,
                        "page": p.page,
                        "chunk_index": p.chunk_index,
                        "char_count": len(p.text),
                        "token_count": p.token_count,
                    }
                )
                for p in pieces
            ]
            doc_repo.add_chunks(self.db, chunks)

            # 8. Hoàn tất
            doc.status = DocumentStatus.READY
            self.db.commit()
            logger.info("Document %s READY with %d chunks", doc.id, len(chunks))
            return doc

        except Exception as exc:
            self.db.rollback()
            failed = self.db.get(Document, doc.id)
            if failed is not None:
                failed.status = DocumentStatus.FAILED
                failed.error_message = str(exc)[:1000]
                self.db.commit()
            logger.exception("Document %s FAILED", doc.id)      # logger.exception cho biết xem lỗi ở đâu 
            raise