"""
MỤC ĐÍCH: Nghiệp vụ "nạp một tài liệu vào knowledge base" (document ingestion)
Đây là nơi điều phối toàn mộ ở mục 6 của spec:

# Điều phối toàn bộ luồng

    Validate file -> SHA-256 -> Check duplicate -> Lưu file gốc
    -> Trích text + Chunk (pipeline) -> Gắn metadata -> Lưu chunks -> READY


Nếu bất kỳ bước xử lý nào lỗi: document chuyển sang FAILED kèm error_message,
để người dùng biết file nào hỏng và vì sao

Phase 2 chạy đồng bộ 
Phase 3 thêm bước embedding (qua EmbeddingProvider - không phụ thuộc Gemni trực tiếp)
và hàm embedding_missing_chunk() để embed bổ sung các chunk nạp từ Phase 2
Phase 12 chuyển phần xử lý sang background job (lúc đó mới dùng PENDING)

"""

import hashlib
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO         
import time

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ApplicationNotFoundError, DocumentNotFoundError ,DuplicateDocumentError, EmptyDocumentError,
    FileTooLargeError, UnsupportedFileTypeError,
)

from app.ingestion.loaders import SUPPORTED_FILE_TYPES
from app.ingestion.pipeline import parse_document
from app.ingestion.tokenizer import TokenCounter, get_token_counter
from app.models import Chunk, Document, DocumentStatus
from app.repositories import application_repository as app_repo
from app.repositories import document_repository as doc_repo
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider



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
    def __init__(
            self, 
            db: Session, 
            settings: Settings | None = None, 
            storage_dir: Path | None = None,
            embedder: EmbeddingProvider | None = None):
        self.db = db
        self.settings = settings or get_settings()
        self.storage_dir = storage_dir or Path(self.settings.storage_dir)
        self._embedder = embedder

    # @property: cho phép gọi một method như là thuộc tính
    # tức dùng obj.name thay vì obj.name()
    @property
    def token_counter(self) -> TokenCounter:
        return get_token_counter(self.settings.tokenizer_encoding)

    @property
    def embedder(self) -> EmbeddingProvider:
        # Tạo muộn: các lỗi validate (file trùng, sai định dạng) không cần API key
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.settings)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        vectors = self.embedder.embed_documents(texts)
        logger.info("Embedded %d chunks in %.0f ms", len(texts), (time.perf_counter() - t0) * 1000)
        return vectors


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

            # 7 . Embedding (Phase 3)
            vectors = self._embed([p.text for p in pieces])
            
            # 8. Metadata enrichment + lưu chunks 
            chunks = [
                Chunk(
                    document_id = doc.id,
                    application_id = application_id,
                    chunk_index = p.chunk_index,
                    text = p.text,
                    page = p.page,
                    embedding = vector,
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
                for p, vector in zip(pieces, vectors, strict=True)      # strict = True, đảm bảo số vector đúng bằng số chunk. Lệch báo lỗi 
            ]
            doc_repo.add_chunks(self.db, chunks)

            # 9. Hoàn tất
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


    def ingest_upload(self, application_id: str, fileobj: BinaryIO, filename: str) -> Document:
        """
        Nạp file người dùng uplo (UploadFile.file của FastAPI).

        Ghi ra file tạm rồi dùng lại ingest_file() - toàn bộ kiểm tra cũ (định dạng, trùng, dung lượng)
        giữ nguyên, Đọc từng 1MB và DỪNG khi vượt MAX_UPLOAD_MB: không để ai gửi file 5GB làm đầy ổ đĩa

        "ingest_upload" không viết lại logic nạp tài liệu, nó chỉ ghi file upload ra file tạm rồi gọi "ingest_file()"

        "ingest_upload: chỉ lo phần "file đến từ Internet"
        """

        app_repo.require(self.db, application_id)
        # Chỉ lấy TÊN file: trình duyệt/kẻ xấu có thể gửi "../../etc/passwd" hoặc "C:\\x\\cv.pdf"
        safe_name = Path(filename.replace("\\", "/")).name or "upload"
        suffix = Path(safe_name).suffix.lower()

        if suffix.lstrip(".") not in SUPPORTED_FILE_TYPES:
            raise UnsupportedFileTypeError(f"'{safe_name}': unsupported type '{suffix or '(none)'}'")

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"upload{suffix}"
            written = 0
            with tmp_path.open("wb") as out:
                for block in iter(lambda: fileobj.read(1024 * 1024), b""):
                    written += len(block)
                    if written > max_bytes:
                        raise FileTooLargeError(f"'{safe_name}' is larger than {self.settings.max_upload_mb}MB")
                    out.write(block)
            return self.ingest_file(application_id, tmp_path, filename=safe_name)

    def list_documents(self, application_id: str) -> list[Document]:
        app_repo.require(self.db, application_id)
        return doc_repo.list_documents(self.db, application_id)

    def get_document(self, application_id: str, document_id: uuid.UUID) -> Document:
        doc = doc_repo.get_document(self.db, application_id, document_id)
        if doc is None: 
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return doc


    def count_chunks(self, application_id: str, document_id: uuid.UUID) -> int:
        return doc_repo.count_chunks(self.db, application_id, document_id)

    def delete_document(self, application_id: str, document_id: uuid.UUID) -> None:
        doc = self.get_document(application_id, document_id)
        storage_path = Path(doc.storage_path)
        doc_repo.delete_document(self.db, application_id, document_id)          # chunk xóa theo CASCADE
        self.db.commit()
        # Xóa file gốc SAU khi commit: nếu commit lỗi thì file vẫn còn, dữ liệu không lệch.
        # (Tên file = hash, duy nhất trong application -> không có document nào khác dùng chung file này.)
        storage_path.unlink(missing_ok=True)
        logger.info("Deleted document %s (app=%s)", document_id, application_id)

    def embed_missing_chunks(self, application_id: str) -> int:
        """Embed các chunk còn thiếu vector. Trả về số chunk đã embed"""
        chunks = doc_repo.list_chunks_without_embedding(self.db, application_id)
        if not chunks: 
            return 0
        vectors = self._embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        self.db.commit()
        return len(chunks)