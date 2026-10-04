"""
Model Document dùng để quản ly tài liệu mà người dùng 
upload lên hệ thống RAG. Mỗi document thuộc một application,
và được xử lý theo pipeline: upload -> lưu file -> chunking -> embedding -> lưu chunks -> dùng cho retrieval

Vai trò chinh:
1. Lưu thông tin file (tên, loại, đường dẫn, hash)
2. Theo dõi trạng thái xử lý (PENDING -> PROCESSING -> READY/FAILED)
3. Liên kết với application (mỗi app có kho tài liệu riêng)
4. Liên kết với cac chunk (dữ liệu dùng cho RAG)
5. Đảm bảo không update trùng tài liệu (Unique constraint)
"""

import enum
import uuid
from sqlalchemy import Enum, ForeignKey, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base, TimestampMixin

# Enum DocumentStatus định nghĩa các trạng thái của document trong quá trình xử lý.
class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"       # mới upload, chưa xử lý
    PROCESSING = "PROCESSING" # đang xử lý (chunking + embedding)
    READY = "READY"           # đã xử lý xong, sẵn sàng cho retrieval
    FAILED = "FAILED"         # xử lý thất bại


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("application_id", "file_hash", name="uq_document_file_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(20))
    storage_path: Mapped[str] = mapped_column(String(500))  # đường dẫn lưu file trên server
    file_hash: Mapped[str] = mapped_column(String(64))      # hash của file, dùng để tránh upload trùng, SHA-256 hash có độ dài 64 ký tự
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    
    error_message: Mapped[str | None] = mapped_column(Text)  # lưu thông tin lỗi nếu status = FAILED

    chunks: Mapped[list["Chunk"]] = relationship(  # noqa: F821
        back_populates="document", cascade="all, delete-orphan"
    )