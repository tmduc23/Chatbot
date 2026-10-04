import uuid
from typing import Any
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, String, Text, Uuid, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base import Base, CreatedAtMixin

# migration: tạo bảng chunks trong database, dùng để lưu thông tin các chunk của document
# Quyết định SCHEMA: gemini-embedding-001 768 chiều

EMBEDDING_DIM = 768

class Chunk(CreatedAtMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_embedding_hnsw", "embedding",
              postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
    )   # tạo index cho cột embedding, dùng để tìm kiếm nhanh hơn khi truy vấn vector

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)  # index của chunk trong document, bắt đầu từ 0
    text: Mapped[str] = mapped_column(Text)  # nội dung chunk
    page: Mapped[int | None] = mapped_column(Integer)  # số trang của chunk, nếu document là PDF

    # metadata_ bị SQLAlchemy giữ chỗ -> attribute là metatdata_, cột DB vẫn tên "metadata"
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)  # metadata của chunk, có thể là số trang, tên file, v.v.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))  # embedding vector của chunk, dùng để tìm kiếm tương tự

    # noqua:FB21
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")  # quan hệ với Document, mỗi chunk thuộc về một document