""" 
Index (Tên_index, cột_cần_index, cách_postgres_tạo_index)

HNSW index -> phục vụ Vector Search 
text -> Embedding Model -> embedding
semantic retrieval
ix_chunks_embedding_hnsw là tên index
embedding là index được tạo dựa trên cột embedding
hnsw: có nghĩa là sử dụng hnsw (hierachical Navigable Small World) để tìm các vector gần nhau nhanh hơn
embedding: vector_cosine_ops có nghĩa là index này dùng cosine distance để xác định độ tương đồng giữa các vector

GIN index (Generalized Inverted Index) -> phù hợp cho những record chứa các từ này (exact keyword) -> phục vụ Full-Text Search (FTS)
search_vector chứa dạng PostgreSQL dùng để tìm kiếm "experienced", "docker", "python", "postgresql" 
text -> PostgreSQL FTS processing
Keyword retrieval
                    USER QUERY
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Vector Embedding          PostgreSQL FTS
             │                       │
             ▼                       ▼
          HNSW Index              GIN Index
             │                       │
             ▼                       ▼
       Vector Ranking           Keyword Ranking
             │                       │
             └───────────┬───────────┘
                         ▼
                        RRF
                         │
                         ▼
                    Top-K chunks
                         │
                         ▼
                        LLM
"""


import uuid
from typing import Any
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, ForeignKeyConstraint, Index, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from app.db.base import Base, CreatedAtMixin

# migration: tạo bảng chunks trong database, dùng để lưu thông tin các chunk của document
# Quyết định SCHEMA: gemini-embedding-001 768 chiều

EMBEDDING_DIM = 1536

# Cấu hình full-text search của Postgres.
# Simple = tách từ + viết thường
# Không stemming (đưa từ về dạn gốc), không bỏ stop-word -> chạy được cho cả tiếng viết lẫn tiếng anh
TS_CONFIG = "simple"


class Chunk(CreatedAtMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = ( 
        Index("ix_chunks_embedding_hnsw", "embedding", 
              postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}), 

         # tạo index cho cột embedding, dùng để tìm kiếm nhanh hơn khi truy vấn vector
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),  

        # Phase 8: chunk phải cùng application với document của nó (migration 0003)
        ForeignKeyConstraint(
            ["document_id", "application_id"], ["documents.id", "documents.application_id"],
            name="fk_chunks_document_same_app", ondelete="CASCADE",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # Phase 8: khóa ngoại của document_id nằm trong ForeignConstraint của 2 cột ở __table_args_
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    # Lưu thẳng application_id ở chunk để filter ngay trong câu search, không cần JOIN
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)

    chunk_index: Mapped[int] = mapped_column(Integer)  # index của chunk trong document, bắt đầu từ 0
    text: Mapped[str] = mapped_column(Text)  # nội dung chunk
    page: Mapped[int | None] = mapped_column(Integer)  # số trang của chunk, nếu document là PDF

    # metadata_ bị SQLAlchemy giữ chỗ -> attribute là metadata_, cột DB vẫn tên "metadata"
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)  # metadata của chunk, có thể là số trang, tên file, v.v.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))  # embedding vector của chunk, dùng để tìm kiếm tương tự

    # Lấy nội dung trong cột text, tự động chuyển thành tsvector -> lưu kết quả vào search_vector
    # Chunk -> text -> to_tsvector("simple", text) -> search_vector -> GIN Index -> Full-Text Search
    # Computed là được tính từ giá trị cột khác
    # TSVECTOR: kiểu dữ liệu PostgreSQL của kết quả, có thể lưu vị trí của từ
    # persisted = True -> kết quả được lưu lại trong database
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(f"to_tsvector('{TS_CONFIG}', text)", persisted=True)
    )
    # quan hệ với Document, mỗi chunk thuộc về một document
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")  # noqua:FB21