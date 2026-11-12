"""chunks.embedding: vector(768) -> vector(1536)

Revision ID: 0005
Revises: 0004

MỤC ĐÍCH: tăng số chiều embedding lên 1536 (gemini-embedding-2 khuyên dùng 768 / 1536 / 3072).

Postgres không đổi được vector 768 chiều thành 1536 chiều (không có cách "kéo dài" vector cho đúng),
nên migration này:
  1. bỏ index HNSW (index gắn với số chiều cũ);
  2. xoá hết vector cũ (đặt NULL) rồi đổi kiểu cột;
  3. tạo lại index HNSW.
Chữ của chunk giữ nguyên. Sau migration PHẢI chạy scripts/reembed_all.py --all,
nếu không thì tìm kiếm vector trả về rỗng (mọi chunk đều chưa có vector).
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "ix_chunks_embedding_hnsw"


def _set_dim(dim: int) -> None:
    op.drop_index(INDEX, table_name="chunks")
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dim})")
    op.create_index(INDEX, "chunks", ["embedding"],
                    postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})


def upgrade() -> None:
    _set_dim(1536)


def downgrade() -> None:
    _set_dim(768)
