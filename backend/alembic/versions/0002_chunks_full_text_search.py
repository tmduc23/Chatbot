"""
Revision ID: 0002
Revises: 0001

MỤC ĐÍCH: Thêm khả năng tìm kiếm theo TỪ KHÓA cho bảng chunks

- search vector: cột GENERATED - Postgre tự tính to_tsvector('simple',text)
mỗi khi chunk được thêm/sửa. Các chunk đã có cũng được tính ngay khi chạy migration.
- GTN index: chỉ mục ngược (từ -> danh sách chunk chứa từ đó), giúp tìm từ khóa nhanh 
mà không phải quét toàn bộ bảng.
"""

from collections.abc import Sequence
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR
import sqlalchemy as sa

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None        # đặt tên cho 1 nhánh migration
depends_on: str | Sequence[str] | None = None           # phụ thuộc vào nhánh nào


def upgrade() -> None:
    op.add_column(
        "chunks", 
        sa.Column("search_vector", TSVECTOR,
                  sa.Computed("to_tsvector('simple', text)", persisted=True))
    )
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")

def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")