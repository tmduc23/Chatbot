"""
chunks: BM25 index của extension pg_textsearch

Revision ID: 0006
Revises: 0005


MỤC ĐÍCH: Cho KEYWWORD_RANKING=pg_textsearch: BM25 thật, có IDF, chạy trên index riêng (USING bm25).

Index đọc thẳng cột chữ `text` (không dùng cột search_vector). text_config='simple' giống
  search_vector (TS_CONFIG) để 3 cách xếp hạng (ts_rank / bm25 / pg_textsearch) cùng cách tách từ.

- k1=1.2, b=0.75 nằm TRONG index (muốn đổi phải tạo lại index), không đọc từ .env.
- Index tính IDF trên TOÀN BỘ bảng chunks (mọi application gộp lại), khác bản BM25 SQL
  (tính riêng từng application).

Extension cần Postgres 17/18 + shared_preload_libraries=pg_textsearch (xem guides/nang-postgres-17.md).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX = "ix_chunks_bm25"

def _extension_available() -> bool:
    found = op.get_bind().execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_textsearch'")
    ).first()
    return found is not None


def upgrade() -> None:
    if not _extension_available():
        print("pg_textsearch không có trên server này -> bỏ qua index BM25 (KEYWORD_RANKING=pg_textsearch sẽ báo lỗi)")
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")
    op.execute(
        f'CREATE INDEX IF NOT EXISTS {INDEX} ON chunks USING bm25 ("text") '
        "WITH (text_config = 'simple', k1 = 1.2, b = 0.75)"
    )

def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {INDEX}")
    op.execute("DROP EXTENSION IF EXISTS pg_textsearch")