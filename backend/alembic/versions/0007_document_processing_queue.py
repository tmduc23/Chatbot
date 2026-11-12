"""documents: thêm các cột để bảng documents làm HÀNG ĐỢI xử lý nền

Revision ID: 0007
Revises: 0006

MỤC ĐÍCH (Phase 12a): upload chỉ lưu file + tạo document PENDING rồi trả lời ngay (HTTP 202).
Worker (scripts/worker.py) lấy document PENDING ra xử lý (parse -> chunk -> embed -> lưu).
  - attempts     : đã thử bao nhiêu lần (lỗi tạm thời thì thử lại, tối đa WORKER_MAX_ATTEMPTS)
  - available_at : sớm nhất lúc nào được xử lý (thử lại thì lùi về sau: 30s, 60s, 120s...)
  - locked_at    : lúc worker nhận việc; PROCESSING quá lâu = worker đã chết -> worker khác lấy lại
Index một phần (partial index): chỉ chứa các dòng PENDING -> tìm việc nhanh dù bảng có
hàng nghìn document READY.
"""

from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None



def upgrade() -> None:
     # server_default: các document cũ (đã READY/FAILED) tự nhận giá trị, cột vẫn NOT NULL được
    op.add_column("documents", sa.Column("attempts", sa.Integer, nullable=False, server_default="0"))
    op.add_column("documents", sa.Column("available_at", sa.DateTime(timezone=True), nullable=False,
                                         server_default=sa.func.now()))
    op.add_column("documents", sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_documents_queue", "documents", ["available_at"],
                    postgresql_where=sa.text("status = 'PENDING'"))


def downgrade() -> None:
    op.drop_index("ix_documents_queue", table_name="documents")
    op.drop_column("documents", "locked_at")
    op.drop_column("documents", "available_at")
    op.drop_column("documents", "attempts")
 