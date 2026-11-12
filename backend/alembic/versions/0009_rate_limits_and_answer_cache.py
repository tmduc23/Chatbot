"""bảng rate_limits (bộ đếm request) + answer_cache (cache câu trả lời)

Revision ID: 0009
Revises: 0008

MỤC ĐÍCH (Phase 12c):
  - rate_limits : (key, window_start) -> count. Mỗi request chat/login/upload tăng 1 dòng bằng
                  INSERT ... ON CONFLICT DO UPDATE (nguyên tử: 2 request cùng lúc không đếm sót).
                  Đếm trong Postgres thay vì RAM: backend chạy nhiều process / restart vẫn đúng.
  - answer_cache: key = sha256(application + câu hỏi + "dấu vân tay" tài liệu + cấu hình RAG).
                  Tài liệu hay cấu hình đổi -> key đổi -> tự không dùng câu trả lời cũ nữa.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rate_limits",
        sa.Column("key", sa.String(200), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("count", sa.Integer, nullable=False),
    )
    # dọn cửa sổ cũ: DELETE ... WHERE window_start < ...
    op.create_index("ix_rate_limits_window_start", "rate_limits", ["window_start"])

    op.create_table(
        "answer_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("application_id", sa.String(50),
                  sa.ForeignKey("applications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hits", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_answer_cache_application_id", "answer_cache", ["application_id"])
    op.create_index("ix_answer_cache_expires_at", "answer_cache", ["expires_at"])


def downgrade() -> None:
    op.drop_table("answer_cache")
    op.drop_table("rate_limits")
