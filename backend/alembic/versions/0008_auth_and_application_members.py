"""users đăng nhập được + bảng application_members + applications.is_public

Revision ID: 0008
Revises: 0007

MỤC ĐÍCH (Phase 12b): biết AI đang gọi API và người đó được làm gì trên application nào.
  - users.password_hash  : hash scrypt của mật khẩu (không bao giờ lưu mật khẩu gốc)
  - users.is_active      : false = khoá tài khoản, token cũ cũng hết hiệu lực ngay
  - users.is_superadmin  : làm được mọi việc trên mọi application
  - application_members  : (application, user) -> vai trò viewer | editor
  - applications.is_public: true = khách không đăng nhập cũng chat được (CV cho nhà tuyển dụng)
Mọi application cũ nhận is_public = false (an toàn: khoá trước, mở sau bằng scripts/manage_users.py).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()))
    op.add_column("users", sa.Column("is_superadmin", sa.Boolean, nullable=False, server_default=sa.false()))
    op.add_column("applications", sa.Column("is_public", sa.Boolean, nullable=False, server_default=sa.false()))

    op.create_table(
        "application_members",
        sa.Column("application_id", sa.String(50),
                  sa.ForeignKey("applications.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Uuid, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('viewer', 'editor')", name="ck_application_members_role"),
    )
    # "user này thuộc những application nào" (lúc đăng nhập / mỗi request) -> cần index theo user_id
    op.create_index("ix_application_members_user_id", "application_members", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_application_members_user_id", table_name="application_members")
    op.drop_table("application_members")
    op.drop_column("applications", "is_public")
    op.drop_column("users", "is_superadmin")
    op.drop_column("users", "is_active")
    op.drop_column("users", "password_hash")
