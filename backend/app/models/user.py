import uuid

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, CreatedAtMixin


class User(CreatedAtMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)   # luôn lưu chữ thường
    name: Mapped[str | None] = mapped_column(String(200))
    # Phase 12b (migration 0008). Chỉ lưu HASH của mật khẩu, không bao giờ lưu mật khẩu gốc.
    # NULL = tài khoản chưa đặt mật khẩu -> không đăng nhập được
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    # superadmin: tạo application, làm mọi việc trên mọi application
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))


# Vai trò trong 1 application, xếp từ thấp lên cao
ROLE_VIEWER = "viewer"   # hỏi đáp (chat) trên application riêng tư
ROLE_EDITOR = "editor"   # + upload/xoá tài liệu, xem mọi phiên chat của application
ROLES = (ROLE_VIEWER, ROLE_EDITOR)


class ApplicationMember(CreatedAtMixin, Base):
    """User X có vai trò Y trong application Z. Không có dòng = không có quyền gì (trừ app công khai)."""
    __tablename__ = "application_members"
    __table_args__ = (CheckConstraint("role IN ('viewer', 'editor')", name="ck_application_members_role"),)

    # khoá chính 2 cột: 1 user chỉ có 1 vai trò trong 1 application
    application_id: Mapped[str] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    role: Mapped[str] = mapped_column(String(20))
