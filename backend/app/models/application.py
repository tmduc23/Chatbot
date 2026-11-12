"""
Bảng application: mỗi knowledge domain (cv, projects) là một dòng.
"""

from sqlalchemy import Boolean, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base, TimestampMixin

# Định nghĩa cấu trúc bảng "applications" trong database, dùng để lưu thông tin các ứng dụng RAG
class Application(TimestampMixin, Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)   # "cv", "projects", primary_key=True: đánh dấu id là khóa chính
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    # Phase 12b: true = ai cũng chat được (không cần đăng nhập), vd. CV cho nhà tuyển dụng.
    # Mặc định false: application mới là RIÊNG TƯ cho tới khi bạn chủ động mở.
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
