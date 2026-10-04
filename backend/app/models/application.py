from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base, TimestampMixin

# Định nghĩa cấu trúc bảng "applications" trong database, dùng để lưu thông tin các ứng dụng RAG
class Application(TimestampMixin, Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)   # "cv", "projects"
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)