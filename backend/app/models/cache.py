"""Bảng của Phase 12c: bộ đếm rate limit và cache câu trả lời (migration 0009)."""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, CreatedAtMixin


class RateLimitCounter(Base):
    """1 dòng = "key này đã gọi bao nhiêu lần trong cửa sổ thời gian bắt đầu lúc window_start"."""
    __tablename__ = "rate_limits"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)        # vd. "chat:ip:1.2.3.4:60"
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, index=True)
    count: Mapped[int] = mapped_column(Integer)


class AnswerCacheEntry(CreatedAtMixin, Base):
    __tablename__ = "answer_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)          # sha256 hex
    application_id: Mapped[str] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)                             # để xem bằng mắt khi debug
    payload: Mapped[dict] = mapped_column(JSONB)                            # AnswerSnapshot dạng JSON
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    hits: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
