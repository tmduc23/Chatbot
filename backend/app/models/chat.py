import uuid
from sqlalchemy import ForeignKey, String, Text, Uuid, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base, CreatedAtMixin, TimestampMixin

class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"

    id : Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True) # 
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)


class Message(CreatedAtMixin, Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" hoặc "assistant"
    content: Mapped[str] = mapped_column(Text)      # nội dung tin nhắn
    # metadata_: Mapped[dict[str, str]] = mapped_column("metadata", JSONB, default=dict)  # metadata của tin nhắn, có thể là thông tin về nguồn dữ liệu, thời gian, v.v.

class RetrievalLog(CreatedAtMixin, Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)  # truy vấn của người dùng
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)  # danh sách các chunk được truy xuất, lưu dưới dạng JSON
    latency_ms: Mapped[float | None] = mapped_column(Float)  # thời gian truy xuất (tính bằng giây)