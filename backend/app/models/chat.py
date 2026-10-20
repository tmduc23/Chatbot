import uuid

from sqlalchemy import BigInteger, Float, ForeignKey, ForeignKeyConstraint, Identity, String, Text, UniqueConstraint, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, CreatedAtMixin, TimestampMixin

class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"

    # Phase 8: để retrieval_logs tham chiếu được cặp (id, application_id) - migration 0003
    __table_args__ = (UniqueConstraint("id", "application_id", name="uq_chat_sessions_id_app"),)

    
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # nullable: recruiter chưa cần đăng nhập ở giai đoạn đầu
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL")) # 
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id", ondelete="CASCADE"), index=True)


class Message(CreatedAtMixin, Base):
    __tablename__ = "messages"


    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" hoặc "assistant"

    # Phase 8: số thứ tự do Postgres tự tăng -> sắp xếp tin nhắn chắc chắn đúng
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True))
    content: Mapped[str] = mapped_column(Text)      # nội dung tin nhắn
    # metadata_: Mapped[dict[str, str]] = mapped_column("metadata", JSONB, default=dict)  # metadata của tin nhắn, có thể là thông tin về nguồn dữ liệu, thời gian, v.v.

    # Phase 9 (migration 0004): chỉ có ý nghĩa với tin nhắn "assistant"
    sources: Mapped[list[dict]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    abstain_reason: Mapped[str | None] = mapped_column(String(50))              # None = trả lời bình thường
    grounding_verdict: Mapped[str | None] = mapped_column(String(20))           # None = grounding check tắt


class RetrievalLog(CreatedAtMixin, Base):
    __tablename__ = "retrieval_logs"
    # Phase 8: log của 1 phiên chat phải cùng application với phiên chat đó
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "application_id"], ["chat_sessions.id", "chat_sessions.application_id"],
            name="fk_retrieval_logs_session_same_app", ondelete="CASCADE"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # Phase 8: Khóa ngoại của session_id nằm trong ForeignKeyConstraint 2 cột ở __table__args
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)

    # Phase 8: application_id có mặt ở mọi bảng nghiệp vụ
    application_id: Mapped[str] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    query: Mapped[str] = mapped_column(Text)  # truy vấn của người dùng

    # danh sách các chunk được truy xuất, lưu dưới dạng JSON
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)  
    scores: Mapped[list[float]] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[float | None] = mapped_column(Float)  # thời gian truy xuất (tính bằng mili giây)

    # Phase 9: câu trả lời (tin nhắn)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), index=True
    )