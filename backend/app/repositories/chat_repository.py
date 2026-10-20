"""
MỤC ĐÍCH: Truy vấn chat_sessions, messages, retrieval_logs (Phase 8 chuẩn bị, phase 9 dùng cho /chat)

Cùng quy tắc với document_repository: mọi hàm ĐỌC đều nhận application_id và lọc trong WHERE
    - session_id đến từ client (trình duyệt gửi lên) -> không được tin;
    - get_session(app, id) chỉ trả phiên chat nếu nó THUỘC đúng application đó.
messages không có cột application_id: chúng thuộc 1 phiên chat, nên lọc qua JOIN chat_sessions

Repository KHÔNG commit (service commit)

- **`add_message(db, session, ...)` nhận object `ChatSession`, không nhận `session_id`**: để có object, 
người gọi buộc phải đi qua `get_session(db, application_id, id)`, tức là đã kiểm tra phiên chat thuộc đúng app. 
Đây là mẹo thiết kế để "không thể quên kiểm tra quyền".
- **`list_messages` JOIN `chat_sessions`**: bảng `messages` không có cột `application_id` 
(không cần, vì tin nhắn luôn thuộc 1 phiên chat). 
Lọc application qua phiên chat.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models import ChatSession, Message, RetrievalLog

def create_session(db: Session, application_id: str, user_id: uuid.UUID | None = None) -> ChatSession:
    session = ChatSession(application_id=application_id, user_id = user_id)
    db.add(session)
    db.flush()      # để có session.id ngay
    return session

def get_session(db: Session, application_id: str, session_id: uuid.UUID) -> ChatSession:
    return db.scalars(
        select(ChatSession).where(ChatSession.application_id == application_id, ChatSession.id == session_id)
    ).first()

def add_message(
    db: Session, session: ChatSession, role: str, content: str, *,
    sources: list[dict] | None = None, abstain_reason: str | None = None, grounding_verdict: str | None = None,
) -> Message:
    # Nhận cả object ChatSession (đã lấy qua get_session) thay vì session_id trần
    # -> buộc người gọi phải kiểm tra quyền trước khi ghi.
    message = Message(
        session_id=session.id, role=role, content=content,
        sources=sources or [], abstain_reason=abstain_reason, grounding_verdict=grounding_verdict,
    )
    db.add(message)
    db.flush()
    return message


def list_messages(db: Session, application_id: str, session_id: uuid.UUID) -> list[Message]:
    return list(db.scalars(
        select(Message)
        .join(ChatSession, ChatSession.id == Message.session_id)            
        .where(ChatSession.application_id == application_id, Message.session_id == session_id)
        .order_by(Message.seq)             # không dùng created_at: 2 tin nhắn cùng transaction có created_at bằng nhau
    ))


# Lấy N tin cuối sau đó đảo lại
def recent_messages(db: Session, application_id: str, session_id: uuid.UUID, limit: int) -> list[Message]:
    """limit: tin nhắn CUỐI của phiên chat, trả theo thứ tự cũ -> mới (Phase 9: lịch sử cho bước viết lại câu hỏi)"""
    newest_first = list(db.scalars(
        select(Message)
        .join(ChatSession, ChatSession.id == Message.session_id)
        .where(ChatSession.application_id == application_id, Message.session_id == session_id)
        .order_by(Message.seq.desc())
        .limit(limit)
    ))
    return newest_first[::-1]

@dataclass(frozen=True)
class SessionSummary:
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    message_count: int
    first_question: str | None


# lấy số tin nhắn và câu hỏi đầu tiên của từng phiên
def list_sessions(db: Session, application_id: str, limit: int = 50) -> list[SessionSummary]:
    """Phiên chat của 1 application, mới nhất trước, kèm số tin nhắn và câu hỏi đầu tiên (làm "tiêu đề")"""
    # Subquery tương quan: với MỖI phiên chat, lấy content của tin nhắn user đầu tiên

    # Lọc tin nhắn đầu tiên của user là gì
    first_question = (
        select(Message.content)
        .where(Message.session_id == ChatSession.id, Message.role == "user")
        .order_by(Message.seq)
        .limit(1)
        .scalar_subquery()
    )

    # Lọc 1 chat session có bao nhiêu tin nhắn
    message_count = (
        select(func.count(Message.id)).where(Message.session_id == ChatSession.id).scalar_subquery()
    )

    rows = db.execute(
        select(ChatSession.id, ChatSession.created_at, ChatSession.updated_at, message_count, first_question)
        .where(ChatSession.application_id == application_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id)
        .limit(limit)
    ).all()
    return [SessionSummary(*row) for row in rows]         

def touch_session(db: Session, session: ChatSession) -> None:
    """Cập nhật updated_at = thời điểm có tin nhắn mới (để list_sessions sắp đúng "mới nhất trước")."""
    session.updated_at = func.now()
    db.flush()

def add_retrieval_log(
    db: Session, *, application_id: str, query: str, chunk_ids: list[str], scores: list[float],
    latency_ms: float | None = None, session_id: uuid.UUID | None = None, message_id: uuid.UUID | None = None
) -> RetrievalLog:
    log = RetrievalLog(
        application_id = application_id, session_id = session_id, message_id = message_id, query = query,
        retrieved_chunk_ids = chunk_ids, scores = scores, latency_ms = latency_ms,
    )
    db.add(log)
    db.flush()
    return log

