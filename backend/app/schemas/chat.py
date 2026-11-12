"""
Pydantic schema cho /api/v1/chat (Phase 9).

Schema = "hợp đồng" giữa backend và frontend:
  - Schema ĐẦU VÀO (ChatRequest): FastAPI kiểm tra body request. Sai kiểu/thiếu field/quá dài -> 422,
    route chưa hề chạy, chưa tốn lần gọi LLM nào.
  - Schema ĐẦU RA (các class *Out, ChatResponse): chỉ những field khai báo ở đây mới ra JSON.
    Cột nào không khai báo (vd. password_hash ở Phase 12b) không bao giờ bị lộ.
  - Trang /docs (Swagger) được tạo tự động từ chính các class này.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.services.rag_service import MAX_QUESTION_CHARS


# Đầu vào

class ChatRequest(BaseModel):
    """Body của POST /api/v1/chat"""
    application_id: str = Field(min_length=1, max_length=50, examples=["cv"])
    session_id: uuid.UUID | None = Field(default=None, description="null = bắt đầu phiên chat mới")
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, examples=["Bạn ấy đã làm những dự án nào"])
    # Chế độ paper: null = hỏi cả application; [id...] = chỉ tìm trong các tài liệu này (tối đa 10)
    document_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=10,
                                                 description="null = hỏi cả application")


# Đầu ra: POST /chat

class SourceOut(BaseModel):
    """1 nguồn trích dẫn. Chỉ trả thông tin để HIỂN THỊ, không trả nguyên văn chunk (JSON gọn, ít lộ văn bản gốc)"""

    number: int             # khớp với [Source n] trong answer -> frontend biến thành link bấm được
    document_id: uuid.UUID  # tài liệu chứa đoạn được trích
    filename: str           # tên file để người đọc nhận ra, vd "cv.pdf"
    page: int | None        # số trang (PDF): None với file không có trang (.txt, .md)
    chunk_id: uuid.UUID     # đoạn cụ thể đã dùng
    text: str | None = None     # Phase 10b: đoạn trích LLM đã đọc. Tin nhắn lưu trước Phase 10b không có -> None


class ChatMetadata(BaseModel):
    """Thông tin "hậu trường" của 1 lượt chat: để debug, đánh giá và để frontend hiện thêm chi tiết."""

    retrieval_count: int                # số chunk ứng viên (hybrid search)
    reranked_count: int                 # số chunk reranker chấm
    relevant_count: int                 # số chunk qua ngưỡng -> đưa vào LLM (0 = từ chối)
    abstain_reason: str | None          # lý do từ chối: None = có trả lời
    standalone_question: str | None     # câu hỏi sau khi condenser viết lại (None = giữ nguyên câu gốc)
    latency_ms: float                   # thời gian xử lý của lượt chat
    cached: bool = False                # Phase 12c: True = lấy từ cache câu trả lời, không gọi API nào


class ChatResponse(BaseModel):
    """Kết quả POST /api/v1/chat."""

    session_id: uuid.UUID               # fronteend lưu lại và gửi kèm ở câu hỏi sau để HỎI TIẾP trong phiên này
    message_id: uuid.UUID               # id của tin nhắn câu trả lời (khớp retrieval_logs.message_id)
    answer: str                         # câu trả lời (hoặc câu từ chối)
    sources: list[SourceOut]            # các nguồn được trích trong answer
    abstained: bool                     # True = hệ thống từ chối vì không tìm thấy trong tài liệu
    metadata: ChatMetadata              # gom phần kỹ thuật vào 1 object con, phần chính (answer, sources) gọn hơn


# ĐẦU RA: xem lại phiên chat

class SessionSummaryOut(BaseModel):
    """1 dòng trong danh sách phiên: GET /api/v1/chat/sessions (admin)."""

    # from_attributes: chop phép tạo thẳng từ object có thuộc tính (ở đây là chat_repo.SessionSummary)
    # thay vì phải chuyển sang dict trước: SessionSummaryOut.model_validate(summary)
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime                # thời điểm có tin nhắn mới nhất (danh sách sắp theo cột này)
    message_count: int                  # từ scalar subquery count(...) trong list_sessions
    first_question: str | None          # câu hỏi user đầu tiên, dùng làm "tiêu đề" phiên


class MessageOut(BaseModel):
    """1 tin nhắn khi mở lại phiên chat"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str                   # "user" | "assistant"
    content: str                # câu hỏi GỐC (user) hoặc câu trả lời (assistant)
    sources: list[SourceOut]    # đọc từ cột JSONB messages.sources -> mởi lại phiên vẫn thấy đúng trích dẫn
    abstain_reason: str | None
    created_at: datetime


class SessionDetailOut(BaseModel):
    """
    Toàn bộ 1 phiên: GET /api/v1/chat/sessions/{id}.

    Không có from_attributes: route tự tạo bằng tay, ghép từ 2 nguồn (object phiên + list tin nhắn):
        SessionDetailOut(id=session.id, application_id=..., created_at=..., messages=messages)
    """

    id: uuid.UUID
    application_id: str
    created_at: datetime
    messages: list[MessageOut]