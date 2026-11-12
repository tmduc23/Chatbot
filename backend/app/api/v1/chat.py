"""
Route /api/v1/chat
"""

import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import RateLimit, get_access, get_chat_service
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse, SessionDetailOut, SessionSummaryOut
from app.services.access import AccessControl
from app.services.chat_service import ChatService


router = APIRouter(prefix="/chat", tags=["chat"])


# Phase 12c: mỗi người tối đa RATE_LIMIT_CHAT_PER_MINUTE / _PER_DAY câu hỏi -> quá thì 429
@router.post("", response_model=ChatResponse, dependencies=[Depends(RateLimit("chat"))])
def chat(
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    access: AccessControl = Depends(get_access),
):
    access.require_chat(body.application_id)
    # Phase 12b: phiên chat mới gắn với user đang đăng nhập (khách: None).
    # Tiếp tục phiên của NGƯỜI KHÁC -> 404 như phiên không tồn tại.
    turn = service.chat(body.application_id, body.query, body.session_id, body.document_ids,
                        user_id=access.principal.user_id)
    a = turn.answer   # Phase 12c: AnswerSnapshot (có thể lấy từ cache)
    return ChatResponse(
        session_id=turn.session_id,
        message_id=turn.message_id,
        answer=a.answer,
        sources=a.sources,
        abstained=a.abstained,
        metadata=ChatMetadata(
            retrieval_count=a.retrieval_count,
            reranked_count=a.reranked_count,
            relevant_count=a.relevant_count,
            abstain_reason=a.abstain_reason,
            standalone_question=turn.standalone_question,
            latency_ms=round(turn.total_ms, 1),
            cached=a.cached,
        ),
    )


# Liệt kê MỌI phiên chat = đọc được hội thoại của người khác -> chỉ editor của application / admin
@router.get("/sessions", response_model=list[SessionSummaryOut])
def list_sessions(
    application_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
    service: ChatService = Depends(get_chat_service),
    access: AccessControl = Depends(get_access),
):
    access.require_editor(application_id)
    return service.list_sessions(application_id, limit)


# Xem 1 phiên chat. Phiên của khách: ai có session_id (UUID ngẫu nhiên, không đoán được) thì xem được,
# giống "link chia sẻ". Phiên của user đã đăng nhập (Phase 12b): chỉ chủ phiên, editor, admin.
@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(
    session_id: uuid.UUID,
    application_id: str = Query(...),
    service: ChatService = Depends(get_chat_service),
    access: AccessControl = Depends(get_access),
):
    access.require_chat(application_id)
    session, messages = service.get_session(application_id, session_id, can_read=access.can_read_session)
    return SessionDetailOut(id=session.id, application_id=session.application_id,
                            created_at=session.created_at, messages=messages)