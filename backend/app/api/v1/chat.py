"""
Route /api/v1/chat
"""

import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_chat_service, require_admin
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse, SessionDetailOut, SessionSummaryOut
from app.services.chat_service import ChatService


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, service: ChatService = Depends(get_chat_service)):
    turn = service.chat(body.application_id, body.query, body.session_id)
    r = turn.result
    return ChatResponse(
        session_id=turn.session_id,
        message_id=turn.message_id,
        answer=r.answer,
        sources=[s.to_dict() for s in r.sources],
        abstained=r.abstained,
        metadata=ChatMetadata(
            retrieval_count=len(r.retrieval.candidates),
            reranked_count=len(r.retrieval.reranked),
            relevant_count=len(r.retrieval.relevant),
            abstain_reason=r.abstain_reason,
            standalone_question=turn.standalone_question,
            latency_ms=round(turn.total_ms, 1),
        ),
    )

# Liệt kê MỌI phiên chat = đọc được hội thoại của người khác -> chỉ admin
@router.get("/sessions", response_model=list[SessionSummaryOut], dependencies=[Depends(require_admin)])
def list_sessions(
    application_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
    service: ChatService = Depends(get_chat_service),
):
    return service.list_sessions(application_id, limit)

# Xem 1 phiên chat: ai có session_id (UUID ngẫu nhiên, không đoán được) thì xem được —
# giống "link chia sẻ". Frontend dùng để mở lại cuộc chat của chính mình.
@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(
    session_id: uuid.UUID,
    application_id: str = Query(...),
    service: ChatService = Depends(get_chat_service)
):
    session, messages = service.get_session(application_id, session_id)
    return SessionDetailOut(id=session.id, application_id=session.application_id, 
                            created_at=session.created_at, messages=messages)