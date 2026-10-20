"""
MỤC ĐÍCH: ChatService - nghiệp vụ " 1 lượt hỏi đáp trong 1 phiên chat (Phase 9: POST /api/v1/chat)

RAGService (Phase 6-8) trả lời 1 câu hỏi ĐƠN LẺ. ChatService bọc thêm phần HỘI THOẠI:
    
    {application_id, session_id | null, query}
    -> application có tồn tại ?                     (404 nếu không)
    -> kiểm tra câu hỏi                             (422 nếu rỗng/quá dài)
    -> session_id có -> phiên chat đó có THUỘC app này? (404 nếu không - không tin id từ client)
    -> lấy vài lượt hỏi-đáp gần nhát -> viết lại câu hỏi thành câu độc lập (tùy chọn, fail-open)
    -> RAGService.ask(câu độc lập)                  (LLM lỗi -> LLMError -> 503, KHÔNG ghi gì)
    -> Chưa có phiên chat -> tạo mới
    -> ghi tin nhắn user + assistant (kèm sources, lý do từ chối, grounding)
    -> ghi retrieval_log (chunk nào, điểm boa nhiêu, mất bao lâu)
    -> commit 1 lần: hoặc lưu đủ cả lượt, hoặc không lưu gì cả

Tạo phiên chat SAU khi gọi LLM: nếu LLM lỗi thì không để lại phiên chat rỗng trong database 
"""

import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.exceptions import ChatSessionNotFoundError
from app.models import ChatSession, Message
from app.rag.question_condenser import QuestionCondenser
from app.repositories import application_repository as app_repo
from app.repositories import chat_repository as chat_repo
from app.services.rag_service import RAGAnswer, RAGService, validate_question


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurn:
    session_id: uuid.UUID
    message_id: uuid.UUID                   # id tin nhắn assistant vừa lưu
    result: RAGAnswer
    standalone_question: str | None         # câu đã viết lại (None = dùng nguyên câu gốc)
    total_ms: float


class ChatService:
    def __init__(
        self, 
        db: Session,
        rag: RAGService,
        condenser: QuestionCondenser | None = None,
        history_turns: int = 3,
    ):
        self.db = db
        self.rag = rag
        self.condenser = condenser          # None = tắt bước viết lại câu hỏi
        self.history_turns = history_turns

    def chat(self, application_id: str, query: str, session_id: uuid.UUID | None = None) -> ChatTurn:
        t0 = time.perf_counter()
        app_repo.require(self.db, application_id)       # kiểm tra app
        question = validate_question(query)


        session: ChatSession | None = None
        history: list[tuple[str, str]] = []
        if session_id is not None:
            session = chat_repo.get_session(self.db, application_id, session_id)            # tìm phiên thuộc đúng app
            if session is None:
                raise ChatSessionNotFoundError(f"Chat session {session_id} not found")
            # 1 lượt = 1 câu user + 1 câu assistant
            recent = chat_repo.recent_messages(self.db, application_id, session_id, self.history_turns * 2)
            history = [(m.role, m.content) for m in recent]

        
        standalone = question
        if self.condenser is not None and history:
            standalone = self.condenser.condense(question, history).question

        result = self.rag.ask(standalone, application_id)

        if session is None:
            session = chat_repo.create_session(self.db, application_id)
        chat_repo.add_message(self.db, session, "user", question)

        answer_msg = chat_repo.add_message(
            self.db, session, "assistant", result.answer,
            sources = [s.to_dict() for s in result.sources],
            abstain_reason = result.abstain_reason,
            grounding_verdict = result.grounding.verdict if result.grounding else None
        )

        reranked = result.retrieval.reranked
        chat_repo.add_retrieval_log(
            self.db, application_id=application_id, session_id=session.id, message_id=answer_msg.id,
            query = standalone,
            chunk_ids=[str(r.chunk.chunk_id) for r in reranked],
            scores = [round(r.score, 4) for r in reranked],
            latency_ms=round(result.total_ms, 1),
        )

        """Cập nhật updated_at = thời điểm có tin nhắn mới (để list_sessions sắp đúng "mới nhất trước")."""
        chat_repo.touch_session(self.db, session)
        self.db.commit()

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("chat app=%s session=%s rewritten=%s abstaine%s total_ms=%.0f",
                    application_id, session.id, standalone != question, result.abstained, total_ms)

        return ChatTurn(
            session_id=session.id, message_id=answer_msg.id, result=result,
            standalone_question=standalone if standalone != question else None, total_ms=total_ms
        )


    def list_sessions(self, application_id: str, limit: int = 50) -> list[chat_repo.SessionSummary]:
        app_repo.require(self.db, application_id)
        return chat_repo.list_sessions(self.db, application_id, limit)

    def get_session(self, application_id: str, session_id: uuid.UUID) -> tuple[ChatSession, list[Message]]:
        app_repo.require(self.db, application_id)
        session = chat_repo.get_session(self.db, application_id, session_id)
        if session is None: 
            raise ChatSessionNotFoundError(f"Chat session {session_id} not found")
        return session, chat_repo.list_messages(self.db, application_id, session_id)