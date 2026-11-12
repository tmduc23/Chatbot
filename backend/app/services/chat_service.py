"""
MỤC ĐÍCH: ChatService - nghiệp vụ "1 lượt hỏi đáp trong 1 phiên chat" (Phase 9: POST /api/v1/chat)

RAGService (Phase 6-8) trả lời 1 câu hỏi ĐƠN LẺ. ChatService bọc thêm phần HỘI THOẠI:

    {application_id, session_id | null, query}
    -> application có tồn tại?                          (404 nếu không)
    -> kiểm tra câu hỏi                                 (422 nếu rỗng/quá dài)
    -> session_id có -> phiên chat đó có THUỘC app này?  (404 nếu không - không tin id từ client)
       Phase 12b: và có thuộc user đang chat không?     (404 nếu là phiên của user khác)
    -> lấy vài lượt hỏi-đáp gần nhất -> viết lại câu hỏi thành câu độc lập (tùy chọn, fail-open)
    -> cache có câu trả lời cho câu độc lập này?      (Phase 12c: có -> bỏ qua bước dưới, 0 lời gọi API)
    -> RAGService.ask(câu độc lập)                      (LLM lỗi -> LLMError -> 503, KHÔNG ghi gì)
    -> chưa có phiên chat -> tạo mới (gắn với user_id)
    -> ghi tin nhắn user + assistant (kèm sources, lý do từ chối, grounding)
    -> ghi retrieval_log (chunk nào, điểm bao nhiêu, mất bao lâu)
    -> commit 1 lần: hoặc lưu đủ cả lượt, hoặc không lưu gì cả

Tạo phiên chat SAU khi gọi LLM: nếu LLM lỗi thì không để lại phiên chat rỗng trong database.
"""

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.exceptions import ChatSessionNotFoundError, DocumentNotFoundError
from app.core.metrics import ANSWER_CACHE, CHAT_ANSWERS
from app.models import ChatSession, Message
from app.rag.question_condenser import QuestionCondenser
from app.repositories import application_repository as app_repo
from app.repositories import chat_repository as chat_repo
from app.repositories import document_repository as doc_repo
from app.services.answer_cache import AnswerCache, AnswerSnapshot
from app.services.rag_service import RAGService, validate_question

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurn:
    session_id: uuid.UUID
    message_id: uuid.UUID                   # id tin nhắn assistant vừa lưu
    answer: AnswerSnapshot                  # Phase 12c: bản rút gọn của RAGAnswer (lưu được vào cache)
    standalone_question: str | None         # câu đã viết lại (None = dùng nguyên câu gốc)
    total_ms: float


class ChatService:
    def __init__(
        self,
        db: Session,
        rag: RAGService,
        condenser: QuestionCondenser | None = None,
        history_turns: int = 3,
        cache: AnswerCache | None = None,
    ):
        self.db = db
        self.rag = rag
        self.condenser = condenser          # None = tắt bước viết lại câu hỏi
        self.history_turns = history_turns
        self.cache = cache                  # None = tắt cache câu trả lời (Phase 12c)

    def chat(self, application_id: str, query: str, session_id: uuid.UUID | None = None,
             document_ids: Sequence[uuid.UUID] | None = None,
             user_id: uuid.UUID | None = None) -> ChatTurn:
        """user_id (Phase 12b): người đang chat (None = khách). Phiên mới gắn với user này;
        tiếp tục phiên đã gắn với user KHÁC -> ChatSessionNotFoundError (không lộ là phiên có tồn tại)."""
        t0 = time.perf_counter()
        app_repo.require(self.db, application_id)       # kiểm tra app
        question = validate_question(query)

        # Chế độ paper: mọi tài liệu được chọn phải THUỘC application này (không tin id từ client)
        # Sai 1 id -> 404 rõ ràng, thay vì lặng lẽ tìm ra 0 kết quả rồi trả lời "không tìm thấy".
        for document_id in document_ids or []:
            if doc_repo.get_document(self.db, application_id, document_id) is None:
                raise DocumentNotFoundError(f"Document {document_id} not found")

        session: ChatSession | None = None
        history: list[tuple[str, str]] = []
        if session_id is not None:
            session = chat_repo.get_session(self.db, application_id, session_id)   # tìm phiên thuộc đúng app
            # Phase 12b: phiên của user khác -> báo "không tồn tại" (phiên của khách: user_id None, ai cũng tiếp tục được)
            if session is None or (session.user_id is not None and session.user_id != user_id):
                raise ChatSessionNotFoundError(f"Chat session {session_id} not found")
            # 1 lượt = 1 câu user + 1 câu assistant
            recent = chat_repo.recent_messages(self.db, application_id, session_id, self.history_turns * 2)
            history = [(m.role, m.content) for m in recent]

        standalone = question
        if self.condenser is not None and history:
            standalone = self.condenser.condense(question, history).question

        snap = self._answer(application_id, standalone, document_ids)

        if session is None:
            session = chat_repo.create_session(self.db, application_id, user_id=user_id)
        chat_repo.add_message(self.db, session, "user", question)

        answer_msg = chat_repo.add_message(
            self.db, session, "assistant", snap.answer,
            sources=snap.sources,
            abstain_reason=snap.abstain_reason,
            grounding_verdict=snap.grounding_verdict,
        )
        chat_repo.add_retrieval_log(
            self.db, application_id=application_id, session_id=session.id, message_id=answer_msg.id,
            query=standalone,
            chunk_ids=snap.chunk_ids,
            scores=snap.scores,
            latency_ms=snap.rag_ms,
        )

        # Cập nhật updated_at = thời điểm có tin nhắn mới (để list_sessions sắp đúng "mới nhất trước")
        chat_repo.touch_session(self.db, session)
        self.db.commit()

        total_ms = (time.perf_counter() - t0) * 1000
        # Phase 12d: tỉ lệ từ chối trả lời, tỉ lệ trúng cache theo application
        outcome = snap.abstain_reason or "answered"
        CHAT_ANSWERS.labels(application_id, outcome, str(snap.cached).lower()).inc()
        logger.info("chat app=%s session=%s rewritten=%s abstained=%s cached=%s total_ms=%.0f",
                    application_id, session.id, standalone != question, snap.abstained, snap.cached, total_ms)

        return ChatTurn(
            session_id=session.id, message_id=answer_msg.id, answer=snap,
            standalone_question=standalone if standalone != question else None, total_ms=total_ms,
        )

    def _answer(self, application_id: str, question: str,
                document_ids: Sequence[uuid.UUID] | None) -> AnswerSnapshot:
        """Phase 12c: lấy từ cache nếu có, không thì chạy RAG rồi cất vào cache (ghi cùng commit với lượt chat).
        document_ids (chế độ paper) nằm trong key: cùng câu hỏi ở 2 paper khác nhau là 2 câu trả lời khác nhau."""
        scope = {"document_ids": list(document_ids)} if document_ids else {}
        if self.cache is None:
            return AnswerSnapshot.from_rag(self.rag.ask(question, application_id, **scope))
        key = self.cache.key_for(application_id, question, document_ids)
        cached = self.cache.get(key)
        ANSWER_CACHE.labels("hit" if cached is not None else "miss").inc()   # Phase 12d
        if cached is not None:
            return cached
        snap = AnswerSnapshot.from_rag(self.rag.ask(question, application_id, **scope))   # LLM lỗi -> raise, không cache
        self.cache.put(key, application_id, question, snap)
        return snap

    def list_sessions(self, application_id: str, limit: int = 50) -> list[chat_repo.SessionSummary]:
        app_repo.require(self.db, application_id)
        return chat_repo.list_sessions(self.db, application_id, limit)

    def get_session(
        self, application_id: str, session_id: uuid.UUID,
        can_read: Callable[[ChatSession], bool] | None = None,
    ) -> tuple[ChatSession, list[Message]]:
        """can_read (Phase 12b): hàm quyết định người đang hỏi có được đọc phiên này không.
        Không được đọc -> 404 như phiên không tồn tại."""
        app_repo.require(self.db, application_id)
        session = chat_repo.get_session(self.db, application_id, session_id)
        if session is None or (can_read is not None and not can_read(session)):
            raise ChatSessionNotFoundError(f"Chat session {session_id} not found")
        return session, chat_repo.list_messages(self.db, application_id, session_id)