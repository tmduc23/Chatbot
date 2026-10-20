"""
MỤC ĐÍCH: Các "dependency" của FASTAPI (Phase 9) - route khai báo cần gì, FastAPI tự tạo và truyền vào.

    def chat(body: ChatRequest, service: ChatService = Depends(get_chat_service)): ...

Lợi ích: 
    - Route không tự tạo DB session, không tự độc .env, không biết Gemini hay OpenRouter.
    - Test thay bằng bản giả: app.dependency_overides[get_llm_provider] = lambda: FakeLLM()
    -> test toàn bộ API mà không tốn 1 request nào của Gemini/OpenRouter

Provider (embedding, reranker, LLM) tạo 1 LẦN rồi dùng chung (lru_cache): tạo client SDK mỗi request
là lãng phí. DB session thì ngược lại: MỖI request 1 session riêng (get_db), đóng khi xong
"""

import secrets
from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.llm import LLMProvider, get_llm
from app.rag.question_condenser import QuestionCondenser
from app.rag.reranking import Reranker, get_reranker
from app.services.application_service import ApplicationService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService, build_rag_service


__all__ = ["get_db"]


# Provider dùng chung (tạo khi request lần đầu tiên cần đến)

@lru_cache 
def get_embedder() -> EmbeddingProvider:
    return get_embedding_provider(get_settings())

@lru_cache
def get_reranker_provider() -> Reranker:
    return get_reranker(get_settings())

@lru_cache
def get_llm_provider() -> LLMProvider:
    return get_llm(get_settings())


def get_condenser(
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
) -> QuestionCondenser | None:
    return QuestionCondenser(llm) if settings.chat_condense_question else None


# Service (mỗi request 1 bản, gắn với DB session của request đó)

def get_application_service(db: Session = Depends(get_db)) -> ApplicationService:
    return ApplicationService(db)


def get_document_service(
    db: Session = Depends(get_db),
    embedder: EmbeddingProvider = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    return DocumentService(db, settings=settings, embedder=embedder)

def get_rag_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    embedder: EmbeddingProvider = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker_provider),
    llm: LLMProvider = Depends(get_llm_provider),
) -> RAGService:
    return build_rag_service(db, settings, embedder, reranker, llm)

def get_chat_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    rag: RAGService = Depends(get_rag_service),
    condenser: QuestionCondenser | None = Depends(get_condenser),
) -> ChatService:
    return ChatService(db, rag, condenser, history_turns=settings.chat_history_turns)


# Bảo vệ API quản trị (tạm thời, đến Phase 12 có đăng nhập thật)

def require_admin(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:

    """
    Tạo application, upload/xem/xóa tài liệu, xem MỌI phiên chat: chỉ admin.

    - Có ADMIN_API_KEY trong .env -> header X-API-KEY phải khớp, sai/thiếu -> 401.
    - Không có ADMIN_API_KEY      -> chỉ cho phép khi APP_ENV=development (máy local)
                                    môi trường khác -> 403 (quên đặt key thì KHÓA, không MỞ).
    """
    expected = settings.admin_api_key.get_secret_value() if settings.admin_api_key else ""
    if not expected:
        if settings.app_env == "development":
            return
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API is disabled: ADMIN_API_KEY is not set")
    # compare_digest: so sánh mất thời gian như nhau dù sai ở ký tự nào -> không đoán dần được key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-KEY")

    