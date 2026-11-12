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

from functools import lru_cache

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, RateLimitExceededError
from app.core.metrics import RATE_LIMITED
from app.db.session import get_db
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.embeddings.cache import CachedEmbeddingProvider

from app.rag.llm import LLMProvider, get_llm
from app.rag.question_condenser import QuestionCondenser
from app.rag.reranking import Reranker, get_reranker
from app.services.access import AccessControl, api_key_matches
from app.services.answer_cache import AnswerCache

from app.services.application_service import ApplicationService
from app.services.auth_service import ANONYMOUS, SERVICE, AuthService, Principal
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService, build_rag_service
from app.services.rate_limiter import RateLimiter



__all__ = ["get_db"]


# Provider dùng chung (tạo khi request lần đầu tiên cần đến)

@lru_cache 
def get_embedder() -> EmbeddingProvider:
    settings = get_settings()
    provider = get_embedding_provider(settings)
    # Phase 12c: câu hỏi lặp lại -> lấy vector từ RAM, không gọi API embedding
    if settings.embedding_cache_size > 0:
        return CachedEmbeddingProvider(provider, settings.embedding_cache_size)
    return provider


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
    cache = AnswerCache(db, settings) if settings.answer_cache_enabled else None   # Phase 12c
    return ChatService(db, rag, condenser, history_turns=settings.chat_history_turns, cache=cache)


# ---- Đăng nhập + phân quyền (Phase 12b) ----
# Thay cho require_admin của Phase 9: quy tắc "ai được làm gì" nằm ở app/services/access.py

# auto_error=False: không có header Authorization thì trả None (khách), không tự báo lỗi.
# Swagger (/docs) nhờ đó hiện nút "Authorize" để dán token.
bearer_scheme = HTTPBearer(auto_error=False)


def get_auth_service(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> AuthService:
    return AuthService(db, settings)


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    auth: AuthService = Depends(get_auth_service),
) -> Principal:
    """Ai đang gọi API? Gửi thông tin đăng nhập SAI thì báo 401 ngay, không coi như khách."""
    principal = _resolve_principal(credentials, x_api_key, settings, auth)
    # Phase 12d: middleware ghi vào log "request" (user id, KHÔNG ghi email -> log ít dữ liệu cá nhân hơn)
    request.state.user = str(principal.user_id) if principal.user_id else principal.kind
    return principal


def _resolve_principal(credentials: HTTPAuthorizationCredentials | None, x_api_key: str | None,
                       settings: Settings, auth: AuthService) -> Principal:
    if not settings.auth_enabled:
        return ANONYMOUS   # chế độ Phase 9: không có user, AccessControl tự kiểm X-API-Key
    if x_api_key is not None:
        if api_key_matches(x_api_key, settings):
            return SERVICE
        raise AuthenticationError("Invalid X-API-Key")
    if credentials is None:
        return ANONYMOUS
    return auth.principal_from_token(credentials.credentials)


def get_access(
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None),
) -> AccessControl:
    return AccessControl(db, principal, settings, x_api_key=x_api_key)


# ---- Rate limiting (Phase 12c) ----

def client_ip(request: Request, settings: Settings) -> str:
    """IP của người gọi. X-Forwarded-For CHỈ tin khi backend đứng sau proxy của chính mình:
    header này ai cũng tự đặt được -> tin bừa thì kẻ spam đổi IP giả mỗi request là né được giới hạn."""
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[0].strip()   # IP đầu tiên = client gốc (proxy nối thêm IP phía sau)
    return request.client.host if request.client else "unknown"


def _limits(scope: str, s: Settings) -> list[tuple[int, int]]:
    """(số request tối đa, độ dài cửa sổ tính bằng giây) cho từng loại API."""
    return {
        "chat": [(s.rate_limit_chat_per_minute, 60), (s.rate_limit_chat_per_day, 86400)],
        "login": [(s.rate_limit_login_per_minute, 60)],
        "upload": [(s.rate_limit_upload_per_minute, 60)],
    }[scope]


class RateLimit:
    """Dependency: `dependencies=[Depends(RateLimit("chat"))]` trên route.

    Đếm theo user nếu đã đăng nhập (đổi IP không né được), theo IP nếu là khách.
    X-API-Key (script của bạn, eval) không bị giới hạn.
    """

    def __init__(self, scope: str):
        self.scope = scope

    def __call__(
        self,
        request: Request,
        principal: Principal = Depends(get_principal),
        settings: Settings = Depends(get_settings),
        db: Session = Depends(get_db),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not settings.rate_limit_enabled:
            return
        # AUTH_ENABLED=false thì principal luôn là khách -> kiểm key trực tiếp
        if principal.kind == "service" or api_key_matches(x_api_key, settings):
            return
        who = f"user:{principal.user_id}" if principal.user_id else f"ip:{client_ip(request, settings)}"
        limiter = RateLimiter(db)
        for limit, window in _limits(self.scope, settings):
            result = limiter.hit(f"{self.scope}:{who}:{window}", limit, window)
            if not result.allowed:
                RATE_LIMITED.labels(self.scope).inc()   # Phase 12d
                raise RateLimitExceededError(
                    f"Too many requests. Try again in {result.retry_after} seconds.", retry_after=result.retry_after
                )


    