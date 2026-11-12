"""
MỤC ĐÍCH: Đổi lỗi nghiệp vụ -> HTTP status, ở MỘT chỗ duy nhất (Phase 9, mục 21)

Service raise lỗi có tên rõ ràng (ApplicationNotFoundError, DuplicateDocumentError...).
Route KHÔNG try/except. Handler ở đây bắt và trả {"detail": "..."} với đúng status code.

Không bao giờ trả về stack trace cho cleint: lỗi lạ -> 500 + câu chung chung, chi tiết chỉ nằm trong log.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError, ApplicationAlreadyExistsError, ApplicationNotFoundError, AuthDisabledError, AuthenticationError,
    ChatSessionNotFoundError, DocumentNotFoundError, DuplicateDocumentError, EmbeddingError, EmptyDocumentError,
    FileTooLargeError, IngestionError, InvalidQuestionError, LLMError, PermissionDeniedError, RateLimitExceededError,
    RerankError, UnsupportedFileTypeError, UserAlreadyExistsError, UserNotFoundError, WeakPasswordError,
)
from app.core.metrics import UPSTREAM_ERRORS as UPSTREAM_ERRORS_TOTAL   # tên UPSTREAM_ERRORS đã dùng ở dưới

logger = logging.getLogger(__name__)

# Thứ tự quan trọng: lớp CON trước lớp CHA (DuplicateDocumentError trước IngestionError)

STATUS_BY_ERROR: list[tuple[type[AppError], int]] = [
    (ApplicationNotFoundError, 404),
    (DocumentNotFoundError, 404),
    (ChatSessionNotFoundError, 404),
    (UserNotFoundError, 404),
    (AuthDisabledError, 404),
    (AuthenticationError, 401),       # Phase 12b
    (PermissionDeniedError, 403),     # Phase 12b
    (RateLimitExceededError, 429),    # Phase 12c
    (UserAlreadyExistsError, 409),

    (ApplicationAlreadyExistsError, 409),
    (DuplicateDocumentError, 409),
    (FileTooLargeError, 413),
    (UnsupportedFileTypeError, 415),
    (EmptyDocumentError, 422),
    (InvalidQuestionError, 422),
    (WeakPasswordError, 422),
    (IngestionError, 422),
]

# Lỗi do dịch vụ bên ngoài (Gemini, OpenRouter): không phải lỗi của người dùng, thường tự hết sau ít phút.
# Không đưa str(exc) cho client: có thể chứa tên model, URL, thông tin tài khoản của provider.

UPSTREAM_ERRORS = (LLMError, EmbeddingError, RerankError)
UPSTREAM_MESSAGE = "AI service is temporarily unavailable. Please try again in a moment."

def status_for(exc: AppError) -> int:
    for error_type, code in STATUS_BY_ERROR:
        if isinstance(exc, error_type):             # (obj, classinfo): (biến cần kiểm tra, kiểu dữ liệu)
            return code
    return 400

# xử lý lỗi nghiệp vụ (AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    if isinstance(exc, UPSTREAM_ERRORS):
        logger.warning("upstream error on %s %s: %s", request.method, request.url.path, exc)
        UPSTREAM_ERRORS_TOTAL.labels(type(exc).__name__).inc()   # Phase 12d: đếm để cảnh báo khi LLM lỗi nhiều
        return JSONResponse(status_code=503, content={"detail": UPSTREAM_MESSAGE})
    code = status_for(exc)
    logger.info("%s %s -> %d %s", request.method, request.url.path, code, exc)
        # Chuẩn HTTP: 401 phải nói client cần đăng nhập KIỂU GÌ (ở đây: Bearer token)
    headers = {"WWW-Authenticate": "Bearer"} if code == 401 else None
    if isinstance(exc, RateLimitExceededError):
        headers = {"Retry-After": str(exc.retry_after)}   # chuẩn HTTP: client biết chờ bao lâu

    return JSONResponse(status_code=code, content={"detail": str(exc)}, headers=headers)



# bắt lỗi không lường trước được ví dụ KeyError 
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register_error_handlers(app: FastAPI) -> None:
    # AppError bắt được mọi lớp con -> chỉ cần đăng ký 1 lần
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)