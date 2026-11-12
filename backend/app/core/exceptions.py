"""
MỤC ĐÍCH: Định nghĩa các lỗi nghiệp vụ (business errors) của hệ thống.

Thay vì raise Exception chung chung, mỗi tình huống có một class riêng.
Nhờ vậy:
  - script/CLI biết lỗi nào cần bỏ qua (vd. file trùng) và lỗi nào cần báo;
  - Phase 9, API sẽ map mỗi lỗi sang đúng HTTP status (404, 409, 415, 422...).
"""

class AppError(Exception):
    """Lỗi gốc của ứng dụng"""

class ApplicationNotFoundError(AppError):
    """application_id không tồn tại.                     -> HTTP 404"""

class IngestionError(AppError):
    """Lỗi gốc cho mọi vấn đề khi nạp tài liệu."""

class UnsupportedFileTypeError(IngestionError):
    """Định dạng file không hỗ trợ.                       -> HTTP 415"""

class FileTooLargeError(IngestionError):
    """File vượt quá dung lượng cho phép.                 -> HTTP 413"""

class DuplicateDocumentError(IngestionError):
    """File (cùng SHA-256) đã có trong application.       -> HTTP 409"""

class EmptyDocumentError(IngestionError):
    """Không trích được chữ nào (vd. PDF scan dạng ảnh).  -> HTTP 422"""

class EmbeddingError(AppError):
    """Gọi embedding provider thất bại (sai API key, hết quota, mạng...). -> HTTP 429/502/503"""

class RerankError(AppError):
    """Reranker thất bại (API lỗi, trả về sai định dạng ...)"""

class LLMError(AppError):
    """Gọi LLM thất bại (sai key, hết quota, bị chặn, trả về rỗng ...) -> HTTP 503 ở Phase 9"""

class ApplicationAlreadyExistsError(AppError):
    """Tạo application với id đã có.                     -> HTTP 409"""

class DocumentNotFoundError(AppError):
    """document_id không tồn tại TRONG application này.  -> HTTP 404"""

class ChatSessionNotFoundError(AppError):
    """session_id không tồn tại TRONG application này.   -> HTTP 404"""

class InvalidQuestionError(AppError, ValueError):
    """Câu hỏi rỗng hoặc quá dài.                         -> HTTP 422
    Kế thừa thêm ValueError để code/test cũ (pytest.raises(ValueError)) vẫn đúng."""

# ---- Đăng nhập + phân quyền (Phase 12b) ----

class AuthenticationError(AppError):
    """Chưa đăng nhập, token sai/hết hạn, sai email hoặc mật khẩu.  -> HTTP 401 ("bạn là ai?")"""


class PermissionDeniedError(AppError):
    """Đã đăng nhập nhưng không đủ vai trò cho việc này.          -> HTTP 403 ("bạn không được làm việc này")"""


class AuthDisabledError(AppError):
    """Gọi /auth/* khi AUTH_ENABLED=false.                        -> HTTP 404"""


class UserAlreadyExistsError(AppError):
    """Tạo user với email đã có.                                  -> HTTP 409"""


class UserNotFoundError(AppError):
    """Email không tồn tại (dùng trong scripts/manage_users.py)."""


class WeakPasswordError(AppError, ValueError):
    """Mật khẩu quá ngắn.                                         -> HTTP 422"""

# ---- Rate limiting (Phase 12c) ----

class RateLimitExceededError(AppError):
    """Gọi quá nhiều lần trong 1 khoảng thời gian.                 -> HTTP 429 + header Retry-After"""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after   # số giây nên chờ trước khi thử lại

