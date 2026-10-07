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