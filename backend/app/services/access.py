"""
MỤC ĐÍCH (Phase 12b): AUTHORIZATION — "bạn được làm gì, trên application nào?" (spec mục 25).

Quy tắc, viết ở MỘT chỗ duy nhất:

    việc                                   ai được làm
    -------------------------------------  ---------------------------------------------------------
    tạo application                        superadmin / X-API-Key
    xem + chat trên application            app công khai: tất cả | app riêng tư: viewer, editor, admin
    upload/xem/xoá tài liệu, xem MỌI phiên  editor của application đó, admin
    xem 1 phiên chat                       chủ phiên chat, editor, admin; phiên của khách = ai có link

Không bao giờ tin application_id từ client: route nào nhận application_id cũng phải gọi
require_chat / require_editor TRƯỚC khi gọi service.

Mã lỗi:
  401: chưa đăng nhập (khách) mà việc này cần đăng nhập
  403: đã đăng nhập, application CÓ thấy được (thành viên hoặc công khai) nhưng vai trò không đủ
  404: application riêng tư mà bạn không phải thành viên -> trả lời như KHÔNG TỒN TẠI,
       để người ngoài không dò được tên các application riêng tư (vd. "hr-salary")

AUTH_ENABLED=false: giữ nguyên Phase 9 — API quản trị dùng X-API-Key (hoặc mở khi development
chưa đặt ADMIN_API_KEY), chat không cần đăng nhập.
"""
import secrets
from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ApplicationNotFoundError, AuthenticationError, PermissionDeniedError
from app.models import ROLE_EDITOR, Application, ChatSession
from app.repositories import application_repository as app_repo
from app.services.auth_service import Principal


def api_key_matches(x_api_key: str | None, settings: Settings) -> bool:
    """X-API-Key gửi lên có khớp ADMIN_API_KEY trong .env không. Chưa đặt ADMIN_API_KEY -> luôn False."""
    expected = settings.admin_api_key.get_secret_value() if settings.admin_api_key else ""
    # compare_digest: so sánh mất thời gian như nhau dù sai ở ký tự nào -> không đoán dần được key
    return bool(expected and x_api_key and secrets.compare_digest(x_api_key, expected))


class AccessControl:
    def __init__(self, db: Session, principal: Principal, settings: Settings, x_api_key: str | None = None):
        self.db = db
        self.principal = principal
        self.settings = settings
        self.x_api_key = x_api_key

    @property
    def enabled(self) -> bool:
        return self.settings.auth_enabled

    def _legacy_admin(self) -> None:
        """AUTH_ENABLED=false: đúng hành vi require_admin của Phase 9."""
        expected = self.settings.admin_api_key.get_secret_value() if self.settings.admin_api_key else ""
        if not expected:   # "ADMIN_API_KEY=" để trống trong .env cũng là chưa đặt
            if self.settings.app_env == "development":
                return
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin API is disabled: ADMIN_API_KEY is not set")
        if not api_key_matches(self.x_api_key, self.settings):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")

    def _login_required(self) -> None:
        if self.principal.kind == "anonymous":
            raise AuthenticationError("Login required")

    def _app(self, application_id: str) -> Application | None:
        return app_repo.get(self.db, application_id)

    def _not_found(self, application_id: str) -> ApplicationNotFoundError:
        return ApplicationNotFoundError(f"Application '{application_id}' not found")

    # ---- không raise: dùng để lọc / quyết định hiển thị ----

    def can_view(self, app: Application) -> bool:
        if not self.enabled or self.principal.is_admin:
            return True
        return app.is_public or self.principal.role_in(app.id) is not None

    def is_editor(self, application_id: str) -> bool:
        if not self.enabled:
            return True   # chế độ cũ: route đã chặn bằng X-API-Key trước khi tới đây
        return self.principal.is_admin or self.principal.role_in(application_id) == ROLE_EDITOR

    def filter_visible(self, apps: Iterable[Application]) -> list[Application]:
        return [a for a in apps if self.can_view(a)]

    def can_read_session(self, session: ChatSession) -> bool:
        if session.user_id is None:
            return True   # phiên chat của khách: ai có session_id (UUID không đoán được) thì xem được
        return session.user_id == self.principal.user_id or self.is_editor(session.application_id)

    # ---- raise: đặt ở đầu route ----

    def require_superadmin(self) -> None:
        if not self.enabled:
            return self._legacy_admin()
        self._login_required()
        if not self.principal.is_admin:
            raise PermissionDeniedError("Superadmin required")

    def require_chat(self, application_id: str) -> None:
        """Xem application + hỏi đáp trên nó."""
        if not self.enabled or self.principal.is_admin:
            return   # application không tồn tại -> service tự trả 404
        app = self._app(application_id)
        if app is not None and self.can_view(app):
            return
        self._login_required()             # khách -> 401 (không nói app có tồn tại hay không)
        raise self._not_found(application_id)   # đã đăng nhập nhưng không phải thành viên -> 404

    def require_editor(self, application_id: str) -> None:
        """Quản lý tài liệu, xem mọi phiên chat của application."""
        if not self.enabled:
            return self._legacy_admin()
        if self.principal.is_admin:
            return
        self._login_required()
        if self.principal.role_in(application_id) == ROLE_EDITOR:
            return
        app = self._app(application_id)
        if app is not None and self.can_view(app):
            raise PermissionDeniedError(f"Editor role required on application '{application_id}'")
        raise self._not_found(application_id)
