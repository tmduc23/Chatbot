"""
MỤC ĐÍCH (Phase 12b): AUTHENTICATION — "bạn là ai?".

    POST /auth/login {email, password} -> kiểm mật khẩu -> token (JWT, hết hạn sau ACCESS_TOKEN_MINUTES)
    Mỗi request: header "Authorization: Bearer <token>" -> principal_from_token -> Principal (ai, vai trò gì)

Principal = "người đang gọi API" sau khi đã xác thực. 3 loại:
  - anonymous : không gửi gì (khách xem CV)
  - user      : đăng nhập bằng email + mật khẩu
  - service   : gửi X-API-Key đúng (script, cron, curl của bạn) = quyền superadmin

Vai trò đọc lại từ DB ở MỖI request (không nhét vào token): thu hồi quyền hay khoá tài khoản
có hiệu lực ngay, không phải chờ token hết hạn.
Phần "được làm gì" (authorization) nằm ở app/services/access.py.
"""
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    AuthenticationError, UserAlreadyExistsError, UserNotFoundError, WeakPasswordError,
)
from app.core.security import (
    DUMMY_PASSWORD_HASH, MIN_JWT_SECRET_CHARS, MIN_PASSWORD_CHARS, InvalidTokenError, create_access_token,
    decode_access_token, hash_password, verify_password,
)
from app.models import ROLES, User
from app.repositories import application_repository as app_repo
from app.repositories import user_repository as user_repo

INVALID_LOGIN = "Invalid email or password"   # CÙNG 1 câu cho sai email và sai mật khẩu


@dataclass(frozen=True)
class Principal:
    kind: Literal["anonymous", "user", "service"]
    user_id: uuid.UUID | None = None
    email: str | None = None
    is_superadmin: bool = False
    roles: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))   # {application_id: role}

    @property
    def is_admin(self) -> bool:
        """Làm được mọi việc trên mọi application."""
        return self.kind == "service" or self.is_superadmin

    def role_in(self, application_id: str) -> str | None:
        return self.roles.get(application_id)


ANONYMOUS = Principal(kind="anonymous")
SERVICE = Principal(kind="service")


@dataclass(frozen=True)
class LoginResult:
    access_token: str
    expires_in: int     # giây
    user: User


def check_auth_config(settings: Settings) -> None:
    """Gọi lúc backend khởi động: bật đăng nhập mà quên/đặt JWT_SECRET yếu -> KHÔNG chạy (fail fast)."""
    if not settings.auth_enabled:
        return
    secret = settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""
    if len(secret) < MIN_JWT_SECRET_CHARS:
        raise RuntimeError(
            f"AUTH_ENABLED=true requires JWT_SECRET with at least {MIN_JWT_SECRET_CHARS} characters. "
            'Generate one: python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_CHARS:
        raise WeakPasswordError(f"Password must be at least {MIN_PASSWORD_CHARS} characters")


class AuthService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    def _secret(self) -> str:
        check_auth_config(self.settings)
        if self.settings.jwt_secret is None:
            raise RuntimeError("JWT_SECRET is not set")
        return self.settings.jwt_secret.get_secret_value()

    # ---- đăng nhập ----

    def login(self, email: str, password: str) -> LoginResult:
        user = user_repo.get_by_email(self.db, email)
        # Luôn chạy verify (kể cả email không tồn tại) -> thời gian trả lời như nhau
        ok = verify_password(password, user.password_hash if user else DUMMY_PASSWORD_HASH)
        if user is None or not ok or not user.is_active:
            raise AuthenticationError(INVALID_LOGIN)
        minutes = self.settings.access_token_minutes
        token = create_access_token(user.id, self._secret(), minutes)
        return LoginResult(access_token=token, expires_in=minutes * 60, user=user)

    def principal_from_token(self, token: str) -> Principal:
        try:
            user_id = decode_access_token(token, self._secret())
        except InvalidTokenError:
            raise AuthenticationError("Invalid or expired token") from None
        user = user_repo.get(self.db, user_id)
        if user is None or not user.is_active:
            # token còn hạn nhưng tài khoản đã bị xoá/khoá -> từ chối ngay
            raise AuthenticationError("Invalid or expired token")
        return Principal(
            kind="user", user_id=user.id, email=user.email, is_superadmin=user.is_superadmin,
            roles=MappingProxyType(user_repo.roles_of(self.db, user.id)),
        )

    # ---- quản lý user (scripts/manage_users.py gọi; mỗi hàm tự commit) ----

    def create_user(self, email: str, password: str, name: str | None = None,
                    is_superadmin: bool = False) -> User:
        _check_password(password)
        if user_repo.get_by_email(self.db, email) is not None:
            raise UserAlreadyExistsError(f"User '{user_repo.normalize_email(email)}' already exists")
        try:
            user = user_repo.create(self.db, email, hash_password(password), name, is_superadmin)
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raise UserAlreadyExistsError(f"User '{user_repo.normalize_email(email)}' already exists") from None
        return user

    def require_user(self, email: str) -> User:
        user = user_repo.get_by_email(self.db, email)
        if user is None:
            raise UserNotFoundError(f"User '{user_repo.normalize_email(email)}' not found")
        return user

    def set_password(self, email: str, password: str) -> None:
        _check_password(password)
        self.require_user(email).password_hash = hash_password(password)
        self.db.commit()

    def set_active(self, email: str, active: bool) -> None:
        self.require_user(email).is_active = active
        self.db.commit()

    def grant(self, email: str, application_id: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        user = self.require_user(email)
        app_repo.require(self.db, application_id)
        user_repo.set_role(self.db, application_id, user.id, role)
        self.db.commit()

    def revoke(self, email: str, application_id: str) -> bool:
        removed = user_repo.remove_role(self.db, application_id, self.require_user(email).id)
        self.db.commit()
        return removed

    def set_public(self, application_id: str, public: bool) -> None:
        app_repo.require(self.db, application_id).is_public = public
        self.db.commit()
