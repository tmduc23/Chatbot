"""Route /api/v1/auth (Phase 12b): đăng nhập lấy token, xem "tôi là ai".

Không có route đăng ký (sign up): tài khoản do bạn tạo bằng scripts/manage_users.py.
Hệ thống nội bộ không cần người lạ tự tạo tài khoản -> bớt hẳn 1 cửa để tấn công.
"""
from fastapi import APIRouter, Depends

from app.api.deps import RateLimit, get_auth_service, get_principal
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthDisabledError, AuthenticationError
from app.schemas.auth import LoginRequest, MembershipOut, MeOut, TokenResponse
from app.services.auth_service import AuthService, Principal

router = APIRouter(prefix="/auth", tags=["auth"])


def _require_enabled(settings: Settings) -> None:
    if not settings.auth_enabled:
        raise AuthDisabledError("Authentication is disabled (AUTH_ENABLED=false)")


# Phase 12c: giới hạn số lần thử theo IP -> dò mật khẩu hàng triệu lần là không thể
@router.post("/login", response_model=TokenResponse, dependencies=[Depends(RateLimit("login"))])
def login(body: LoginRequest, auth: AuthService = Depends(get_auth_service),
          settings: Settings = Depends(get_settings)):
    _require_enabled(settings)
    result = auth.login(body.email, body.password)
    return TokenResponse(access_token=result.access_token, expires_in=result.expires_in)


@router.get("/me", response_model=MeOut)
def me(principal: Principal = Depends(get_principal), settings: Settings = Depends(get_settings)):
    _require_enabled(settings)
    if principal.kind != "user":
        raise AuthenticationError("Login required")
    return MeOut(
        id=principal.user_id, email=principal.email, is_superadmin=principal.is_superadmin,
        memberships=[MembershipOut(application_id=a, role=r) for a, r in sorted(principal.roles.items())],
    )
