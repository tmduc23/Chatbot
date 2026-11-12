"""Pydantic schema cho /api/v1/auth (Phase 12b)."""
import uuid

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320, examples=["duc@example.com"])
    # max_length: chặn gửi mật khẩu 10 MB để bắt server chạy scrypt trên chuỗi khổng lồ
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # giây


class MembershipOut(BaseModel):
    application_id: str
    role: str


class MeOut(BaseModel):
    id: uuid.UUID
    email: str
    is_superadmin: bool
    memberships: list[MembershipOut]
