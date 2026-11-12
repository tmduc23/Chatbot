"""MỤC ĐÍCH: Phase 12b — hash mật khẩu (scrypt) và token đăng nhập (JWT). Không cần database."""
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core.config import Settings
from app.core.security import (
    InvalidTokenError, create_access_token, decode_access_token, hash_password, verify_password,
)
from app.services.auth_service import check_auth_config

SECRET = "test-secret-" + "x" * 40


def test_password_hash_roundtrip_and_salt() -> None:
    h1, h2 = hash_password("correct horse battery"), hash_password("correct horse battery")
    assert h1.startswith("scrypt$") and "correct horse" not in h1
    assert h1 != h2                                   # salt ngẫu nhiên: cùng mật khẩu, khác hash
    assert verify_password("correct horse battery", h1) and verify_password("correct horse battery", h2)
    assert not verify_password("correct horse batterY", h1)


@pytest.mark.parametrize("stored", [None, "", "garbage", "bcrypt$1$2$3$4$5", "scrypt$x$8$1$abc$def",
                                    "scrypt$16384$8$1$!!!$???"])
def test_verify_password_never_raises_on_bad_hash(stored: str | None) -> None:
    assert verify_password("anything", stored) is False


def test_access_token_roundtrip() -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id, SECRET, minutes=5)
    assert decode_access_token(token, SECRET) == user_id


def test_expired_token_rejected() -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    token = create_access_token(uuid.uuid4(), SECRET, minutes=60, now=old)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, SECRET)


def test_token_signed_with_other_secret_or_tampered_rejected() -> None:
    token = create_access_token(uuid.uuid4(), SECRET, minutes=5)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, "another-secret-" + "y" * 40)
    header, payload, signature = token.split(".")
    forged_claims = {"sub": str(uuid.uuid4()), "exp": 9999999999, "type": "access"}
    forged_payload = jwt.encode(forged_claims, "whatever-key-" + "z" * 40, algorithm="HS256").split(".")[1]
    with pytest.raises(InvalidTokenError):                 # đổi nội dung, giữ chữ ký cũ
        decode_access_token(f"{header}.{forged_payload}.{signature}", SECRET)


def test_alg_none_token_rejected() -> None:
    """Tấn công kinh điển: token "alg": "none" (không chữ ký). Thư viện PHẢI từ chối."""
    unsigned = jwt.encode({"sub": str(uuid.uuid4()), "exp": 9999999999, "type": "access"}, None, algorithm="none")
    with pytest.raises(InvalidTokenError):
        decode_access_token(unsigned, SECRET)


def test_token_without_type_or_bad_sub_rejected() -> None:
    no_type = jwt.encode({"sub": str(uuid.uuid4()), "exp": 9999999999}, SECRET, algorithm="HS256")
    bad_sub = jwt.encode({"sub": "admin", "exp": 9999999999, "type": "access"}, SECRET, algorithm="HS256")
    for token in (no_type, bad_sub, "not-a-jwt"):
        with pytest.raises(InvalidTokenError):
            decode_access_token(token, SECRET)


@pytest.mark.parametrize("secret", [None, "short"])
def test_auth_enabled_requires_strong_jwt_secret(secret: str | None) -> None:
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        check_auth_config(Settings(auth_enabled=True, jwt_secret=secret))
    check_auth_config(Settings(auth_enabled=False, jwt_secret=None))      # tắt đăng nhập: không bắt buộc
    check_auth_config(Settings(auth_enabled=True, jwt_secret=SECRET))     # đủ dài: OK
