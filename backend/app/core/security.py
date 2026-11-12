"""
MỤC ĐÍCH (Phase 12b): 2 việc mật mã duy nhất của hệ thống — hash mật khẩu và ký/kiểm token đăng nhập.

1. Mật khẩu: lưu HASH scrypt, không lưu mật khẩu gốc.
   - scrypt CỐ Ý chậm và tốn RAM (~16 MB mỗi lần) -> kẻ trộm database thử hàng tỷ mật khẩu sẽ rất tốn kém.
     Không dùng sha256(password): GPU tính hàng tỷ sha256 mỗi giây.
   - Mỗi mật khẩu 1 "salt" ngẫu nhiên -> 2 người cùng mật khẩu vẫn có 2 hash khác nhau.
   - Có sẵn trong Python (hashlib.scrypt), không cần thêm thư viện.
   Chuỗi lưu trong DB:  scrypt$16384$8$1$<salt base64>$<hash base64>
   (ghi kèm tham số -> sau này tăng độ khó vẫn kiểm được mật khẩu cũ)

2. Token đăng nhập: JWT ký bằng HMAC-SHA256 (thư viện PyJWT).
   - Đăng nhập đúng -> server trả token {"sub": user_id, "exp": hết hạn} kèm chữ ký.
   - Mỗi request gửi lại header "Authorization: Bearer <token>". Server kiểm chữ ký bằng JWT_SECRET:
     ai sửa 1 ký tự trong token (vd. đổi sub thành id của admin) thì chữ ký không còn khớp.
   - Token KHÔNG mã hoá (ai cũng đọc được nội dung bằng base64) -> không bao giờ để bí mật trong token.
   - Chỉ chấp nhận đúng thuật toán HS256 (chặn tấn công "alg": "none").
"""
import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt

JWT_ALGORITHM = "HS256"
MIN_JWT_SECRET_CHARS = 32
MIN_PASSWORD_CHARS = 10

# Tham số scrypt: N = 2^14 (độ khó), r = 8, p = 1 -> ~16 MB RAM, ~50 ms mỗi lần hash
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN = 2**14, 8, 1, 32


class InvalidTokenError(Exception):
    """Token sai chữ ký, hết hạn, sai định dạng hoặc thiếu thông tin."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
                            dklen=_SCRYPT_DKLEN)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    """True nếu mật khẩu khớp hash. Hash hỏng / lạ / None -> False (không raise)."""
    if not stored:
        return False
    try:
        algo, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt_b64),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    # so sánh thời gian cố định: không lộ "đúng được bao nhiêu byte đầu"
    return hmac.compare_digest(actual, expected)


# Hash của 1 mật khẩu không ai biết. Đăng nhập bằng email KHÔNG tồn tại vẫn chạy verify với hash này
# -> thời gian trả lời như email có thật, kẻ tấn công không đo được email nào đã đăng ký.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(16))


def create_access_token(user_id: uuid.UUID, secret: str, minutes: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=minutes), "type": "access"}
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: str) -> uuid.UUID:
    """Kiểm chữ ký + hạn dùng, trả về user_id. Mọi lỗi -> InvalidTokenError."""
    try:
        claims = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM], options={"require": ["sub", "exp"]})
        if claims.get("type") != "access":
            raise InvalidTokenError("not an access token")
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidTokenError(str(exc)) from exc
