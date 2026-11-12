"""
MỤC ĐÍCH (Phase 12d): STRUCTURED LOGGING — log là dữ liệu cho MÁY đọc, không chỉ là chữ cho người đọc.

    text (dev):  2026-10-18 10:00:01 INFO app.services.chat_service: chat app=cv ... total_ms=812
    json (prod): {"ts": "2026-10-18T10:00:01.123Z", "level": "INFO", "logger": "app.services.chat_service",
                  "message": "chat app=cv ... total_ms=812", "request_id": "3f9c2a1b7d4e4f0a"}

JSON: mỗi dòng là 1 object -> công cụ log (Loki, CloudWatch, Datadog, hoặc chỉ `jq`) lọc được theo trường:
"mọi log của request_id=3f9c..." , "mọi request status=500 hôm qua".

REQUEST ID: mỗi request 1 mã ngẫu nhiên, gắn vào MỌI dòng log trong lúc xử lý request đó, và trả về
cho client qua header X-Request-ID. Người dùng báo lỗi kèm mã -> grep ra đúng những dòng log liên quan,
dù cùng lúc có 100 request khác đang chạy.

CHE SECRET: log có thể bị đọc bởi nhiều người/hệ thống hơn database. Thư viện bên ngoài đôi khi in
cả URL có key, hoặc lỗi có chứa header. Mọi dòng log đi qua bộ lọc: giá trị của MỌI biến SecretStr
trong Settings (API key, JWT_SECRET...), token "Bearer ...", chuỗi dạng JWT -> "***".
"""
import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from pydantic import SecretStr

from app.core.config import Settings

# ContextVar: mỗi request (mỗi luồng xử lý) thấy giá trị CỦA RIÊNG NÓ, không lẫn với request khác
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_PATTERNS = [
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer ***"),
    (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+"), "***"),             # JWT: luôn bắt đầu bằng eyJ ('{"' base64)
    (re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+"), r"\1***"),
]
MIN_SECRET_LEN = 8   # chuỗi quá ngắn thì không che (tránh thay nhầm mọi chữ "abc" trong log)

# Thuộc tính sẵn có của LogRecord -> không đưa vào JSON như trường "extra"
# (color_message: bản có mã màu terminal mà uvicorn tự gắn vào log của nó -> bỏ)
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "request_id",
                                                                        "color_message"}


def secret_values(settings: Settings) -> list[str]:
    """Giá trị thật của mọi biến SecretStr trong Settings (đọc tự động -> thêm key mới không cần sửa ở đây)."""
    values = []
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            raw = value.get_secret_value().strip()
            if len(raw) >= MIN_SECRET_LEN:
                values.append(raw)
    return sorted(values, key=len, reverse=True)   # dài trước: key chứa key khác vẫn che đủ


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class TextFormatter(logging.Formatter):
    def __init__(self, secrets: list[str]):
        super().__init__("%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s")
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = None
        return redact(super().format(record), self.secrets)


class JsonFormatter(logging.Formatter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        # logger.info("request", extra={"status": 200, "duration_ms": 12}) -> thành trường riêng trong JSON
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(data, ensure_ascii=False, default=str), self.secrets)


def build_handler(settings: Settings, stream=None) -> logging.Handler:
    handler = logging.StreamHandler(stream or sys.stdout)
    secrets = secret_values(settings)
    handler.setFormatter(JsonFormatter(secrets) if settings.log_format == "json" else TextFormatter(secrets))
    handler.addFilter(RequestIdFilter())
    return handler


def configure_logging(settings: Settings) -> None:
    """Gọi 1 lần lúc khởi động (main.py, scripts/worker.py). Thay basicConfig của các phase trước."""
    root = logging.getLogger()
    root.handlers = [build_handler(settings)]
    root.setLevel(settings.log_level)
    # uvicorn có handler riêng in dạng text -> cho đi qua handler của mình để cùng 1 định dạng
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # access log của uvicorn bị thay bằng log "request" của middleware (có request_id, route, thời gian)
    logging.getLogger("uvicorn.access").disabled = True
