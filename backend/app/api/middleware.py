"""
MỤC ĐÍCH (Phase 12d): việc chung cho MỌI request, viết 1 lần ở middleware thay vì lặp ở từng route.

    request ─> [request_id] ─> route ─> response
                    │                       │
                    └── đo thời gian ───────┴─> log "request" + metrics + header bảo mật + X-Request-ID
"""
import logging
import re
import time
import uuid

from fastapi import FastAPI, Request, Response

from app.core.logging import request_id_var
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = logging.getLogger("app.request")

# Chấp nhận X-Request-ID client gửi lên (vd. frontend, proxy) nếu "sạch": chặn chèn ký tự xuống dòng,
# chuỗi dài 10 KB vào log (log injection)
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Header bảo mật cho response của API (JSON). OWASP Secure Headers.
API_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",          # trình duyệt không "đoán" JSON là HTML/script
    "X-Frame-Options": "DENY",                    # không cho trang khác nhúng vào iframe (clickjacking)
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",   # JSON không cần tải gì cả
    "Cache-Control": "no-store",                  # câu trả lời, danh sách tài liệu: không cho proxy/trình duyệt lưu
}


def _route_template(request: Request) -> str:
    """"/api/v1/documents/{document_id}" thay vì "/api/v1/documents/8f3a..." (giữ metrics nhỏ, không lộ id).

    route.path có thể thiếu prefix "/api/v1" (tuỳ phiên bản FastAPI gắn router con thế nào) -> lấy phần
    đầu từ URL thật, phần sau (có {tham_số}) từ route.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not template:
        return "unmatched"   # 404 không khớp route nào: gộp chung 1 nhãn, không để URL lạ thành nhãn
    actual = request.url.path.rstrip("/").split("/")
    pattern = template.rstrip("/").split("/")
    prefix = actual[: max(1, len(actual) - len(pattern) + 1)]
    return "/".join(prefix + pattern[1:]) or "/"


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe(request: Request, call_next) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = _route_template(request)
            if route != "/metrics":   # Prometheus hỏi mỗi 15s -> không tự đếm chính nó
                HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                HTTP_LATENCY.labels(request.method, route).observe(elapsed)
                logger.info("request %s %s -> %d in %.0f ms", request.method, route, status, elapsed * 1000,
                            extra={"method": request.method, "route": route, "status": status,
                                   "duration_ms": round(elapsed * 1000, 1),
                                   "user": getattr(request.state, "user", None)})
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/api/"):
            for name, value in API_SECURITY_HEADERS.items():
                response.headers.setdefault(name, value)
        return response
