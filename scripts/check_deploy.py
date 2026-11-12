"""
MỤC ĐÍCH (Phase 13): kiểm tra 1 bản deploy TỪ BÊN NGOÀI, đúng như nhà tuyển dụng (và kẻ tấn công) nhìn thấy.

    python scripts/check_deploy.py https://cv.example.com
    python scripts/check_deploy.py https://cv.example.com --app cv --ask "Bạn ấy học trường nào?"   # +1 lần gọi LLM
    python scripts/check_deploy.py https://localhost --insecure        # thử trên máy, chứng chỉ tự ký của Caddy

Chỉ dùng thư viện chuẩn của Python (urllib, socket, ssl): chạy được ở bất kỳ máy nào có Python 3.11+,
không cần cài gì, không cần .env. Không gửi API key nào. Mã thoát: 0 = mọi kiểm tra đạt, 1 = có kiểm tra trượt.

Mỗi kiểm tra là 1 hàm nhận `http` (gửi request) và trả về Check. Test truyền `http` giả hoặc TestClient,
nên các kiểm tra được test mà không cần server thật.
"""
import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]     # tên header viết thường
    body: str


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


# (method, url, headers, body) -> Response. Không tự đi theo redirect: redirect cũng là thứ cần kiểm tra.
Http = Callable[[str, str, dict[str, str], bytes | None], Response]
PortOpen = Callable[[str, int], bool]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def urllib_http(insecure: bool = False, timeout: float = 15.0) -> Http:
    context = ssl.create_default_context()
    if insecure:   # CHỈ để thử trên máy với chứng chỉ tự ký; server thật phải đạt khi KHÔNG có --insecure
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=context))

    def send(method: str, url: str, headers: dict[str, str], body: bytes | None) -> Response:
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as r:
                return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:   # 3xx/4xx/5xx vẫn là 1 câu trả lời hợp lệ để kiểm tra
            return Response(e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode("utf-8", "replace"))
    return send


def tcp_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------- các kiểm tra ----------------

def check_https_redirect(http: Http, base: str) -> Check:
    parts = urlsplit(base)
    if parts.scheme != "https":
        return Check("http -> https", False, "URL phải bắt đầu bằng https://")
    plain = urlunsplit(("http", parts.netloc, "/", "", ""))
    r = http("GET", plain, {}, None)
    location = r.headers.get("location", "")
    ok = r.status in (301, 302, 307, 308) and location.startswith("https://")
    return Check("http -> https", ok, f"{r.status} Location={location or '-'}")


def check_health(http: Http, base: str) -> Check:
    r = http("GET", f"{base}/health", {}, None)
    return Check("/health", r.status == 200, f"{r.status} {r.body[:60]}")


def check_ready(http: Http, base: str) -> Check:
    r = http("GET", f"{base}/ready", {}, None)
    return Check("/ready (backend thấy database)", r.status == 200, f"{r.status} {r.body[:60]}")


def check_hsts(http: Http, base: str) -> Check:
    value = http("GET", f"{base}/health", {}, None).headers.get("strict-transport-security", "")
    return Check("HSTS", "max-age=" in value, value or "thiếu header Strict-Transport-Security")


def check_api_headers(http: Http, base: str) -> Check:
    r = http("GET", f"{base}/api/v1/applications", {}, None)
    missing = [h for h in ("x-content-type-options", "x-frame-options", "x-request-id") if h not in r.headers]
    return Check("API trả lời + security headers", r.status == 200 and not missing,
                 f"{r.status}" + (f", thiếu {', '.join(missing)}" if missing else ""))


def check_no_server_banner(http: Http, base: str) -> Check:
    server = http("GET", f"{base}/health", {}, None).headers.get("server", "")
    return Check("không lộ tên server", "uvicorn" not in server.lower(), f"Server: {server or '(không có)'}")


def check_hidden_paths(http: Http, base: str) -> Check:
    """/docs, /openapi.json: bản đồ API. /metrics: số liệu nội bộ. Không được đọc từ Internet."""
    leaked = []
    for path in ("/docs", "/openapi.json", "/metrics"):
        r = http("GET", f"{base}{path}", {}, None)
        if r.status == 200 and ("swagger" in r.body.lower() or '"openapi"' in r.body or "# HELP" in r.body):
            leaked.append(path)
    return Check("/docs, /openapi.json, /metrics bị ẩn", not leaked, "lộ: " + ", ".join(leaked) if leaked else "")


def check_cors_rejects_other_sites(http: Http, base: str) -> Check:
    evil = "https://evil.example"
    r = http("OPTIONS", f"{base}/api/v1/chat",
             {"Origin": evil, "Access-Control-Request-Method": "POST"}, None)
    allowed = r.headers.get("access-control-allow-origin", "")
    return Check("CORS chặn trang web lạ", allowed not in (evil, "*"), f"allow-origin={allowed or '-'}")


def check_admin_needs_login(http: Http, base: str) -> Check:
    """Xoá 1 document KHÔNG tồn tại: cấu hình đúng -> 401/403; cấu hình hở -> 404. Không bao giờ xoá nhầm gì."""
    r = http("DELETE", f"{base}/api/v1/documents/{uuid.uuid4()}?application_id=cv", {}, None)
    return Check("API quản trị đòi đăng nhập", r.status in (401, 403), f"{r.status}")


def check_ports_closed(port_open: PortOpen, base: str, ports: tuple[int, ...] = (5432, 8000, 3000, 9090)) -> Check:
    """Từ Internet chỉ được thấy 80/443 (Caddy). Postgres, backend, frontend, Prometheus phải nằm sau."""
    host = urlsplit(base).hostname or ""
    opened = [p for p in ports if port_open(host, p)]
    return Check("chỉ mở 80/443", not opened, "đang mở: " + ", ".join(map(str, opened)) if opened else "")


def check_chat(http: Http, base: str, app: str, question: str) -> Check:
    body = json.dumps({"application_id": app, "query": question}).encode()
    r = http("POST", f"{base}/api/v1/chat", {"Content-Type": "application/json"}, body)
    if r.status != 200:
        return Check(f"chat với app '{app}'", False, f"{r.status} {r.body[:80]}")
    data = json.loads(r.body)
    return Check(f"chat với app '{app}'", bool(data.get("answer")),
                 f"{len(data.get('sources', []))} nguồn, abstained={data.get('abstained')}")


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def run_checks(http: Http, base: str, port_open: PortOpen | None = None, app: str | None = None,
               question: str | None = None, behind_proxy: bool = True) -> Report:
    """behind_proxy=False: chỉ chạy kiểm tra của backend (test dùng TestClient, không có Caddy/HTTPS)."""
    base = base.rstrip("/")
    report = Report()
    if behind_proxy:
        report.checks += [check_https_redirect(http, base), check_hsts(http, base), check_no_server_banner(http, base)]
    report.checks += [check_health(http, base), check_ready(http, base), check_api_headers(http, base),
                      check_hidden_paths(http, base), check_cors_rejects_other_sites(http, base),
                      check_admin_needs_login(http, base)]
    if port_open is not None:
        report.checks.append(check_ports_closed(port_open, base))
    if app and question:
        report.checks.append(check_chat(http, base, app, question))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a deployed RAG site from the outside")
    parser.add_argument("url", help="vd. https://cv.example.com")
    parser.add_argument("--app", help="application_id công khai để thử chat, vd. cv")
    parser.add_argument("--ask", help="Câu hỏi thử (tốn 1 lần embedding + rerank + LLM)")
    parser.add_argument("--insecure", action="store_true", help="Bỏ kiểm tra chứng chỉ (CHỈ khi thử trên máy)")
    parser.add_argument("--skip-ports", action="store_true", help="Không dò cổng 5432/8000/3000/9090")
    args = parser.parse_args()

    report = run_checks(urllib_http(args.insecure), args.url,
                        port_open=None if args.skip_ports else tcp_port_open, app=args.app, question=args.ask)
    for c in report.checks:
        print(f"{'✅' if c.ok else '❌'} {c.name:<36} {c.detail}")
    print("\nĐẠT" if report.ok else "\nCÓ KIỂM TRA TRƯỢT: sửa trước khi gửi link cho nhà tuyển dụng")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
