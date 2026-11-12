"""
MỤC ĐÍCH (Phase 12d): tự rà soát cấu hình lúc khởi động — phần "security review" chạy được bằng code.

Không chặn khởi động (bạn có thể CỐ Ý chạy production không cần đăng nhập, vd. chỉ có app CV công khai),
nhưng in WARNING rõ ràng vào log để không ai "quên" mà không biết.
"""
from app.core.config import Settings


def security_warnings(settings: Settings) -> list[str]:
    if settings.app_env == "development":
        return []
    warnings = []
    if not settings.admin_api_key and not settings.auth_enabled:
        warnings.append("ADMIN_API_KEY is not set and AUTH_ENABLED=false: admin API is locked (403)")
    if not settings.auth_enabled:
        warnings.append("AUTH_ENABLED=false: every application is readable by anyone who knows its id")
    if not settings.rate_limit_enabled:
        warnings.append("RATE_LIMIT_ENABLED=false: one client can use up the whole LLM/embedding quota")
    if "change_me" in settings.database_url:
        warnings.append("DATABASE_URL still uses the example password 'change_me'")
    if "*" in settings.cors_origins:
        warnings.append("CORS_ORIGINS contains '*': any website can call this API from a browser")
    if settings.log_format != "json":
        warnings.append("LOG_FORMAT is not 'json': logs are harder to search and alert on")
    if not settings.metrics_token:
        warnings.append("METRICS_TOKEN is not set: /metrics returns 403")
    # Phase 13: chạy thật trên Internet
    if any(o.strip().startswith("http://") for o in settings.cors_origins.split(",")):
        warnings.append("CORS_ORIGINS has an http:// origin: a public site should only be served over https")
    if settings.expose_api_docs:
        warnings.append("EXPOSE_API_DOCS=true: /docs and /openapi.json are readable by anyone")
    if settings.trust_forwarded_for:
        warnings.append("TRUST_FORWARDED_FOR=true: make sure a trusted reverse proxy sets X-Forwarded-For")
    return warnings
