"""
MỤC ĐÍCH: tạo ứng dụng FastAPI. Phase 12d: tách thành create_app(settings) để test dựng được app
với cấu hình production (tắt /docs, CORS chặt) mà không phải sửa .env.
"""
import logging

from fastapi import FastAPI

# Middleware dùng để xử lý CORS
# CORS giúp backend kiểm soát: Frontend nào được phép gửi request tới backend
# Frontend chạy: http://localhost:3000
# Backend chạy:  http://localhost:8000
# Hai origin khác nhau nên cần CORS để cho phép Frontend gọi Backend.
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.middleware import register_middleware
from app.api.v1 import api_v1_router
from app.core.config import Settings, get_settings
from app.core.config_check import security_warnings
from app.core.logging import configure_logging
from app.services.auth_service import check_auth_config

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    # Phase 12b: AUTH_ENABLED=true mà thiếu/yếu JWT_SECRET -> dừng ngay lúc khởi động, không chạy "nửa an toàn"
    check_auth_config(settings)
    for warning in security_warnings(settings):
        logger.warning("security config: %s", warning)

    # Phase 12d: /docs liệt kê mọi route và schema -> bản đồ cho kẻ tấn công. Mặc định chỉ bật khi dev
    docs = settings.expose_api_docs if settings.expose_api_docs is not None else settings.app_env == "development"
    app = FastAPI(
        title="Multi-Application RAG", version="0.12.0",
        docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    # Middleware chạy ngược thứ tự thêm vào: CORS (thêm sau) bọc ngoài cùng -> preflight OPTIONS
    # được trả lời trước khi tới middleware log/metrics
    register_middleware(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        # Phase 12d: chỉ cho đúng method/header frontend dùng, thay vì "*"
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],   # cho JavaScript đọc được 2 header này
    )
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(api_v1_router, prefix="/api/v1")
    return app


# Lấy toàn bộ cấu hình của ứng dụng (đọc .env)
settings = get_settings()
configure_logging(settings)   # Phase 12d: thay logging.basicConfig — text hoặc JSON, có request_id, che secret
app = create_app(settings)
