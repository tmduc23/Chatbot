"""Route GET /metrics (Phase 12d): định dạng văn bản của Prometheus.

Không công khai: metrics lộ tên application, số request, tỉ lệ lỗi... Chỉ Prometheus (có token) đọc được.
  - Có METRICS_TOKEN  -> header "Authorization: Bearer <METRICS_TOKEN>" phải khớp, sai/thiếu -> 401
  - Không có token    -> chỉ cho phép khi APP_ENV=development, môi trường khác -> 403 (quên đặt thì KHÓA)
"""
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.metrics import DOCUMENTS
from app.db.session import get_db
from app.models import Document, DocumentStatus

router = APIRouter(tags=["monitoring"])


def require_metrics_access(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.metrics_token.get_secret_value() if settings.metrics_token else ""
    if not expected:
        if settings.app_env == "development":
            return
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Metrics are disabled: METRICS_TOKEN is not set")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing metrics token")


@router.get("/metrics", include_in_schema=False, dependencies=[Depends(require_metrics_access)])
def metrics(db: Session = Depends(get_db)) -> Response:
    # Gauge "documents" đọc từ DB đúng lúc Prometheus hỏi: thấy được hàng đợi của worker (Phase 12a)
    # dù worker là process khác, không có /metrics riêng
    counts = dict(db.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    for s in DocumentStatus:
        DOCUMENTS.labels(s.value).set(counts.get(s, 0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
