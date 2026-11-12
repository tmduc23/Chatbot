"""
MỤC ĐÍCH (Phase 12c): RATE LIMITING — mỗi người chỉ được gọi API tối đa N lần trong 1 khoảng thời gian.

Vì sao cần, với project này:
  - Mỗi câu hỏi /chat tốn 1 lần embedding + 1 lần rerank (Cohere trial: 10 lần/phút, 1000 lần/tháng)
    + 1-2 lần LLM (OpenRouter free có giới hạn/ngày). 1 người bấm liên tục hoặc 1 script spam
    là tiêu hết quota của CẢ hệ thống, người khác nhận 503.
  - /auth/login không giới hạn = cho phép thử hàng triệu mật khẩu (brute force).

Thuật toán: FIXED WINDOW COUNTER (bộ đếm theo cửa sổ cố định).
    window = 60s. Request lúc 10:00:17 -> cửa sổ [10:00:00, 10:01:00) -> count += 1
    count > limit -> 429, Retry-After = số giây còn lại tới 10:01:00
Đơn giản nhất và đủ dùng. Nhược điểm: sát ranh giới 2 cửa sổ có thể lọt tới 2x limit trong ~1 giây
(10 request lúc 10:00:59 + 10 request lúc 10:01:00). Muốn mượt hơn: sliding window / token bucket.

Đếm trong Postgres, 1 câu SQL nguyên tử:
    INSERT ... VALUES (key, window, 1) ON CONFLICT (key, window_start) DO UPDATE SET count = count + 1 RETURNING count
2 request cùng lúc không bao giờ cùng đọc count=5 rồi cùng ghi 6 (lỗi "lost update" của cách
SELECT rồi UPDATE). Nhiều process uvicorn / restart backend vẫn đếm đúng, không cần Redis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.models import RateLimitCounter

logger = logging.getLogger(__name__)

_UPSERT = text("""
    INSERT INTO rate_limits (key, window_start, count) VALUES (:key, :window_start, 1)
    ON CONFLICT (key, window_start) DO UPDATE SET count = rate_limits.count + 1
    RETURNING count
""")

KEEP_OLD_WINDOWS = timedelta(days=2)   # cửa sổ dài nhất là 1 ngày -> giữ dư 2 ngày rồi xoá


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    count: int          # số request trong cửa sổ hiện tại, tính cả request này
    limit: int
    retry_after: int    # giây tới cửa sổ kế tiếp (chỉ có nghĩa khi allowed=False)


def window_start(now: datetime, window_seconds: int) -> datetime:
    """Làm tròn xuống đầu cửa sổ: 10:00:17 với window 60s -> 10:00:00."""
    epoch = int(now.timestamp())
    return datetime.fromtimestamp(epoch - epoch % window_seconds, tz=timezone.utc)


class RateLimiter:
    def __init__(self, db: Session):
        self.db = db

    def hit(self, key: str, limit: int, window_seconds: int, now: datetime | None = None) -> RateLimitResult:
        """Đếm 1 request cho `key`, COMMIT ngay (request sau đó lỗi thì lần thử vẫn bị tính)."""
        now = now or datetime.now(timezone.utc)
        start = window_start(now, window_seconds)
        count = self.db.execute(_UPSERT, {"key": key, "window_start": start}).scalar_one()
        if count == 1:
            # cửa sổ mới vừa mở -> tiện tay dọn các cửa sổ đã hết hạn từ lâu (bảng không phình mãi)
            self.db.execute(delete(RateLimitCounter).where(RateLimitCounter.window_start < now - KEEP_OLD_WINDOWS))
        self.db.commit()
        retry_after = max(1, int((start + timedelta(seconds=window_seconds) - now).total_seconds()))
        allowed = count <= limit
        if not allowed:
            logger.warning("rate limit exceeded key=%s count=%d limit=%d", key, count, limit)
        return RateLimitResult(allowed=allowed, count=count, limit=limit, retry_after=retry_after)