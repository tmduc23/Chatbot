"""
chat: lưu nguồn trích dẫ + lý do từ chối vào message, nối retrieval_logs với câu trả lời

Revision ID: 0004
Revises: 0003

MỤC ĐÍCH: Phase 9 bắt đầu GHI hội thoại thật qua POST /api/v1/chat. Bảng messages hiện chỉ có
role + content, chưa đủ để:
    - mở lại 1 phiên chat cũ mà vẫn phát hiện nguồn [Source n]      -> messages.sources (JSONB)
    - thống kê "bao nhiêu % câu bị từ chối, vì sao" (Phase 11)  -> messages.abstain_reason
    - biết câu nào bị grounding check chấm SUPPORTED/PARTIAL/UNSUPPORTED -> messages.grounding_verdict
    - từ 1 câu trả lời tìm lại retrieval đã chạy (chunk nào, điểm bao nhiêu) -> retrieval_logs.message_id
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default '[]': các tin nhắn cũ (nếu có) tự nhận danh sách rỗng, cột vẫn NOT NULL được
    op.add_column("messages", sa.Column("sources", postgresql.JSONB, nullable=False,
                                        server_default=sa.text("'[]'::jsonb")))
    op.add_column("messages", sa.Column("abstain_reason", sa.String(50), nullable=True))
    op.add_column("messages", sa.Column("grounding_verdict", sa.String(20), nullable=True))

    # SET NULL: xóa tin nhắn thì log retrieval vẫn giữ (log dùng để thống kê, không phụ thuộc vào tin nhắn)
    op.add_column("retrieval_logs", sa.Column("message_id", sa.Uuid, nullable=True))
    op.create_foreign_key("fk_retrieval_logs_message", "retrieval_logs", "messages",
                          ["message_id"], ["id"], ondelete="SET NULL")