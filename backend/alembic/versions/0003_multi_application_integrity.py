"""
multi-application: ràng buộc cùng application ở tầng database + retrieval_logs.applcation_id

Revision ID: 0003
Revises: 0002

MỤC ĐÍCH: Code đã luôn lọc theo application, (Phase 1-7). Migration này thêm lớp bảo vệ
thứ 2 ở DATABASE, để dù code có bug thì dữ liệu không thể "lệch application":

1. chunks(document_id, application_id) -> documents (id, application_id)
    Chunk KHÔNG THỂ mang application_id khác với document của nó.
    (Trước đây chunk có 2 khóa ngoại lệ riêng lẻ: document_id hợp lệ + application_id hợp lệ,
    nhưng không có gì bắt 2 giá trị đó phải KHỚP NHAU.)
    Khóa ngoại 2 cột THAY THẾ khóa ngoại cũ chunks.document_id -> documents.id
    (giữ cả 2 thì SQLAlchemy không biết JOIN chunks-documents theo khóa nào.)

2. retrieval_logs có thể thêm cột appilcation_id(spec: application_id có mặt ở mọi nơi),
    và (session_id, application_id) -> chat_sessions (id, application_id):
    log của 1 phiên chat phải cùng application với phiên chat đó (cũng thay khóa ngoại cũ).

3. messages.seq: số thứ tự tăng dần (IDENTITY) để sắp xếp tin nhắn
    created_at = now() thời điểm BẮT ĐẦU transaction -> câu hỏi và câu trả lời ghi trong cùng 
    1 transaction có created_at GIỐNG HỆT NHAU, sắp theo created_at không biết cái nào trước.

Khóa ngoại nhiều cột cần phía được tham chiếu có UNIQUE trên đúng các cột đó, nên phải thêm
UNIQUE (id, application_id) cho documents và chat_session trước (id đã là khóa chính nên cặp 
(id, application_id) chắc chắn unique - chỉ là Postgres yêu cầu khai báo rõ ràng.)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- 1. chunk phải cùng application với document của nó ---
    op.create_unique_constraint("uq_documents_id_app", "documents", ["id", "application_id"])
    # Tên do Postgres tự đặt ở migration 0001 (bảng_cột_fkey)
    op.drop_constraint("chunks_document_id_fkey", "chunks", type_="foreignkey")
    op.create_foreign_key(
        "fk_chunks_document_same_app", "chunks", "documents",
        ["document_id", "application_id"], ["id", "application_id"], ondelete="CASCADE",
    )

    # --- 2. retrieval_logs.application_id ---
    # thêm cột application_id
    op.add_column("retrieval_logs", sa.Column("application_id", sa.String(50), nullable=True))
    # Log cũ (nếu có): lấy application từ phiên chat; log không gắn phiên chat thì không biết
    # thuộc application nào -> xóa (bảng log, không phải dữ liệu nghiệp vụ)
    op.execute(
        "UPDATE retrieval_logs AS r SET application_id = s.application_id "  # mỗi log r, tìm phiên chat s có cùng session_id
        "FROM chat_sessions AS s WHERE r.session_id = s.id"     
    )
    op.execute("DELETE FROM retrieval_logs WHERE application_id IS NULL")   # log nào NULL thì xóa
    op.alter_column("retrieval_logs", "application_id", nullable=False)     # khóa cột thành NOT-NULL

    # --- 3. log của 1 phiên chat phải cùng application với phiên chat ---
    op.create_unique_constraint("uq_chat_sessions_id_app", "chat_sessions", ["id", "application_id"])
    op.drop_constraint("retrieval_logs_session_id_fkey", "retrieval_logs", type_="foreignkey")
    op.create_foreign_key(
        "fk_retrieval_logs_session_same_app", "retrieval_logs", "chat_sessions",
        ["session_id", "application_id"], ["id", "application_id"], ondelete="CASCADE"
    )

    # --- 4. thứ tự tin nhắn ---
    # Thêm cột IDENTITY vào bảng đã có dữ liệu: Postgres tự đánh số các dòng cũ
    op.add_column("messages", sa.Column("seq", sa.BigInteger, sa.Identity(always=True), nullable=False))


def downgrade() -> None:
    op.drop_column("messages", "seq")
    op.drop_constraint("fk_retrieval_logs_session_same_app", "retrieval_logs", type_="foreignkey")
    op.create_foreign_key("retrieval_logs_session_id_fkey", "retrieval_logs", "chat_sessions",
                          ["session_id"], ["id"], ondelete="CASCADE")
    op.drop_constraint("uq_chat_sessions_id_app", "chat_sessions", type_="unique")
    op.drop_index("ix_retrieval_logs_application_id", table_name="retrieval_logs")
    op.drop_constraint("fk_retrieval_logs_application", "retrieval_logs", type_="foreignkey")
    op.drop_column("retrieval_logs", "application_id")
    op.drop_constraint("fk_chunks_document_same_app", "chunks", type_="foreignkey")
    op.create_foreign_key("chunks_document_id_fkey", "chunks", "documents",
                          ["document_id"], ["id"], ondelete="CASCADE")
    op.drop_constraint("uq_documents_id_app", "documents", type_="unique")