"""
MỤC ĐÍCH: Tạo / xoá INDEX BM25 RIÊNG cho từng application (pg_textsearch).

Vì sao cần:
  Index chung ix_chunks_bm25 tính IDF và độ dài trung bình trên TOÀN BẢNG chunks. Nạp thêm app khác
  (vd. peerqa) là điểm keyword của qasper đổi theo, dù kết quả vẫn chỉ lấy chunk của qasper.
  Index riêng có điều kiện WHERE application_id = '<id>' -> chỉ tính thống kê trên chunk của app đó.

  CREATE INDEX ix_chunks_bm25_qasper_<hash> ON chunks USING bm25 ("text")
      WITH (text_config = 'english', k1 = 1.2, b = 0.75) WHERE application_id = 'qasper';

  - Tạo 1 LẦN cho mỗi app (trước hay sau khi nạp đều được): chunk nạp sau tự vào index.
  - text_config / k1 / b chép từ index chung -> app có index riêng và app dùng index chung
    cùng cách tách từ.
  - Retriever (KeywordRetriever) tự nhận ra index riêng theo tên, không cần đổi .env.

Dùng qua: python scripts/create_bm25_index.py --app qasper   (xem --help)
"""
import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.rag.retrieval.keyword_retriever import PGTS_INDEX, app_index_name, sql_literal

DEFAULT_OPTIONS = {"text_config": "english", "k1": "1.2", "b": "0.75"}
_OPTION = re.compile(r"^(text_config|k1|b)=([A-Za-z0-9_.]+)$")


def read_index_options(db: Session, index: str = PGTS_INDEX) -> dict[str, str]:
    """Đọc WITH (...) của index chung: {'text_config': 'english', 'k1': '1.2', 'b': '0.75'}.

    Không có index chung -> DEFAULT_OPTIONS. Chỉ nhận giá trị dạng chữ/số (vì sẽ ghép vào DDL).
    """
    row = db.execute(
        text("SELECT c.reloptions FROM pg_class c WHERE c.relname = :name AND c.relkind = 'i'"), {"name": index}
    ).first()
    options = dict(DEFAULT_OPTIONS)
    for item in (row[0] or []) if row else []:
        m = _OPTION.match(item)
        if m:
            options[m.group(1)] = m.group(2)
    return options


def extension_ready(db: Session) -> bool:
    return db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch'")).first() is not None


def create_app_index(db: Session, application_id: str, options: dict[str, str] | None = None) -> str:
    """Tạo index riêng cho app (đã có thì bỏ qua). Trả về tên index. Caller tự commit."""
    opts = options or read_index_options(db)
    for key, value in opts.items():   # ghép vào DDL -> chỉ cho phép giá trị an toàn
        if not _OPTION.match(f"{key}={value}"):
            raise ValueError(f"giá trị không hợp lệ: {key}={value!r}")
    name = app_index_name(application_id)
    db.execute(text(
        f'CREATE INDEX IF NOT EXISTS {name} ON chunks USING bm25 ("text") '
        f"WITH (text_config = '{opts['text_config']}', k1 = {opts['k1']}, b = {opts['b']}) "
        f"WHERE application_id = {sql_literal(application_id)}"
    ))
    return name


def drop_app_index(db: Session, application_id: str) -> str:
    name = app_index_name(application_id)
    db.execute(text(f"DROP INDEX IF EXISTS {name}"))
    return name


def list_bm25_indexes(db: Session) -> list[tuple[str, str]]:
    """[(tên index, câu CREATE INDEX)] của mọi index BM25 trên bảng chunks."""
    return [tuple(r) for r in db.execute(text(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks' AND indexdef ILIKE '%USING bm25%' "
        "ORDER BY indexname"
    )).all()]
