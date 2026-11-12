"""
MỤC ĐÍCH: Tạo / xoá / liệt kê index BM25 RIÊNG cho từng application (KEYWORD_RANKING=pg_textsearch).

    python scripts/create_bm25_index.py --list                 # index BM25 đang có
    python scripts/create_bm25_index.py --app qasper           # tạo index riêng cho qasper
    python scripts/create_bm25_index.py --app qasper --app peerqa
    python scripts/create_bm25_index.py --all                  # mọi application đang có
    python scripts/create_bm25_index.py --app qasper --drop    # xoá -> quay về index chung

Tạo 1 lần cho mỗi app; chunk nạp sau tự vào index. Không cần restart backend: retriever hỏi DB
xem app có index riêng không mỗi khi được tạo mới (mỗi request / mỗi lần chạy script).
"""
import argparse
import sys

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models import Chunk
from app.rag.retrieval.bm25_index import (
    create_app_index, drop_app_index, extension_ready, list_bm25_indexes, read_index_options,
)
from app.repositories import application_repository as app_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Index BM25 riêng cho từng application")
    parser.add_argument("--app", action="append", default=[], help="id application (lặp lại được)")
    parser.add_argument("--all", action="store_true", help="mọi application đang có")
    parser.add_argument("--drop", action="store_true", help="xoá index riêng thay vì tạo")
    parser.add_argument("--list", action="store_true", help="liệt kê index BM25 trên bảng chunks")
    args = parser.parse_args()

    with SessionLocal() as db:
        if not extension_ready(db):
            sys.exit("Chưa có extension pg_textsearch (xem guides/nang-postgres-17.md và migration 0006).")
        if args.list or not (args.app or args.all):
            for name, ddl in list_bm25_indexes(db):
                print(f"{name}\n    {ddl}")
            return

        known = {a.id for a in app_repo.list_all(db)}
        apps = sorted(known) if args.all else args.app
        unknown = [a for a in apps if a not in known]
        if unknown:
            sys.exit(f"Không có application: {', '.join(unknown)} (xem: python scripts/list_apps.py)")

        options = read_index_options(db)
        print(f"Tuỳ chọn (chép từ index chung): {options}")
        for app_id in apps:
            if args.drop:
                print(f"- {app_id}: đã xoá {drop_app_index(db, app_id)}")
                continue
            n = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.application_id == app_id))
            print(f"+ {app_id}: {create_app_index(db, app_id, options)} ({n} chunk)")
        db.commit()


if __name__ == "__main__":
    main()
