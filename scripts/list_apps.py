"""
MỤC ĐÍCH: Liệt kê các application và số document / chunk của từng cái (Phase 8).
    python scripts/list_apps.py

Dùng để kiểm tra nhanh: tài liệu đang nằm ở application nào, có bị nạp nhầm không
"""
from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.repositories import application_repository as app_repo


def main() -> None:
    with SessionLocal() as db:
        apps = app_repo.list_all(db)
        if not apps:
            print("Chưa có application nào. Tạo bằng: python scripts/ingest.py --app <id> --app-name <tên> <file>")
            return
        docs = dict(db.execute(select(Document.application_id, func.count()).group_by(Document.application_id)).all())
        chunks = dict(db.execute(select(Chunk.application_id, func.count()).group_by(Chunk.application_id)).all())
        print(f"{'application':<15} {'documents':>9} {'chunks':>7}  name / description")

        for a in apps:
            desc = f" — {a.description}" if a.description else ""
            print(f"{a.id:<15} {docs.get(a.id, 0):>9} {chunks.get(a.id, 0):>7}  {a.name}{desc}")

if __name__ == "__main__":
    main()