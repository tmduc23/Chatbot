"""
MỤC ĐÍCH: Embed lại TOÀN BỘ chunk sau khi đổi EMBEDDING_PROVIDER / EMBEDDING_MODEL.

Vì sao bắt buộc: vector của 2 model khác nhau nằm ở 2 "không gian" khác nhau,
so cosine giữa câu hỏi (model mới) và chunk (model cũ) cho ra số vô nghĩa.

    python scripts/reembed_all.py --app cv      # 1 application
    python scripts/reembed_all.py --all         # mọi application (nên dùng khi đổi model)

Mỗi application được commit riêng: lỗi ở application nào thì application đó giữ vector cũ.
"""
import argparse
import logging
import sys

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.repositories import application_repository as app_repo
from app.services.document_service import DocumentService


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed every chunk with the current embedding model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app", help="application_id, vd. cv")
    group.add_argument("--all", action="store_true", help="every application")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = get_settings()
    model = s.cohere_embedding_model if s.embedding_provider == "cohere" else s.embedding_model
    print(f"Embedding with {s.embedding_provider} / {model}")

    failed = 0
    with SessionLocal() as db:
        app_ids = [a.id for a in app_repo.list_all(db)] if args.all else [args.app]
        service = DocumentService(db)
        for app_id in app_ids:
            try:
                count = service.reembed_all_chunks(app_id)
                print(f"  {app_id}: re-embedded {count} chunk(s)")
            except Exception as exc:  # in lỗi rồi làm tiếp application khác
                db.rollback()
                failed += 1
                print(f"  {app_id}: FAILED, old vectors kept ({exc})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()