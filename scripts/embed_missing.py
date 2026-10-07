"""
MỤC ĐÍCH: Embed bổ sung các chunk chưa có vector (embedding IS NULL)

Dùng khi:
    - chunk được nạp ở Phase 2 (lúc chưa có embedding);
    - lần embed trước bị lỗi giữa chừng (vd. hết quota free tier)

    python scripts/embed_missing.py --app cv
"""

import argparse
import logging

from app.db.session import SessionLocal
from app.services.document_service import DocumentService



def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks that have no embedding yet")
    parser.add_argument("--app", required=True, help="application_id, vd. cv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with SessionLocal() as db:
        count = DocumentService(db).embed_missing_chunks(args.app)
        print(f"Embedded {count} chunk(s) in application '{args.app}'")


if __name__ == "__main__":
    main()