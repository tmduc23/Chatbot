"""
MỤC ĐÍCH: In nội dung các chunk ĐANG CÓ trong DB của 1 application -> chép expected_texts cho đúng nguyên văn.

    python scripts/show_chunks.py --app cv
    python scripts/show_chunks.py --app cv --find "TOEIC"     # chỉ chunk chứa đoạn này (so như dataset)

`missing_ground_truth` báo một câu không khớp chunk nào. Chạy `--find` để xem đoạn chữ có bị cắt giữa 2 chunk không, 
hoặc có bị gõ sai không.

"""

import argparse

from sqlalchemy import select

from app.db.session import SessionLocal
from app.evaluation.dataset import normalize
from app.models import Chunk, Document


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stored chunks of an application")
    parser.add_argument("--app", required=True)
    parser.add_argument("--find", help="Chỉ in chunk chứa đoạn chữ này")
    args = parser.parse_args()

    with SessionLocal() as db:
        rows = db.execute(
            select(Document.filename, Chunk.chunk_index, Chunk.text)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.application_id == args.app)
            .order_by(Document.filename, Chunk.chunk_index)
        ).all()

    shown = 0
    for row in rows:
        if args.find and normalize(args.find) not in normalize(row.text):
            continue
        shown += 1
        print(f"\n===== {row.filename} #{row.chunk_index} ({len(row.text)} ký tự) =====\n{row.text}")
    print(f"\n{shown}/{len(rows)} chunk")


if __name__ == "__main__":
    main()