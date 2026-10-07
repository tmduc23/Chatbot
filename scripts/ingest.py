"""
MỤC ĐÍCH: Công cụ dòng lệnh để nạp tài liệu vào hệ thống khi chuuaw có API (Phase 9)

Ví dụ (chạy trong container backend):
    python scripts/ingest.py --app cv --app-name "My CV" data/samples/cv.pdf
    python scripts/ingest.py --app projects data/samples/*.md --show 5

Script chỉ là "vỏ": đọc tham số, gọi DocumentService, in kết quả.
Mọi logic nằm trong service -> API sau này dùng lại y nguyên.
"""

import argparse
import logging
from pathlib import Path

from app.core.exceptions import ApplicationNotFoundError, DuplicateDocumentError, IngestionError
from app.db.session import SessionLocal
from app.repositories import application_repository as app_repo
from app.repositories import document_repository as doc_repo
from app.services.document_service import DocumentService



def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into an application")
    parser.add_argument("files", nargs="+", type=Path, help="Đường dẫn file (pdf, docx, txt, md)")
    parser.add_argument("--app", required=True, help="application_id, vd. cv")
    parser.add_argument("--app-name", help="Tạo application nếu chưa tồn tại, với tên này")
    parser.add_argument("--show", type=int, default=3, help="In N chunk đầu tiên của mỗi file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with SessionLocal() as db:
        if args.app_name and app_repo.get(db, args.app) is None:
            app_repo.create(db, args.app, args.app_name)
            db.commit()
            print(f"Created application '{args.app}")

        service = DocumentService(db)
        for path in args.files:
            print(f"\n=== {path} ===")
            try: 
                doc = service.ingest_file(args.app, path)
            except DuplicateDocumentError as e:
                print(f"SKIP {e}")
                continue
            except (ApplicationNotFoundError, IngestionError) as e:
                print(f"ERROR {e}")
                continue

            chunks = doc_repo.list_chunks(db, args.app, doc.id)
            print(f"OK    document_id={doc.id} status={doc.status.value} chunks={len(chunks)}")

            for c in chunks[:args.show]:
                preview = c.text[:200].replace("\n", " ⏎ ")
                print(f" [#{c.chunk_index} page={c.page} len={len(c.text)}] {preview}")

if __name__ == "__main__":
    main()