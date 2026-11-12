"""
MỤC ĐÍCH (Phase 12a): tiến trình worker xử lý tài liệu nền (service `worker` trong docker-compose).

    python scripts/worker.py            # chạy mãi, dừng bằng Ctrl+C / docker compose stop worker
    python scripts/worker.py --drain    # xử lý hết hàng đợi rồi thoát (tiện khi thử tay)

Dừng "êm" (graceful shutdown): SIGTERM (docker stop) chỉ đặt cờ dừng. Document đang xử lý được
làm xong rồi mới thoát, không bỏ dở giữa chừng.
"""

import argparse
import logging
import signal
import threading

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal
from app.rag.embeddings import get_embedding_provider
from app.services.document_service import DocumentService
from app.workers.document_worker import run_forever, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Background document worker")
    parser.add_argument("--drain", action="store_true", help="Xử lý hết document đang chờ rồi thoát")
    args = parser.parse_args()


    settings = get_settings()

    configure_logging(settings)   # Phase 12d: cùng định dạng log (text/JSON, che secret) với backend
    embedder = get_embedding_provider(settings)   # tạo 1 lần, dùng cho mọi document

    def make_service(db):
        return DocumentService(db, settings=settings, embedder=embedder)

    if args.drain:
        count = 0
        with SessionLocal() as db:
            while run_once(db, make_service(db), settings) is not None:
                count += 1
        print(f"Đã xử lý {count} document.")
        return

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    logging.getLogger(__name__).info("worker started, poll every %ss", settings.worker_poll_seconds)
    processed = run_forever(SessionLocal, make_service, settings, stop)
    logging.getLogger(__name__).info("worker stopped after %d document(s)", processed)


if __name__ == "__main__":
    main()