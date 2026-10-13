"""
MỤC ĐÍCH: Thử tìm kiếm từ dòng lệnh - xem câu hỏi tìm ra những chunk nào, điểm bao nhiêu, chạy bằng command line.
    python scripts/search.py --app cv "Kinh nghiệm với Docker?"                 # mode theo .env
    python scripts/search.py --app cv "pgvector" --mode keyword -k 3
    python scripts/search.py --app cv "What projects used FastAPI?" --mode vector

Dùng để "cảm nhận" chất lượng retrieval trước khi có LLM:
nếu chunk đúng không nằm trong top kết quả thì LLM cũng không thể trả lời đúng.
"""

import argparse
import logging
import time


from app.db.session import SessionLocal
from app.rag.retrieval import RETRIEVAL_MODES, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search an application's chunks")
    parser.add_argument("query", help="Câu hỏi")
    parser.add_argument("--app", required=True, help="application_id, vd. cv")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, help="Mặc định: RETRIEVAL_MODE trong .env")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


    with SessionLocal() as db:      # Tạo session để có thể truy vấn
        retriever = build_retriever(db, mode=args.mode)
        t0 = time.perf_counter()
        results = retriever.retrieve(args.query, args.app, args.top_k)
        elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nQuery: {args.query!r}  app={args.app}  mode={type(retriever).__name__}  "
          f"results={len(results)}   total={elapsed:.0f} ms\n")


    for rank, r in enumerate(results, start=1):
        preview = r.text[:160].replace("\n", " ⏎ ")
        ranks = f"  ranks={r.ranks}" if r.ranks else ""
        print(f"{rank}. score={r.score:.4f}  {r.filename} p.{r.page} #{r.chunk_index}{ranks}")
        print(f"   {preview}\n")