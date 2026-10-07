"""
MỤC ĐÍCH: Thử vector search từ dòng lệnh - xem câu hỏi tìm ra những chunk nào, điểm bao nhiêu.

    python scripts/search.py --app cv "Kinh nghiệm với Docker"
    python scripts/search.py --app cv " What projects used FastAPI?" -k 3

Dùng để kiểm tra chất lượng retrieval trước khi có LLM:
nếu chunk đúng không nằm trong top kết quả thì LLM không thể trả lời đúng
"""
import argparse
import logging
import time

from app.db.session import SessionLocal
from app.rag.embeddings import get_embedding_provider
from app.rag.retrieval import VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector search over an application")
    parser.add_argument("query", help="Câu hỏi")
    parser.add_argument("--app", required=True, help="application_id, vd. cv")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with SessionLocal() as db:
        retriever = VectorRetriever(db, get_embedding_provider())
        t0 = time.perf_counter()
        results = retriever.retrieve(args.query, args.app, args.top_k)
        elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nQuery: {args.query!r}  app={args.app}  results={len(results)}  total={elapsed:.0f} ms\n")
    for rank, r in enumerate(results, start=1):
        preview = r.text[:160].replace("\n", " ⏎ ")
        print(f"{rank}. score={r.score:.3f}  {r.filename} p.{r.page} #{r.chunk_index}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()