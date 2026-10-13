"""
MỤC ĐÍCH: Xem reranker thayy đổi thứ tự kết quả như nào cho một câu hỏi.

    python scripts/rerank_demo --app cv "Bạn ấy đã làm dự án nào với Computer Vision?"

In ra 2 bảng:
    1. Trước Rerank: top RETRIEVAL_TOP_K theo hybrid search.
    2. Sau Rerank: top RERANKER_TOP_K, kèm "hạng cũ -> hạng mới" và điểm của reranker. 
"""

import argparse
import logging

from app.db.session import SessionLocal
from app.rag.retrieval_pipeline import build_retrieval_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Show retrieval results before/after reranking")
    parser.add_argument("query")
    parser.add_argument("--app", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    with SessionLocal() as db:
        result = build_retrieval_pipeline(db).run(args.query, args.app)

    print(f"\nQuery: {args.query!r}\n")
    print(f"BEFORE rerank - {len(result.candidates)} candidates ({result.retrieval_ms:.0f} ms)")
    for rank, c in enumerate(result.candidates, start=1):
        print(f" {rank:>2}. [{c.score:.4f}] {c.filename} #{c.chunk_index} | {c.text[:70].replace(chr(10), ' ')}")

    print(f"\nAFTER rerank - top {len(result.reranked)} ({result.rerank_ms:.0f} ms)")
    for new_rank, r in enumerate(result.reranked, start=1):
        moved = r.retrieval_rank - new_rank
        arrow = f"↑{moved}" if moved > 0 else (f"↓{-moved}" if moved < 0 else "=")
        print(f" {new_rank:>2}. [{r.score:.2f}] (was #{r.retrieval_rank:<} {arrow:>3})"
              f"{r.chunk.filename} #{r.chunk.chunk_index} | {r.chunk.text[:60].replace(chr(10), ' ')}")

if __name__ == "__main__":
    main()