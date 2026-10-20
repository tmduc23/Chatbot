"""
MỤC ĐÍCH: Hỏi chatbot từ dòng lệnh - chạy toàn bộ RAG: retrieval -> rerank -> lọc ngưỡng -> LLM.

    python scripts/ask.py --app cv "Bạn ấy có kinh nghiệm Docker không?"
    python scripts/ask.py --app cv "What is the expected salary?" --debug

--debug: in thêm điểm rerank của từng chunk, chunk nào qua ngưỡng (✓) / bị loại (✗), và log chi tiết.

vậy --debug giúp g ì

--debug giúp bạn tìm nguyên nhân khi chatbot trả lời không như mong đợi. Không có nó, bạn chỉ thấy câu trả lời cuối mà không biết vì sao.

Ba tình huống hay gặp:

Câu có đáp án mà bị từ chối (no_relevant_chunks)
Nhìn bảng rerank, thấy chunk đúng được 0.35 ✗ trong khi ngưỡng là 0.40. Vậy ngưỡng hơi cao, hạ xuống hoặc chạy tune_threshold.py.
Trả lời sai hoặc thiếu
Bảng rerank không có chunk chứa đáp án. Nghĩa là lỗi ở bước tìm kiếm chứ không phải ở LLM. Log INFO cho thấy chunk đó đứng hạng mấy trước khi rerank, 
từ đó biết nên sửa hybrid search hay reranker.
Câu lạc đề vẫn được trả lời
Thấy chunk “Hobbies” được 0.55 ✓. Ngưỡng quá thấp, cần nâng lên.

result.retrieval là một RetrievalResul gồm candidates (chunk từ hybrid search), reranked (chunk sau rerank), relevant (chunk trong reranked)
"""

import argparse
import logging

from app.db.session import SessionLocal
from app.services.rag_service import build_rag_service
from app.core.exceptions import ApplicationNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG chatbot a question")
    parser.add_argument("question")
    parser.add_argument("--app", required=True)
    parser.add_argument("--debug", action="store_true", help="In ra điểm rerank và log chi tiết")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING)

    # mở kết nối tới đb, sau đó tự đóng
    with SessionLocal() as db:
        try:
            result = build_rag_service(db).ask(args.question, args.app)
        except ApplicationNotFoundError as exc:
            raise SystemExit(f"Lỗi: {exc}. Xem danh sách: python scripts/list_apps")

    r = result.retrieval
    if args.debug:
        print(f"\nRerank (min_score={r.min_score:.2f}):")
        for rank, c in enumerate(r.reranked, start=1):
            mark = "✓" if c in r.relevant else "✗"
            print(f"  {mark} {rank}. [{c.score:.2f}] {c.chunk.filename} #{c.chunk.chunk_index} | "
                  f"{c.chunk.text[:60].replace(chr(10), ' ')}")

    print(f"\nQ: {result.question}\n")
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            page = f", page {s.page}" if s.page is not None else ""
            print(f"  [Source {s.number}] {s.filename}{page} (chunk #{s.chunk_index}, rerank {s.score:.2f})")

    if result.abstained:
        print(f"\n(từ chối: {result.abstain_reason})")
    print(f"\nretrieval {r.retrieval_ms:.0f} ms | rerank {r.rerank_ms:.0f} ms | llm {result.llm_ms:.0f} ms"
          f" | total {result.total_ms:.0f} ms | tokens in={result.input_tokens} out={result.output_tokens}")


if __name__ == "__main__":
    main()