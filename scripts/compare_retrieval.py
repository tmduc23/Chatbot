"""
MỤC ĐÍCH: So sánh các chiến lược tìm kiếm: vector, keyword, hybrid (RRF), và (tuỳ chọn) hybrid + rerank.

Chế độ 1 — Xem bằng mắt (qualitative): in top-k của từng chiến lược cạnh nhau.
    python scripts/compare_retrieval.py --app cv "pgvector" "containerization" -k 3

Chế độ 2 — Chấm điểm (quantitative) trên bộ câu hỏi có đáp án:
    python scripts/compare_retrieval.py --app cv --eval eval/retrieval_smoke.json -k 5
    python scripts/compare_retrieval.py --app cv --eval eval/retrieval_smoke.json -k 5 --rerank

  File JSON: [{"query": "...", "expected_text": "đoạn chữ phải có trong chunk đúng"}, ...]

  Metrics:
    Hit@k = tỉ lệ câu hỏi có ÍT NHẤT 1 chunk đúng trong top-k.
            Trả lời câu hỏi "có tìm thấy không?" — không quan tâm hạng mấy.
    MRR@k = trung bình của 1/(hạng của chunk đúng đầu tiên); không tìm thấy -> 0.
            (hạng 1 -> 1.0, hạng 2 -> 0.5, hạng 3 -> 0.33, hạng 5 -> 0.2)
            Trả lời câu hỏi "chunk đúng nằm GẦN ĐẦU đến mức nào?".
            Cột hybrid+rr có thể tăng CẢ HAI: reranker xem RETRIEVAL_TOP_K (~20) ứng viên rồi mới
            giữ k, nên chunk đúng đang ở hạng 10 của hybrid có thể được kéo vào top-k (Hit tăng)
            và lên gần đầu (MRR tăng). Nếu hybrid đã tìm thấy hết thì chỉ còn MRR thay đổi.

--rerank: thêm cột "hybrid+rr" = hybrid lấy RETRIEVAL_TOP_K ứng viên rồi reranker giữ lại k.
  Reranker được chọn bằng RERANKER_PROVIDER (gemini | cohere). Đổi tạm cho 1 lần chạy:
    docker compose exec -e RERANKER_PROVIDER=cohere backend python scripts/compare_retrieval.py ... --rerank
  Mỗi câu hỏi tốn 1 lần gọi API; free/trial tier có giới hạn số request/phút
  (code tự chờ và thử lại khi gặp HTTP 429).

Embedding của mỗi câu hỏi được cache -> các chiến lược dùng chung, tiết kiệm quota.
Đây là đánh giá nhanh; Phase 11 làm evaluation đầy đủ.

DESIGN PATTERN dùng trong file này:
  - Strategy:  mỗi cột (vector, keyword, hybrid, hybrid+rr) là một "chiến lược" tìm kiếm,
               cùng tuân theo interface chung `Searcher`. show() và evaluate() chỉ gọi
               search(query, app, k) mà không cần biết bên trong là chiến lược nào.
  - Adapter:   pipeline.run() trả RetrievalResult (khác dạng) -> lambda bọc lại thành
               list[RetrievedChunk] cho khớp với Searcher.
  - Decorator/Proxy: CachedQueryEmbedder bọc provider thật, thêm cache mà không sửa provider.
  - Open/Closed (SOLID): thêm chiến lược mới = thêm 1 phần tử vào `searchers`,
               KHÔNG phải sửa show() hay evaluate().
"""
import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.reranking import get_reranker
from app.rag.retrieval import RETRIEVAL_MODES, RetrievedChunk, build_retriever
from app.rag.retrieval_pipeline import RetrievalPipeline, application_domain_resolver

# Interface chung cho MỌI chiến lược tìm kiếm (Strategy pattern):
#   nhận (query, application_id, k) -> trả list[RetrievedChunk] đã xếp hạng.
# Bất kỳ hàm nào có đúng dạng này đều đưa được vào bảng so sánh.
Searcher = Callable[[str, str, int], list[RetrievedChunk]]   # (query, app, k) -> results


class CachedQueryEmbedder(EmbeddingProvider):
    """
    Bọc một provider thật, nhớ lại vector của các câu hỏi đã embed (Decorator/Proxy pattern).

    Vì sao cần: vector và hybrid đều embed CÙNG một câu hỏi. Không cache thì mỗi câu hỏi
    gọi Gemini Embedding 2-3 lần -> tốn quota, chậm, và dễ bị 429.
    Vẫn kế thừa EmbeddingProvider nên retriever dùng nó y như provider thật.
    """

    def __init__(self, inner: EmbeddingProvider):
        self.inner = inner                              # provider thật, ví dụ Gemini Embedding API
        self.dimension = inner.dimension
        self._cache: dict[str, list[float]] = {}        # câu hỏi -> vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Chỉ dùng khi ingest, script này không gọi -> chuyển thẳng cho provider thật
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:                     # chưa có -> gọi API 1 lần rồi lưu lại
            self._cache[text] = self.inner.embed_query(text)
        return self._cache[text]


def first_relevant_rank(results: list[RetrievedChunk], expected_text: str) -> int | None:
    """Hạng (bắt đầu từ 1) của chunk ĐẦU TIÊN chứa expected_text; None nếu không có trong results."""
    needle = expected_text.lower()
    for rank, r in enumerate(results, start=1):
        if needle in r.text.lower():
            return rank
    return None


def show(queries: list[str], searchers: dict[str, Searcher], app: str, k: int) -> None:
    """Chế độ 1: in top-k của từng chiến lược để so bằng mắt."""
    for q in queries:
        print(f"\n{'=' * 100}\nQUERY: {q}")
        for name, search in searchers.items():
            print(f"\n  --- {name} ---")
            results = search(q, app, k)              # gọi qua interface chung, không cần biết là chiến lược nào
            if not results:
                print("  (no results)")
            for rank, r in enumerate(results, start=1):
                preview = r.text[:70].replace("\n", " ")
                extra = f" {r.ranks}" if r.ranks else ""   # hybrid: hạng trong từng nhánh, ví dụ {'vector': 1, 'keyword': 4}
                print(f"  {rank}. [{r.score:.4f}] {r.filename} #{r.chunk_index}{extra} | {preview}")


def evaluate(eval_file: Path, searchers: dict[str, Searcher], app: str, k: int) -> None:
    """
    Chế độ 2: chấm Hit@k và MRR@k cho mọi chiến lược bằng CÙNG một đoạn code.

    Mỗi ô trong bảng = hạng của chunk đúng đầu tiên ('-' = không có trong top-k).
    Mỗi câu hỏi: tìm thấy -> hit += 1, rr += 1/hạng. Cuối cùng chia cho số câu hỏi.
    """
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    print(f"\nEvaluating {len(cases)} queries, k={k}\n")
    header = f"{'query':<45}" + "".join(f"{m:>11}" for m in searchers)
    print(header + "\n" + "-" * len(header))

    totals = {m: {"hit": 0, "rr": 0.0} for m in searchers}
    for case in cases:
        row = f"{case['query'][:44]:<45}"
        for name, search in searchers.items():
            rank = first_relevant_rank(search(case["query"], app, k), case["expected_text"])
            if rank is not None:
                totals[name]["hit"] += 1
                totals[name]["rr"] += 1.0 / rank
            row += f"{(str(rank) if rank else '-'):>11}"
        print(row)

    n = len(cases)
    print("-" * len(header))
    print(f"{'Hit@' + str(k):<45}" + "".join(f"{totals[m]['hit'] / n:>11.2f}" for m in searchers))
    print(f"{'MRR@' + str(k):<45}" + "".join(f"{totals[m]['rr'] / n:>11.2f}" for m in searchers))
    print("\n(số trong bảng = hạng của chunk đúng đầu tiên; '-' = không tìm thấy trong top-k)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies")
    parser.add_argument("queries", nargs="*", help="Các câu hỏi (chế độ xem bằng mắt)")
    parser.add_argument("--app", required=True)
    parser.add_argument("--eval", type=Path, help="File JSON để chấm điểm")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    # store_true: có ghi --rerank thì args.rerank = True, không ghi thì False
    parser.add_argument("--rerank", action="store_true", help="Thêm cột hybrid + rerank (RERANKER_PROVIDER)")
    args = parser.parse_args()
    if not args.queries and not args.eval:
        parser.error("Cần ít nhất 1 câu hỏi hoặc --eval FILE")
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    embedder = CachedQueryEmbedder(get_embedding_provider(settings))   # mọi chiến lược dùng chung 1 cache

    with SessionLocal() as db:
        # searchers: tên cột -> hàm tìm kiếm (đều là Searcher). Sau vòng lặp có dạng:
        #   {"vector": vector_retriever.retrieve,
        #    "keyword": keyword_retriever.retrieve,
        #    "hybrid": hybrid_retriever.retrieve}
        # Lưu ý: lưu HÀM .retrieve (bound method), không lưu object retriever
        # -> show()/evaluate() gọi search(q, app, k), KHÔNG gọi search.retrieve(...).
        searchers: dict[str, Searcher] = {}
        for mode in RETRIEVAL_MODES:
            retriever = build_retriever(db, settings, embedder=embedder, mode=mode)
            searchers[mode] = retriever.retrieve

        if args.rerank:
            pipeline = RetrievalPipeline(
                build_retriever(db, settings, embedder=embedder, mode="hybrid"),
                get_reranker(settings),                                   # Factory: gemini | cohere theo .env
                # số ứng viên trước rerank phải >= số kết quả cần giữ lại
                retrieval_top_k=max(settings.retrieval_top_k, args.top_k),
                rerank_top_k=args.top_k,                                  # k được cố định ở đây
                domain_resolver=application_domain_resolver(db),
            )
            # Adapter: biến pipeline.run() thành một Searcher.
            #   pipeline.run(q, app)                 -> RetrievalResult
            #   pipeline.run(q, app).reranked        -> list[RerankedChunk]  (đã sắp xếp theo điểm reranker)
            #   [r.chunk for r in ...reranked]       -> list[RetrievedChunk] (đúng dạng Searcher)
            # Tham số k nhận vào cho đúng interface nhưng không dùng, vì rerank_top_k đã = args.top_k.
            searchers["hybrid+rr"] = lambda q, app, k: [r.chunk for r in pipeline.run(q, app).reranked]

        if args.queries:
            show(args.queries, searchers, args.app, args.top_k)
        if args.eval:
            evaluate(args.eval, searchers, args.app, args.top_k)


if __name__ == "__main__":
    main()
