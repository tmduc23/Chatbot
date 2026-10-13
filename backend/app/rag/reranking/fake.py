"""
MỤC ĐÍCH: Reranker GIẢ, chạy offline - dùng cho test

Điểm = tỉ lệ từa khóa của câu hỏi xuất hiện trong chunk (0.0 - 1.0)
Đơn giản nhưng khác thứ tự của retriever -> đủ để test việc "sắp xếp lại" hoạt động
"""

from app.rag.reranking.base import RerankedChunk, Reranker, rank_by_scores
from app.rag.retrieval.base import RetrievedChunk
from app.rag.retrieval.keyword_retriever import extract_terms


class FakeReranker(Reranker):
    def rerank(
            self, query: str, chunks: list[RetrievedChunk], top_k: int, domain: str | None = None
    ) -> list[RerankedChunk]:
        terms = extract_terms(query)
        scores = []
        for c in chunks:
            text = c.text.lower()
            scores.append(sum(t in text for t in terms) / len(terms) if terms else 0.0)
        return rank_by_scores(chunks, scores, top_k)