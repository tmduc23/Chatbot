"""
MỤC ĐÍCH: Test RetrievalPipelin (retrive top N -> rerank top K) bằng retriever + reranker giả

Kiểm tra pipeline:
    - gọi retriever với RETRIEVAL_TOP_K và đúng application_id;
    - chỉ trả về RERANKER_TOP_K kết quả đã sắp xếp lại;
    - giữ lại danh sách ứng viên TRƯỚC rerank để so sánh;
    - từ chối cấu hình vô lý (rerank_top_k > retrieval_top_k)
"""

import uuid

import pytest

from app.core.config import Settings
from app.rag.reranking import PassthroughReranker, get_min_score
from app.rag.reranking.fake import FakeReranker
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval_pipeline import RetrievalPipeline


def _chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(),
    filename=f"doc{i}.txt", page=None, chunk_index=i, text=text, score=1.0 / (i+1))


class _StaticRetriever(Retriever):
    def __init__(self, results: list[RetrievedChunk]):
        self.results, self.calls = results, []

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((application_id, top_k))
        return self.results[:top_k]


RESULTS = [_chunk(i, "generic text about work") for i in range(9)] + [_chunk(9, "Kubernetes Helm GitOps")]\

# test pipeine
def test_pipeline_retrieves_n_and_keeps_k() -> None:
    retriever = _StaticRetriever(RESULTS)
    result = RetrievalPipeline(retriever, FakeReranker(), retrieval_top_k=10, rerank_top_k=3).run("Kubernetes Helm", "cv")

    assert retriever.calls == [("cv", 10)]
    assert len(result.candidates) == 10
    assert len(result.reranked) == 3
    assert result.reranked[0].chunk.chunk_index == 9        # hạng 10 -> hạng 1 nhờ rerank
    assert result.reranked[0].retrieval_rank == 10
    assert result.retrieval_ms >= 0 and result.rerank_ms >= 0

# Kiểm tra pipeline truyền domain (mô tả ứng dụng) cho reranker
def test_pipeline_passes_application_domain_to_reranker() -> None:
    seen: list[str | None] = []

    class _SpyReranker(FakeReranker):
        def rerank(self, query, chunks, top_k, domain=None):
            seen.append(domain)
            return super().rerank(query, chunks, top_k, domain)

    pipeline = RetrievalPipeline(_StaticRetriever(RESULTS), _SpyReranker(), 10, 3,
                                 domain_resolver=lambda app_id: f"docs of {app_id}")
    pipeline.run("q", "projects")
    assert seen == ["docs of projects"]

# từ chối rerank_top_k > retrieval_top_k
def test_pipeline_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RetrievalPipeline(_StaticRetriever([]), FakeReranker(),
        retrieval_top_k=5, rerank_top_k=10)

# test nếu đầu vào rỗng thì trả về list rỗng, không tốn lần gọi API
def test_pipeline_no_candidates() -> None:
    result = RetrievalPipeline(_StaticRetriever([]), FakeReranker(), 20, 5).run("q", "cv")
    assert result.candidates == [] and result.reranked == []

# thêm vào
def test_pipeline_filters_by_min_score() -> None:
    # FakeReranker: "Kubernetes Helm" khớp 2/2 từ với chunk 9 (điểm 1.0), 0/2 với các chunk khác (0.0)
    result = RetrievalPipeline(_StaticRetriever(RESULTS), FakeReranker(), 10, 3, min_score=0.5).run(
        "Kubernetes Helm", "cv")
    assert len(result.reranked) == 3                     # reranked vẫn đủ 3 (compare_retrieval dùng)
    assert [r.chunk.chunk_index for r in result.relevant] == [9]
    assert result.min_score == 0.5

# test Chunk judge = False thì cho qua luôn
def test_pipeline_does_not_filter_unjudged_chunks() -> None:
    result = RetrievalPipeline(_StaticRetriever(RESULTS), PassthroughReranker(), 10, 3, min_score=0.99).run(
        "q", "cv")
    assert len(result.relevant) == 3                     # điểm retrieval khác thang đo -> không lọc

# test min_score của mỗi provider
def test_min_score_defaults_per_provider() -> None:
    # Truyền reranker_min_score=None rõ ràng: nếu .env có RERANKER_MIN_SCORE thì Settings() sẽ đọc nó
    assert get_min_score(Settings(reranker_provider="gemini", reranker_min_score=None)) == 0.4
    assert get_min_score(Settings(reranker_provider="cohere", reranker_min_score=None)) == 0.1
    assert get_min_score(Settings(reranker_provider="none", reranker_min_score=None)) == 0.0
    assert get_min_score(Settings(reranker_provider="cohere", reranker_min_score=0.3)) == 0.3   # .env thắng