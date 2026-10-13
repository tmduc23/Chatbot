"""
MỤC ĐÍCH: Unit test cho Reciprocal Rank Fusion và HybridRetriever (dùng retriever gải, không cần DB)
"""
import uuid

import pytest

from app.core.exceptions import EmbeddingError
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.hybrid_retriever import HybridRetriever

def _chunk(name: str, score: float = 0.0) -> RetrievedChunk:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, name)      # cùng name -> cùng id
    return RetrievedChunk(chunk_id=cid, document_id=cid, filename=name, page=None,
                          chunk_index=0, text=name, score=score)


A, B, C, D = (_chunk(n) for n in "ABCD")

# test công thức
def test_rrf_formula() -> None:
    fused = reciprocal_rank_fusion({"vector": [A, B], "keyword": [C, A]}, k=60)
    by_name = {f.filename: f for f in fused}
    assert by_name["A"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_name["B"].score == pytest.approx(1 / 62)
    assert by_name["C"].score == pytest.approx(1 / 61)

# chunk có ở cả 2 nhánh lên đầu
def test_chunk_found_by_both_ranks_first() -> None:
    fused = reciprocal_rank_fusion({"vector": [B, A, C], "keyword": [D, A]}, k=60)
    assert fused[0].filename == "A"

# không trùng chunk và ranks
def test_no_duplicates_and_ranks_recorded() -> None:
    fused = reciprocal_rank_fusion({"vector": [A, B], "keyword": [A, B]}, k=60)
    assert [f.filename for f in fused] == ["A", "B"]
    assert fused[0].ranks == {"vector": 1, "keyword": 1}

# hòa điểm thì không mất chunk nào
def test_tie_broken_by_best_rank() -> None:
    # B hạng 1 ở keyword, C hạng 1 ở vector -> cùng điểm 1/61, không bị mất chunk nào
    fused = reciprocal_rank_fusion({"vector": [C], "keyword": [B]}, k=60)
    assert {f.filename for f in fused} == {"B", "C"}
    assert fused[0].score == fused[1].score

def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []

def test_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"vector": [A]}, k=0)

# ---------- HybridRetriever ----------
# Giả lập vector
# self.fail: xác định retriever có giả lập lỗi hay không
class _StaticRetriever(Retriever):
    def __init__(self, results: list[RetrievedChunk], fail: bool = False):
        self.results, self.fail, self.calls = results, fail, []

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, application_id, top_k))
        if self.fail:
            raise EmbeddingError("quota exceeded")
        return self.results[:top_k]

"""
Vector Retriever

rank 1 → B
rank 2 → A
rank 3 → C

vector.results	Vector Retriever sẽ trả về kết quả gì
vector.calls	Vector Retriever đã được gọi như thế nào
vector.fail	Có giả lập lỗi hay không

Keyword Retriever

rank 1 → D
rank 2 → A
"""

# gộp đúng và truyền đúng
def test_hybrid_fuses_and_passes_application_id() -> None:
    vector, keyword = _StaticRetriever([B, A, C]), _StaticRetriever([D, A])
    results = HybridRetriever(vector, keyword, candidate_k=10).retrieve("q", "cv", top_k=3)
    assert results[0].filename == "A"       # kết quả rank 1
    assert len(results) == 3                # kết quả độ dài result
    assert vector.calls == [("q", "cv", 10)] and keyword.calls == [("q", "cv", 10)]     # lưu lại lịch sử các lần gọi

# nhánh vector llooix thì vẫn trả kết quả keyword
def test_hybrid_falls_back_to_keyword_when_embedding_fails() -> None:
    results = HybridRetriever(_StaticRetriever([A], fail=True), _StaticRetriever([D])).retrieve("q", "cv", 5)
    assert [r.filename for r in results] == ["D"]