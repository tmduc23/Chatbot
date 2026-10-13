"""
MỤC ĐÍCH: Unit test cho CohereReranker — KHÔNG gọi Cohere thật (không tốn quota, không cần mạng).

Client được thay bằng stub có hàm rerank(...) trả kết quả giả, để kiểm tra logic của mình:
  - gửi đúng model, câu hỏi, document đã cắt theo max_chars, top_n = tất cả;
  - sắp xếp lại theo relevance_score dựa vào index, giữ top_k và hạng cũ;
  - index bị thiếu -> điểm 0; index lạ -> bỏ qua;
  - retry khi HTTP 429; lỗi khác -> fail-open giữ thứ tự cũ (hoặc raise nếu fail_open=False);
  - danh sách rỗng -> không gọi API; factory chọn đúng class và báo lỗi khi thiếu key.
"""

import uuid
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import RerankError
from app.rag.reranking import cohere_reranker as cohere_module
from app.rag.reranking import get_reranker
from app.rag.reranking.cohere_reranker import CohereReranker
from app.rag.retrieval.base import RetrievedChunk



def _chunk(name: str, text: str, score: float = 0.0) -> RetrievedChunk:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    return RetrievedChunk(chunk_id=cid, document_id=cid, filename=name, page=None,
                          chunk_index=0, text=text, score=score)


CHUNKS = [
    _chunk("hobby", "Hobbies: guitar and jazz.", 0.03),
    _chunk("edu", "Bachelor of Computer Science, HUST, GPA 3.6.", 0.02),
    _chunk("rag", "Built a RAG chatbot with Gemini, FastAPI and pgvector.", 0.01),
]


class _HttpError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code

def _response(pairs: list[tuple[int, float]]) -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


class _StubClient:
    def __init__(self, responses: list):
        self.responses = list(responses)       # mỗi phần tử: response giả hoặc Exception
        self.calls: list[dict] = []

    def rerank(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

def _reranker(responses: list, **kwargs) -> tuple[CohereReranker, _StubClient]:
    stub = _StubClient(responses)
    return CohereReranker(api_key="test", model="rerank-multilingual-v3.0", client=stub, **kwargs), stub


# Kiểm tra gửi đúng model, query, documents bị cắt, top_n = tất cả
def test_sends_model_query_and_truncated_documents() -> None:
    reranker, stub = _reranker([_response([(0, 0.1), (1, 0.2), (2, 0.3)])], max_chars=10)
    reranker.rerank("RAG?", CHUNKS, top_k=2)
    call = stub.calls[0]
    assert call["model"] == "rerank-multilingual-v3.0"
    assert call["query"] == "RAG?"
    assert call["documents"] == [c.text[:10] for c in CHUNKS]
    assert call["top_n"] == 3

# Kiểm tra sắp xếp lại theo đúng relevance_score
def test_reorders_by_relevance_score() -> None:
    reranker, _ = _reranker([_response([(2, 0.97), (1, 0.40), (0, 0.05)])])
    result = reranker.rerank("RAG?", CHUNKS, top_k=2)
    assert [r.chunk.filename for r in result] == ["rag", "edu"]
    assert [r.retrieval_rank for r in result] == [3, 2]
    assert result[0].score == pytest.approx(0.97)

# Inddex bị thiếu -> điểm 0, index lạ (ngoài phạm vi) -> bỏ qua
def test_missing_index_gets_zero_and_unknown_index_is_ignored() -> None:
    reranker, _ = _reranker([_response([(1, 0.8), (99, 1.0)])])
    result = reranker.rerank("q", CHUNKS, top_k=3)
    assert result[0].chunk.filename == "edu"
    assert [r.score for r in result[1:]] == [0.0, 0.0]

# Nếu gặp lỗi 429(rare limit) thì thử lại sau một khoảng thời gian
def test_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohere_module.time, "sleep", lambda s: None)
    reranker, stub = _reranker([_HttpError(429), _response([(2, 0.9), (0, 0.1), (1, 0.1)])])
    result = reranker.rerank("q", CHUNKS, top_k=1)
    assert len(stub.calls) == 2
    assert result[0].chunk.filename == "rag"

# Lỗi khác (ví dụ 401) -> fail-open, giữ thứ tự retrieval ban đầu
def test_api_error_falls_back_to_retrieval_order() -> None:
    reranker, stub = _reranker([_HttpError(401)])
    result = reranker.rerank("q", CHUNKS, top_k=2)
    assert len(stub.calls) == 1                       # 401 (sai key) không retry
    assert [r.chunk.filename for r in result] == ["hobby", "edu"]


# Nếu fail_open = False thì ném RerankError khi có lỗi
def test_api_error_raises_when_fail_closed() -> None:
    reranker, _ = _reranker([_HttpError(401)], fail_open=False)
    with pytest.raises(RerankError):
        reranker.rerank("q", CHUNKS, top_k=2)

# Danh sách rỗng -> không gọi API
def test_empty_candidates_do_not_call_api() -> None:
    reranker, stub = _reranker([])
    assert reranker.rerank("q", [], top_k=5) == []
    assert stub.calls == []

# Factory tạo đúng CohereReranker khi cấu hình hợp lệ
def test_factory_build_cohere() -> None:
    reranker = get_reranker(Settings(reranker_provider="cohere",
    cohere_api_key="test_key"))
    assert isinstance(reranker, CohereReranker)
    assert reranker.model == "rerank-multilingual-v3.0"

# Thiếu API key -> báo lỗi
def test_factory_cohere_requires_key() -> None:
    with pytest.raises(ValueError):
        get_reranker(Settings(reranker_provider="cohere",
        cohere_api_key=None))
