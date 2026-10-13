"""
MỤC ĐÍCH: Unit test cho tầng reranking - KHÔNG gọi Gemini thật

GeminiReranker được kiểm tra bằng stub client trả về JSON giả, để xác nhận logic:
    - prompt chứa câu hỏi + đủ passage_id, chunk bị cắt theo max_chars
    - parse JSON -> sắp xếp lại đúng, giữ top_k, điểm đổi về 0-1
    - id bị thiếu -> điểm 0; JSON hỏng -> fail-open, giữ thứ tự cũ (hoặc raise nếu fail_open=False)
    - retry khi HTTP 429; danh sách rỗng -> không gọi API.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import RerankError
from app.rag.reranking import PassthroughReranker, get_reranker
from app.rag.reranking.base import rank_by_scores
from app.rag.reranking.fake import FakeReranker
from app.rag.reranking.gemini import GeminiReranker
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


# ---------- helpers & simple rerankers ----------
def test_rank_by_scores_sorts_and_keeps_retrieval_rank() -> None:
    result = rank_by_scores(CHUNKS, [0.1, 0.5, 0.9], top_k=2)
    assert [r.chunk.filename for r in result] == ["rag", "edu"]
    assert [r.retrieval_rank for r in result] == [3, 2]

def test_rank_by_scores_tie_keeps_original_order() -> None:
    result = rank_by_scores(CHUNKS, [0.5, 0.5, 0.5], top_k=3)
    assert [r.retrieval_rank for r in result] == [1, 2, 3]

def test_passthrough_keeps_order() -> None:
    result = PassthroughReranker().rerank("q", CHUNKS, top_k=2)
    assert [r.chunk.filename for r in result] == ["hobby", "edu"]

def test_fake_reranker_promotes_matching_chunk() -> None:
    result = FakeReranker().rerank("RAG chatbot Gemini", CHUNKS, top_k=1)
    assert result[0].chunk.filename == "rag"


# ---------- GeminiReranker với stub ---------
# SimpleNameSpace là cách để tạo object rỗng có thuộc tính tùy ý, không phải viết class

class _StubModels:
    def __init__(self, responses: list):
        self.responses = list(responses)                # mỗi phần tử: str (JSON) hoặc exception
        self.calls: list[dict] = []                     # ghi lại mỗi lần bị gọi (model, prompt, config), để test kiểm tra reranker đã gửi gì

    def generate_content(self, model: str, contents: str, config: object) -> SimpleNamespace:
        self.calls.append({"model": model, "contents": contents, "config": config})     # ghi lại model, content, config
        item = self.responses.pop(0)            # lấy câu trả lời đầu tiên và xóa nó khỏi danh saschs
        if isinstance(item, Exception):         # nếu lỗi thì ném lỗi đó ra
            raise item
        return SimpleNamespace(text=item, usage_metadata=None)

# Tạo reranker gắn Gemini giả
def _reranker(responses: list, **kwargs) -> tuple[GeminiReranker, _StubModels]:
    stub = _StubModels(responses)
    return GeminiReranker(api_key="test", model="gemini-test", client=SimpleNamespace(models=stub), **kwargs), stub

# viết nhanh câu trả lời JSON giả
def _json(scores: dict[int, float]) -> str:
    return json.dumps([{"id": i, "score": s} for i, s in scores.items()])


def test_gemini_reorders_by_llm_scores() -> None:
    reranker, stub = _reranker([_json({0: 1, 1: 4, 2: 10})])
    result = reranker.rerank("Dự án RAG nào?", CHUNKS, top_k=2)

    assert [r.chunk.filename for r in result] == ["rag", "edu"]
    assert result[0].score == pytest.approx(1.0)          # 10/10
    assert result[0].retrieval_rank == 3                  # được kéo từ hạng 3 lên 1
    prompt = stub.calls[0]["contents"]
    assert "Dự án RAG nào?" in prompt
    assert all(f'<passage id="{i}">' in prompt for i in range(3))
    assert stub.calls[0]["config"].temperature == 0.0

# cắt chunk dài
def test_gemini_truncates_passages() -> None:
    long_chunk = _chunk("long", "x" * 5000)
    reranker, stub = _reranker([_json({0: 5})], max_chars=100)
    reranker.rerank("q", [long_chunk], top_k=1)
    assert "x" * 101 not in stub.calls[0]["contents"]

# Gemini trả thiếu điểm
def test_gemini_missing_ids_get_zero() -> None:
    reranker, _ = _reranker([_json({2: 8})])
    result = reranker.rerank("q", CHUNKS, top_k=3)
    assert result[0].chunk.filename == "rag"
    assert [r.score for r in result[1:]] == [0.0, 0.0]

#  JSON hỏng, giữ thứ tự cũ
def test_gemini_invalid_json_fails_open() -> None:
    reranker, _ = _reranker(["not json at all"])
    result = reranker.rerank("q", CHUNKS, top_k=2)
    assert [r.chunk.filename for r in result] == ["hobby", "edu"]   # thứ tự retrieval giữ nguyên

#  JSON hỏng, báo lỗi

def test_gemini_invalid_json_raises_when_fail_closed() -> None:
    reranker, _ = _reranker(["not json"], fail_open=False)
    with pytest.raises(RerankError):
        reranker.rerank("q", CHUNKS, top_k=2)

# bị 429 thì thử lại
def test_gemini_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.rag.reranking.gemini.time.sleep", lambda s: None)
    err = RuntimeError("rate limited")
    err.code = 429  # type: ignore[attr-defined]
    reranker, stub = _reranker([err, _json({0: 2, 1: 9, 2: 1})])
    result = reranker.rerank("q", CHUNKS, top_k=1)
    assert result[0].chunk.filename == "edu"
    assert len(stub.calls) == 2

# Kiểm tra prompt sử dụng domain truyền vào
def test_gemini_prompt_uses_appilcation_domain() -> None:
    reranker, stub = _reranker([_json({0: 5, 1:5, 2:5})])
    reranker.rerank("q", CHUNKS, top_k=1, domain="software project READMEs")
    assert "software project READMEs" in stub.calls[0]["config"].system_instruction

# Kiểm tra gemini sử dụng domain mặc định không truyền
def test_gemini_prompt_default_domain() -> None:
    reranker, stub = _reranker([_json({0: 5, 1: 5, 2: 5})])
    reranker.rerank("q", CHUNKS, top_k=1)
    assert "a document collection" in stub.calls[0]["config"].system_instruction

def test_gemini_empty_input_does_not_call_api() -> None:
    reranker, stub = _reranker([])
    assert reranker.rerank("q", [], top_k=5) == []
    assert stub.calls == []


# ---------- factory ----------

# Factory chọn đúng class
def test_factory_none_and_fake() -> None:
    assert isinstance(get_reranker(Settings(reranker_provider="none")), PassthroughReranker)
    assert isinstance(get_reranker(Settings(reranker_provider="fake")), FakeReranker)

# Tên không tồn tại phải raise ValueError
def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_reranker(Settings(reranker_provider="magic")) 