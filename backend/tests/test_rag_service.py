"""
MỤC ĐÍCH: Test RAGService từ đầu đến cuối bằng retriever + reranker + LLM giả (offline, không DB).

Kiểm tra các nhánh quan trọng:
  - trả lời bình thường: context có [Source n], trả về đúng nguồn được trích;
  - không chunk nào qua ngưỡng -> từ chối và KHÔNG gọi LLM;
  - LLM trả INSUFFICIENT_CONTEXT -> từ chối;
  - câu từ chối theo ngôn ngữ của câu hỏi;
  - chunk chưa được chấm (reranker fail-open) không bị ngưỡng loại;
  - mô tả application đi vào prompt; câu hỏi rỗng/quá dài bị từ chối.

NO_RELEVANT_CHUNKS = "no_relevant_chunks"   # không chunk nào qua ngưỡng -> không gọi LLM
LLM_INSUFFICIENT = "llm_insufficient"       # LLM đọc context rồi báo không đủ thông tin

test tên gì thì xem file đó, lỗi ở đâu xem file ở đó
"""

import uuid

import pytest

from app.rag.llm.fake import FakeLLM
from app.rag.prompts.answer import INSUFFICIENT_MARKER, NO_ANSWER_EN, NO_ANSWER_VI
from app.rag.reranking.base import PassthroughReranker
from app.rag.reranking.fake import FakeReranker
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.services.rag_service import LLM_INSUFFICIENT, NO_RELEVANT_CHUNKS, RAGService


def _chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), filename=f"doc{i}.txt",
                          page=None, chunk_index=i, text=text, score=1.0 / (i + 1))


class _StaticRetriever(Retriever):
    def __init__(self, results: list[RetrievedChunk]):
        self.results = results

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        return self.results[:top_k]


CHUNKS = [_chunk(0, "Hobbies: guitar and jazz"),
          _chunk(1, "Deployed services with Docker and Kubernetes"),
          _chunk(2, "Bachelor of Computer Science")]


def _service(llm: FakeLLM, reranker=None, min_score: float = 0.5, domain: str | None = None) -> RAGService:
    pipeline = RetrievalPipeline(_StaticRetriever(CHUNKS), reranker or FakeReranker(), 3, 3,
                                 domain_resolver=lambda app_id: domain, min_score=min_score)
    return RAGService(pipeline, llm)

#  trả lời bình thường: context có [Source n], trả về đúng nguồn được trích
def test_answers_with_cited_sources() -> None:
    llm = FakeLLM("Có, đã deploy bằng Docker [Source 1].")
    result = _service(llm).ask("Docker Kubernetes", "cv")

    assert not result.abstained and result.abstain_reason is None
    assert result.answer == "Có, đã deploy bằng Docker [Source 1]."
    assert [s.filename for s in result.sources] == ["doc1.txt"]
    assert "[Source 1]\nDocument: doc1.txt" in llm.calls[0]["prompt"]
    assert "Question:\nDocker Kubernetes" in llm.calls[0]["prompt"]


# không  có relevant -> llm.call rỗng -> không tốn token
def test_no_relevant_chunks_abstains_without_calling_llm() -> None:
    llm = FakeLLM()
    result = _service(llm).ask("Mức lương mong muốn là bao nhiêu?", "cv")   # không từ nào khớp -> điểm 0

    assert result.abstained and result.abstain_reason == NO_RELEVANT_CHUNKS
    assert result.answer == NO_ANSWER_VI
    assert result.sources == [] and result.context is None
    assert llm.calls == []                                                   # không tốn token

# LM trả  INSUFFICIENT_CONTEXT -> từ chối
def test_llm_insufficient_marker_abstains() -> None:
    llm = FakeLLM(INSUFFICIENT_MARKER)
    result = _service(llm).ask("Docker Kubernetes", "cv")
    assert result.abstained and result.abstain_reason == LLM_INSUFFICIENT
    assert len(llm.calls) == 1

# Từ chối theo đúng câu hỏi
def test_abstain_message_follows_question_language() -> None:
    result = _service(FakeLLM()).ask("What is the expected salary?", "cv")
    assert result.answer == NO_ANSWER_EN

# không so với ngưỡng nếu không được rerank do khác thang đo
def test_unjudged_chunks_are_not_filtered() -> None:
    # Passthrough = reranker lỗi rồi fail-open: điểm là điểm retrieval (nhỏ), KHÔNG được so với ngưỡng
    llm = FakeLLM()
    result = _service(llm, reranker=PassthroughReranker(), min_score=0.9).ask("Docker", "cv")
    assert not result.abstained
    assert len(llm.calls) == 1

# test mô tả application có trong system instruction
def test_application_domain_goes_into_system_instruction() -> None:
    llm = FakeLLM()
    _service(llm, domain="a person's CV and portfolio").ask("Docker Kubernetes", "cv")
    assert "a person's CV and portfolio" in llm.calls[0]["system_instruction"]

# test quên trích dẫn thì trả mọi nguồn
def test_uncited_answer_returns_all_context_sources() -> None:
    result = _service(FakeLLM("Có kinh nghiệm Docker."), min_score=0.0).ask("Docker Kubernetes", "cv")
    assert len(result.sources) == 3

# `@pytest.mark.parametrize` chạy cùng 1 test với 3 giá trị khác nhau (`""`, `"   "`, chuỗi 1001 ký tự), nên pytest đếm là **3 test**.
@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_invalid_question_raises(question: str) -> None:
    with pytest.raises(ValueError):
        _service(FakeLLM()).ask(question, "cv")