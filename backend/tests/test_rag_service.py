"""
MỤC ĐÍCH: Test RAGService từ đầu đến cuối bằng retriever + reranker + LLM giả (offline, không DB).

Kiểm tra các nhánh quan trọng:
  - trả lời bình thường: context có [Source n], trả về đúng nguồn được trích;
  - không chunk nào qua ngưỡng -> từ chối và KHÔNG gọi LLM;
  - LLM trả INSUFFICIENT_CONTEXT -> từ chối;
  - câu từ chối theo ngôn ngữ của câu hỏi;
  - chunk chưa được chấm (reranker fail-open) không bị ngưỡng loại;
  - mô tả application đi vào prompt; câu hỏi rỗng/quá dài bị từ chối.

Phase 7 thêm:
    - [Source n] không tồn tại bị xoas, [Source 1 3] được chuẩn hóa; không trích -> không có nguồn;
    - trả lời một phần + INSUFFICIENT_CONTEXT; format to_response() theo mục 13
    - grounding check (judge giả): UNSUPPORTED -> từ chối, PARTIAL -> giữ, không chạy khi đã từ chối
NO_RELEVANT_CHUNKS = "no_relevant_chunks"   # không chunk nào qua ngưỡng -> không gọi LLM
LLM_INSUFFICIENT = "llm_insufficient"       # LLM đọc context rồi báo không đủ thông tin

test tên gì thì xem file đó, lỗi ở đâu xem file ở đó
"""

import json
import uuid

import pytest

from app.rag.llm.fake import FakeLLM
from app.rag.prompts.answer import INSUFFICIENT_MARKER, NO_ANSWER_EN, NO_ANSWER_VI
from app.rag.reranking.base import PassthroughReranker
from app.rag.reranking.fake import FakeReranker
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.rag.retrieval_pipeline import RetrievalPipeline
from app.rag.grounding import GroundingChecker
from app.services.rag_service import GROUNDING_FAILED, LLM_INSUFFICIENT, NO_RELEVANT_CHUNKS, RAGService


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


def _service(llm: FakeLLM, reranker=None, min_score: float = 0.5, domain: str | None = None,
             judge: FakeLLM | None = None) -> RAGService:
    pipeline = RetrievalPipeline(_StaticRetriever(CHUNKS), reranker or FakeReranker(), 3, 3,
                                 domain_resolver=lambda app_id: domain, min_score=min_score)
    return RAGService(pipeline, llm, grounding_checker=GroundingChecker(judge) if judge else None)

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

# LLM trả  INSUFFICIENT_CONTEXT -> từ chối
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
    assert len(result.retrieval.relevant) == 3      # không chunk nào bị ngưỡng 0.9 loại

# test mô tả application có trong system instruction
def test_application_domain_goes_into_system_instruction() -> None:
    llm = FakeLLM()
    _service(llm, domain="a person's CV and portfolio").ask("Docker Kubernetes", "cv")
    assert "a person's CV and portfolio" in llm.calls[0]["system_instruction"]

# --- Phase 7: trích dẫn ---

def test_uncited_answer_has_no_sources() -> None:
    # Phase 6 trả MỌI nguồn khi LLM quên trích. Phase 7: không biết ý nào lấy từ đâu -> không gắn nguồn
    result = _service(FakeLLM("Có kinh nghiệm Docker."), min_score=0.0).ask("Docker Kubernetes", "cv")
    assert not result.abstained
    assert result.sources == []

def test_invalid_citation_is_removed() -> None:
    result = _service(FakeLLM("Đã deploy bằng Docker [Source 1][Source 7].")).ask("Docker Kubernetes", "cv")
    assert result.answer == "Đã deploy bằng Docker [Source 1]."     # context chỉ có 1 nguồn -> [Source 7] bị xóa
    assert result.invalid_citations == [7]
    assert [s.filename for s in result.sources] == ["doc1.txt"]

def test_combined_citation_maps_each_source() -> None:
    # min_score=0 -> cả 3 chunk vào context: Source 1 = doc1 (điểm 1.0), Source 2 = doc0, Source 3 = doc2
    result = _service(FakeLLM("Docker [Source 1, 3]."), min_score=0.0).ask("Docker Kubernetes", "cv")
    assert result.answer == "Docker [Source 1][Source 3]."
    assert [(s.number, s.filename) for s in result.sources] == [(1, "doc1.txt"), (3, "doc2.txt")]

def test_partial_answer_drops_marker_and_keeps_text() -> None:
    result = _service(FakeLLM(f"Đã dùng Docker [Source 1].\n{INSUFFICIENT_MARKER}")).ask("Docker Kubernetes", "cv")
    assert not result.abstained
    assert result.answer == "Đã dùng Docker [Source 1]."

def test_marker_without_citation_abstains() -> None:
    result = _service(FakeLLM(f"Mình không chắc. {INSUFFICIENT_MARKER}")).ask("Docker Kubernetes", "cv")
    assert result.abstained and result.abstain_reason == LLM_INSUFFICIENT


def test_to_response_matches_api_format() -> None:
    data = _service(FakeLLM("Docker [Source 1].")).ask("Docker Kubernetes", "cv").to_response()
    assert data["answer"] == "Docker [Source 1]." and data["abstained"] is False
    assert data["sources"][0]["filename"] == "doc1.txt"
    assert set(data["sources"][0]) == {"number", "document_id", "filename", "page", "chunk_id", "text"}
    json.dumps(data)                 # chuyển được sang JSON (không còn UUIDs)

# --- Phase 7: grounding check (tùy chọn) ---

def test_grounding_is_off_by_default() -> None:
    assert _service(FakeLLM("Docker [Source 1].")).ask("Docker Kubernetes", "cv").grounding is None

def test_grounding_unsupported_abstains() -> None:
    llm, judge = FakeLLM("Có 5 năm kinh nghiệm Docker [Source 1]."), FakeLLM("UNSUPPORTED\n5 years is not stated.")
    result = _service(llm, judge=judge).ask("Docker Kubernetes", "cv")

    assert result.abstained and result.abstain_reason == GROUNDING_FAILED
    assert result.sources == []
    assert result.grounding.verdict == "UNSUPPORTED"
    assert "ANSWER:\nCó 5 năm kinh nghiệm Docker [Source 1]." in judge.calls[0]["prompt"]
    assert "Deployed services with Docker and Kubernetes" in judge.calls[0]["prompt"]   # judge thấy context

def test_grounding_partial_keeps_answer() -> None:
    judge = FakeLLM("PARTIAL\nKubernetes version is not stated.")
    result = _service(FakeLLM("Docker [Source 1]."), judge=judge).ask("Docker Kubernetes", "cv")
    assert not result.abstained
    assert result.grounding.verdict == "PARTIAL"
    assert [s.filename for s in result.sources] == ["doc1.txt"]

def test_grounding_not_called_when_already_abstained() -> None:
    judge = FakeLLM("SUPPORTED\nOK")
    result = _service(FakeLLM(), judge=judge).ask("Mức lương mong muốn là bao nhiêu?", "cv")
    assert result.abstain_reason == NO_RELEVANT_CHUNKS
    assert judge.calls == []                          # không tốn thêm request


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_invalid_question_raises(question: str) -> None:
    with pytest.raises(ValueError):
        _service(FakeLLM()).ask(question, "cv")