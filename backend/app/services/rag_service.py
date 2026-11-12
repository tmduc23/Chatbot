"""
MỤC ĐÍCH: RAGService - "nhạc trưởng" điều phối toàn bộ luồng hỏi đáp (spec mục 31: một orchestrator rõ ràng)

# Thêm _abstain cho mọi nhánh từ chối

    question
    -> kiểm tra application có tồn tại (Phase 8; không có -> ApplicationNotFoundError, HTTP 404)
    -> kiểm tra câu hỏi (rỗng, quá dài)
    -> RetrievalPipeline: hybrid search -> rerank -> lọc ngưỡng       (Phase 3-6)
    -> 0 chunk qua ngưỡng?  -> TỪ CHỐI, không gọi LLM                 (lớp bảo vệ 1: điểm reranker)
    -> context builder: loại trùng, giới hạn độ dài, [Source n]
    -> LLM trả lời CHỈ dựa trên context
    -> LLM báo INSUFFICIENT_CONTEXT? -> TỪ CHỐI                        (lớp bảo vệ 2: LLM tự đánh giá)
    -> kiểm tra trích dẫn: chuẩn hóa, xóa [Source n] không tồn tại     (Phase 7)
    -> (tùy chọn) grounding check: UNSUPPORTED? -> TỪ CHỐI             (lớp bảo vệ 3, Phase 7)
    -> answer + sources (chỉ những nguồn THẬT SỰ được trích)

    
Endpoint /chat ở Phase 9 chỉ cần gọi RAGService.ask() rồi trả result.to_response(), 
không chứa logic RAG nào (spec: "không đặt business logic trong route").


Vì sao 2 lớp? Ngưỡng điểm rẻ và chặn được câu hỏi lạc đề hoàn toàn. Nhưng có câu hỏi "gần đúng":
chunk nói về chủ đề dó (điểm cao) mà không chứa đáp án - lúc đó LLM là bên phát hiện được

(spec: "không đặt business logic trong route").

ask(question, application_id)
 ├─ validate_question            kiểm tra rỗng / quá dài
 ├─ pipeline.run                 hybrid 20 → rerank 5 → lọc ngưỡng   (Phase 3–6)
 ├─ relevant rỗng?               → TỪ CHỐI (no_relevant_chunks), KHÔNG gọi LLM
 ├─ build_context                [Source 1], [Source 2]…, tối đa 6000 ký tự
 ├─ llm.generate                 Gemini trả lời chỉ dựa trên context
 ├─ có INSUFFICIENT_CONTEXT?     → TỪ CHỐI (llm_insufficient)
 └─ _finish                      gói kết quả + nguồn + thời gian + log
"""

"""
MỤC ĐÍCH: RAGService — "nhạc trưởng" điều phối toàn bộ luồng hỏi đáp (spec mục 31: một orchestrator rõ ràng).

    question
      -> kiểm tra application có tồn tại (Phase 8; không có -> ApplicationNotFoundError, HTTP 404)
      -> kiểm tra câu hỏi (rỗng, quá dài)
      -> RetrievalPipeline: hybrid search -> rerank -> lọc ngưỡng       (Phase 3-6)
      -> 0 chunk qua ngưỡng?  -> TỪ CHỐI, không gọi LLM                 (lớp bảo vệ 1: điểm reranker)
      -> context builder: loại trùng, giới hạn độ dài, [Source n]
      -> LLM trả lời CHỈ dựa trên context
      -> LLM báo INSUFFICIENT_CONTEXT? -> TỪ CHỐI                        (lớp bảo vệ 2: LLM tự đánh giá)
      -> kiểm tra trích dẫn: chuẩn hóa, xóa [Source n] không tồn tại     (Phase 7)
      -> (tùy chọn) grounding check: UNSUPPORTED? -> TỪ CHỐI             (lớp bảo vệ 3, Phase 7)
      -> answer + sources (chỉ những nguồn THẬT SỰ được trích)

Endpoint /chat ở Phase 9 chỉ cần gọi RAGService.ask() rồi trả result.to_response(),
không chứa logic RAG nào (spec: "không đặt business logic trong route").
"""
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.metrics import LLM_TOKENS, RAG_STAGE_LATENCY
from app.core.exceptions import InvalidQuestionError
from app.rag.citations import clean_citations
from app.rag.context_builder import BuiltContext, Source, build_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.grounding import GroundingChecker, GroundingResult, get_grounding_checker
from app.rag.llm import LLMProvider, LLMResponse, get_llm
from app.rag.prompts.answer import (
    DEFAULT_ANSWER_PROMPT,
    INSUFFICIENT_MARKER,
    build_answer_prompt,
    build_answer_system_instruction,
    check_answer_prompt,
    is_insufficient,
    no_answer_message,
    remove_insufficient_marker,
)
from app.rag.prompts.grounding import UNSUPPORTED
from app.rag.reranking import Reranker
from app.rag.retrieval_pipeline import RetrievalPipeline, RetrievalResult, build_retrieval_pipeline
from app.repositories import application_repository as app_repo

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000

# Lý do từ chối — để log và đếm trong evaluation
NO_RELEVANT_CHUNKS = "no_relevant_chunks"   # không chunk nào qua ngưỡng -> không gọi LLM
LLM_INSUFFICIENT = "llm_insufficient"       # LLM đọc context rồi báo không đủ thông tin
GROUNDING_FAILED = "grounding_failed"       # grounding checker chấm UNSUPPORTED (Phase 7)


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    application_id: str
    answer: str
    sources: list[Source]            # CHỈ nguồn được trích hợp lệ trong câu trả lời
    abstained: bool                  # True = hệ thống từ chối trả lời
    abstain_reason: str | None       # NO_RELEVANT_CHUNKS | LLM_INSUFFICIENT | GROUNDING_FAILED | None
    retrieval: RetrievalResult       # giữ lại để debug: ứng viên, điểm rerank, ngưỡng
    context: BuiltContext | None     # None nếu từ chối trước khi gọi LLM
    llm_ms: float
    total_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    invalid_citations: list[int] = field(default_factory=list)   # số [Source n] LLM bịa ra, đã bị xóa
    grounding: GroundingResult | None = None                     # None = không chạy grounding check

    def to_response(self) -> dict:
        """Format trả về cho frontend (spec mục 13). Phase 9: endpoint /chat trả đúng dict này."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "abstained": self.abstained,
        }


def validate_question(question: str) -> str:
    q = question.strip()
    if not q:
        raise InvalidQuestionError("Question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise InvalidQuestionError(f"Question is too long (max {MAX_QUESTION_CHARS} characters)")
    return q


class RAGService:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        llm: LLMProvider,
        context_max_chars: int = 6000,
        grounding_checker: GroundingChecker | None = None,
        require_application: Callable[[str], object] | None = None,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT,
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.context_max_chars = context_max_chars
        self.grounding_checker = grounding_checker      # None = tắt grounding check
        # Phase 8: hàm raise ApplicationNotFoundError nếu application không tồn tại.
        # None = bỏ qua (unit test không có DB).
        self.require_application = require_application
        self.answer_prompt = check_answer_prompt(answer_prompt)   # "v3" | "v4" | "v5" (ANSWER_PROMPT trong .env)

    def ask(self, question: str, application_id: str,
            document_ids: Sequence[uuid.UUID] | None = None) -> RAGAnswer:
        """document_ids: None = hỏi cả application; [id...] = chế độ paper (chỉ tìm trong các tài liệu này)."""
        t0 = time.perf_counter()
        if self.require_application is not None:
            # Gõ sai --app (vd. "cvv") trước đây vẫn chạy: retrieval rỗng -> "không có thông tin".
            # Giờ báo lỗi rõ ràng, và không tốn lượt gọi embedding/LLM.
            self.require_application(application_id)
        q = validate_question(question)

        scope = {"document_ids": document_ids} if document_ids else {}
        retrieval = self.pipeline.run(q, application_id, **scope)
        if not retrieval.relevant:
            return self._abstain(q, application_id, retrieval, None, t0, 0.0, NO_RELEVANT_CHUNKS)

        context = build_context(retrieval.relevant, self.context_max_chars)
        # Cùng mô tả application mà reranker đã dùng (1 query DB nhỏ, chấp nhận gọi lại)
        domain = self.pipeline.domain_resolver(application_id) if self.pipeline.domain_resolver else None

        t1 = time.perf_counter()
        response = self.llm.generate(build_answer_prompt(q, context.text, self.answer_prompt),
                                     build_answer_system_instruction(domain, self.answer_prompt))
        llm_ms = (time.perf_counter() - t1) * 1000

        # --- Lớp 2: LLM báo thiếu thông tin ---
        if is_insufficient(response.text):
            return self._abstain(q, application_id, retrieval, context, t0, llm_ms, LLM_INSUFFICIENT, response)

        # --- Trích dẫn (Phase 7) ---
        had_marker = INSUFFICIENT_MARKER in response.text.upper()
        check = clean_citations(remove_insufficient_marker(response.text),
                                valid={s.number for s in context.sources})
        if check.invalid:
            logger.warning("answer cited sources that do not exist: %s (context has %d sources)",
                           check.invalid, len(context.sources))
        if had_marker and not check.cited:
            # vd. "Mình không chắc chắn. INSUFFICIENT_CONTEXT" -> thực chất là từ chối
            return self._abstain(q, application_id, retrieval, context, t0, llm_ms, LLM_INSUFFICIENT, response)
        if not check.cited:
            logger.warning("answer has no valid citation -> returned without sources")
        sources = [s for s in context.sources if s.number in check.cited]

        # --- Lớp 3 (tùy chọn): grounding check ---
        grounding = None
        if self.grounding_checker is not None:
            grounding = self.grounding_checker.check(check.text, context.text)
            logger.info("grounding verdict=%s ms=%.0f reason=%s", grounding.verdict, grounding.ms, grounding.reason)
            if grounding.verdict == UNSUPPORTED:
                return self._abstain(q, application_id, retrieval, context, t0, llm_ms, GROUNDING_FAILED,
                                     response, check.invalid, grounding)

        return self._finish(q, application_id, check.text, sources, None, retrieval, context, t0, llm_ms,
                            response, check.invalid, grounding)

    def _abstain(
        self,
        q: str,
        application_id: str,
        retrieval: RetrievalResult,
        context: BuiltContext | None,
        t0: float,
        llm_ms: float,
        reason: str,
        response: LLMResponse | None = None,
        invalid: list[int] | None = None,
        grounding: GroundingResult | None = None,
    ) -> RAGAnswer:
        return self._finish(q, application_id, no_answer_message(q), [], reason, retrieval, context, t0, llm_ms,
                            response, invalid or [], grounding)

    def _finish(
        self,
        q: str,
        application_id: str,
        answer: str,
        sources: list[Source],
        abstain_reason: str | None,
        retrieval: RetrievalResult,
        context: BuiltContext | None,
        t0: float,
        llm_ms: float,
        response: LLMResponse | None,
        invalid: list[int],
        grounding: GroundingResult | None,
    ) -> RAGAnswer:
        result = RAGAnswer(
            question=q, application_id=application_id, answer=answer, sources=sources,
            abstained=abstain_reason is not None, abstain_reason=abstain_reason,
            retrieval=retrieval, context=context, llm_ms=llm_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            input_tokens=getattr(response, "input_tokens", None),
            output_tokens=getattr(response, "output_tokens", None),
            invalid_citations=invalid, grounding=grounding,
        )
        logger.info("rag_answer app=%s abstained=%s reason=%s sources=%d invalid_citations=%d grounding=%s "
                    "prompt=%s retrieval_ms=%.0f rerank_ms=%.0f llm_ms=%.0f total_ms=%.0f",
                    application_id, result.abstained, abstain_reason, len(sources), len(invalid),
                    grounding.verdict if grounding else "off", f"answer-{self.answer_prompt}",
                    retrieval.retrieval_ms, retrieval.rerank_ms, llm_ms, result.total_ms)
        # Phase 12d: metrics cho Prometheus (ms -> giây, đơn vị chuẩn của Prometheus)
        RAG_STAGE_LATENCY.labels("retrieval").observe(retrieval.retrieval_ms / 1000)
        RAG_STAGE_LATENCY.labels("rerank").observe(retrieval.rerank_ms / 1000)
        if response is not None:
            RAG_STAGE_LATENCY.labels("llm").observe(llm_ms / 1000)
        RAG_STAGE_LATENCY.labels("total").observe(result.total_ms / 1000)
        if result.input_tokens:
            LLM_TOKENS.labels("input").inc(result.input_tokens)
        if result.output_tokens:
            LLM_TOKENS.labels("output").inc(result.output_tokens)
        return result


def build_rag_service(
    db: Session,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    llm: LLMProvider | None = None,
    min_score: float | None = None,
) -> RAGService:
    s = settings or get_settings()
    return RAGService(
        pipeline=build_retrieval_pipeline(db, s, embedder=embedder, reranker=reranker, min_score=min_score),
        llm=llm or get_llm(s),
        context_max_chars=s.context_max_chars,
        grounding_checker=get_grounding_checker(s),
        require_application=lambda application_id: app_repo.require(db, application_id),
        answer_prompt=s.answer_prompt,
    )
