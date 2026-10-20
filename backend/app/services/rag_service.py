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

import logging
import re
import time
from dataclasses import dataclass, field

from collections.abc import Callable
from sqlalchemy.orm import Session 

from app.core.exceptions import InvalidQuestionError
from app.repositories import application_repository as app_repo
from app.rag.citations import clean_citations
from app.core.config import Settings, get_settings
from app.rag.context_builder import BuiltContext, Source, build_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.grounding import GroundingChecker, GroundingResult, get_grounding_checker
from app.rag.llm import LLMProvider, LLMResponse, get_llm
from app.rag.prompts.answer import (
    ANSWER_PROMPT_VERSION,
    INSUFFICIENT_MARKER,
    build_answer_prompt,
    build_answer_system_instruction,
    is_insufficient,
    no_answer_message,
    remove_insufficient_marker,
)

from app.rag.prompts.grounding import UNSUPPORTED
from app.rag.reranking import Reranker
from app.rag.retrieval_pipeline import RetrievalPipeline, RetrievalResult, build_retrieval_pipeline

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000

# Lý do từ chối — để log và đếm trong evaluation
NO_RELEVANT_CHUNKS = "no_relevant_chunks"   # không chunk nào qua ngưỡng -> không gọi LLM
LLM_INSUFFICIENT = "llm_insufficient"       # LLM đọc context rồi báo không đủ thông tin
GROUNDING_FAILED = "grounding_failed"       # grounding checker chấm UNSUPPORTED  (Phase 7)

_CITATION = re.compile(r"\[Source (\d+)\]")


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    application_id: str
    answer: str
    sources: list[Source]                   # CHỈ nguồn được trích hợp lệ trong câu trả lời
    abstained: bool                         # True = hệ thống từ chối trả lời
    abstain_reason: str | None              # NO_RELEVANT_CHUNKS | LLM_INSUFFICIENT | GROUNDING_FAILED | None
    retrieval: RetrievalResult              # giữ lại để debug: ứng viên, điểm rank, ngưỡng
    context: BuiltContext | None            # None nếu từ chối trước khi gọi LLM
    llm_ms: float
    total_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    invalid_citations: list[int] = field(default_factory=list)      # số [Source n] LLM bịa ra, đã bị xóa
    grounding: GroundingResult | None = None                        # None = không chạy grounding check

    def to_response(self) -> dict:
        """Format trả về cho frontend (spec mục 13). Phase 9: endpoint /chat trả đúng dict này."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "abstained": self.abstained,
        }


def validate_question(questrion: str) -> str:
    q = questrion.strip()
    if not q:
        raise InvalidQuestionError("Question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise InvalidQuestionError(f"Question is too long (max {MAX_QUESTION_CHARS} characters)")
    return q

# # Trả về các chuỗisó trong cited number
# def cited_numbers(answer: str) -> set[int]:
#     """'... [source 2] ... [Source 1][Source 2]' -> {1, 2}"""
#     return {int(n) for n in _CITATION.findall(answer)}

# Retrieva Pipelline là rerank xong
class RAGService:
    def __init__(
        self, pipeline: RetrievalPipeline, 
        llm: LLMProvider, 
        context_max_chars: int = 6000, 
        grounding_checker: GroundingChecker | None = None,
        require_application: Callable[[str], object] | None = None,
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.context_max_chars = context_max_chars
        self.grounding_checker = grounding_checker     # None = tắt grounding check

        # Phase 8: hàm raise ApplicationNotFoundError nếu application không tồn tại
        # None = bỏ qua (unit test không có DB)
        self.require_application = require_application

    def ask(self, question: str, application_id: str):
        t0 = time.perf_counter()

        if self.require_application is not None:
            # Gõ sai --app (vd. "cvv") trước đây vẫn chạy: retrieval rỗng -> "không có thông tin"
            # Giờ báo lỗi rõ ràng, và không tốn lượt gọi embedding/LLM
            self.require_application(application_id)
        q = validate_question(question)

        # thực hiện retrieve+rerank
        retrieval = self.pipeline.run(q, application_id)
        # relevant gồm chunk được reranker chấm điểm và điểm >= ngưỡng; chunk không được reranker chấm: luôn được giữ

        # Lớp từ chối 1, không chunk nào qua ngưỡng thì trả về ngay, context=0.0, llm_ms=0.0 -> không tốn request LM, không cho LLM cơ hội bịa
        if not retrieval.relevant:
            return self._abstain(q, application_id, retrieval, None, t0, 0.0, NO_RELEVANT_CHUNKS)

        context = build_context(retrieval.relevant, self.context_max_chars)     # context cho LLM

        # Cùng mô tả application mà reranker đã đùng (1 query DB nhỏ, chấp nhận gọi lại)
        domain = self.pipeline.domain_resolver(application_id) if self.pipeline.domain_resolver else None

        t1 = time.perf_counter()
        # generate: prompt, system_instruction, phản hồi của LLM
        response = self.llm.generate(build_answer_prompt(q, context.text), build_answer_system_instruction(domain))

        llm_ms = (time.perf_counter() - t1) * 1000

        # --- Lớp 2: LLM báo thiếu thông tin ---
        if is_insufficient(response.text):
            return self._abstain(q, application_id, retrieval, context, t0, llm_ms, LLM_INSUFFICIENT, response)

        # Trích dẫn 
        # Ghi nhớ xem LLM có viết INSUFFICIENT_CONTEXT ở bất kỳ đâu không
        # Xử lý những câu trả lời "lưng chừng" và quyết định 3 việc: từ chối hay giữ, gắn nguồn nào và ghi log cảnh báo gì.
        had_marker = INSUFFICIENT_MARKER in response.text.upper()       
        check = clean_citations(remove_insufficient_marker(response.text),
                                valid={s.number for s in context.sources})      # text, cited, invalid

        # Check có nguồn bịa, vẫn giữ câu trả lời, logging cảnh báo
        if check.invalid:
            logger.warning("answer cited sources that do not exist: %s (context has %d sources)",
                           check.invalid, len(context.sources))
        # Có marker và không có cite nào
        if had_marker and not check.cited:
            # Vd: "Mình không chăc chắn. INSUFFICIENT_CONTEXT" -> thực chất là từ chối
            return self._abstain(q, application_id, retrieval, context, t0, llm_ms, LLM_INSUFFICIENT, response)     # abstain_reason=LLM_INSUFFICIENT

        # không có marker -> LLM trả lời quên trích dẫn
        if not check.cited:
            logger.warning("answer has no valid citation -> returned without sources")

        # sources những nguồn thông tin hợp lệ
        sources = [s for s in context.sources if s.number in check.cited]

        # Lớp 3: grounding check
        grounding = None
        if self.grounding_checker is not None:
            # self.llm.generate(build_grounding_prompt(answer, context), GROUNDING_SYSTEM_INSTRUCTION) dòng gọi check ở grounding
            grounding = self.grounding_checker.check(check.text, context.text)
            logger.info("grounding verdict=%s ms=%.0f reason=%s", grounding.verdict, grounding.ms, grounding.reason)
            if grounding.verdict == "UNSUPPORTED":
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
        grouding: GroundingResult | None = None,
    ) -> RAGAnswer:
        # _abstain là lối tắt cho mọi trường hợp từ chối
        # gọi cố định _finish với 2 giá trị cố định answer = no_answer_message(q): từ chối chuẩn, chọn theo ngôn ngữ câu hỏi
        # sources = []: từ chối không bao giờ kèm nguồn
        return self._finish(q, application_id, no_answer_message(q), [], reason, retrieval,context, t0, llm_ms, response, invalid or [], grouding)

    # Gói kết quả (từ chối lớp 1, từ chối lớp 2, trả lời bình thường) -> về một chỗ
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
                    grounding.verdict if grounding else "off", ANSWER_PROMPT_VERSION,
                    retrieval.retrieval_ms, retrieval.rerank_ms, llm_ms, result.total_ms)
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
        require_application=lambda application_id: app_repo.require(db, application_id)
    )