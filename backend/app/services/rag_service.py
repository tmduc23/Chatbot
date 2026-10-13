"""
MỤC ĐÍCH: RAGService - "nhạc trưởng" điều phối toàn bộ luồng hỏi đáp (spec mục 31: một orchestrator rõ ràng)

    question
    -> kiểm tra câu hỏi (rỗng, quá dài)
    -> Retrieval Pipeline: hybrid search -> rerank -> lọc ngưỡng    (Phase 3-5, ngưỡng ở phase 6)
    -> 0 chunk qua ngưỡng? Từ chối, không gọi LLM                   (lớp bảo vệ 1: điểm reranker)
    -> context builder: loại trùng, giới hạn độ dài, [Source n]
    -> LLM trả lời CHỈ dựa trên context
    -> LLM trả lời INSUFFICIENT_CONTEXT? -> từ chối                 (lớp bảo vệ 2)
    -> answer + sources

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
from dataclasses import dataclass

from sqlalchemy.orm import Session 

from app.core.config import Settings, get_settings
from app.rag.context_builder import BuiltContext, Source, build_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import LLMProvider, LLMResponse, get_llm
from app.rag.prompts.answer import (
    ANSWER_PROMPT_VERSION,
    build_answer_prompt,
    build_answer_system_instruction,
    is_insufficient,
    no_answer_message,
)

from app.rag.reranking import Reranker
from app.rag.retrieval_pipeline import RetrievalPipeline, RetrievalResult, build_retrieval_pipeline

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000

# Lý do từ chối — để log và đếm trong evaluation
NO_RELEVANT_CHUNKS = "no_relevant_chunks"   # không chunk nào qua ngưỡng -> không gọi LLM
LLM_INSUFFICIENT = "llm_insufficient"       # LLM đọc context rồi báo không đủ thông tin

_CITATION = re.compile(r"\[Source (\d+)\]")


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    application_id: str
    answer: str
    sources: list[Source]                   # nguồn được trích trong câu trả lời
    abstained: bool                         # True = hệ thống từ chối trả lời
    abstain_reason: str | None              # NO_RELEVANT_CHUNKS | LLM_INSUFFICIENT | None
    retrieval: RetrievalResult              # giữ lại để debug: ứng viên, điểm rank, ngưỡng
    context: BuiltContext | None            # None nếu từ chối trước khi gọi LLM
    llm_ms: float
    total_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def validate_question(questrion: str) -> str:
    q = questrion.strip()
    if not q:
        raise ValueError("Question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question is too long (max {MAX_QUESTION_CHARS} characters)")
    return q

# Trả về các chuỗisó trong cited number
def cited_numbers(answer: str) -> set[int]:
    """'... [source 2] ... [Source 1][Source 2]' -> {1, 2}"""
    return {int(n) for n in _CITATION.findall(answer)}

# Retrieva Pipelline là rerank xong
class RAGService:
    def __init__(self, pipeline: RetrievalPipeline, llm: LLMProvider, context_max_chars: int = 6000):
        self.pipeline = pipeline
        self.llm = llm
        self.context_max_chars = context_max_chars

    def ask(self, question: str, application_id: str):
        t0 = time.perf_counter()
        q = validate_question(question)

        retrieval = self.pipeline.run(q, application_id)
        # relevant gồm chunk được reranker chấm điểm và điểm >= ngưỡng; chunk không được reranker chấm: luôn được giữ
        # Lớp từ chối 1, không chunk nào qua ngưỡng thì trả về ngay, context=0.0, llm_ms=0.0 -> không tốn request LM, không cho LLM cơ hội bịa
        if not retrieval.relevant:
            return self._finish(q, application_id, retrieval, None, t0, 0.0,
                                abstain_reason=NO_RELEVANT_CHUNKS)

        context = build_context(retrieval.relevant, self.context_max_chars)     # context cho LLM

        # Cùng mô tả application mà reranker đã đùng (1 query DB nhỏ, chấp nhận gọi lại)
        domain = self.pipeline.domain_resolver(application_id) if self.pipeline.domain_resolver else None

        t1 = time.perf_counter()
        # generate: prompt, system_instruction
        response = self.llm.generate(build_answer_prompt(q, context.text), build_answer_system_instruction(domain))

        llm_ms = (time.perf_counter() - t1) * 1000

        if is_insufficient(response.text):
            return self._finish(q, application_id, retrieval, context, t0, llm_ms,
                                abstain_reason=LLM_INSUFFICIENT, response=response)
        return self._finish(q, application_id, retrieval, context, t0, llm_ms, response=response)

    # Gói kết quả (từ chối lớp 1, từ chối lớp 2, trả lời bình thường) -> về một chỗ
    def _finish(
        self,
        q: str,
        application_id: str,
        retrieval: RetrievalResult,
        context: BuiltContext | None,
        t0: float,
        llm_ms: float,
        abstain_reason: str | None = None,
        response: LLMResponse | None = None,
    ) -> RAGAnswer:
        if abstain_reason:
            answer, sources = no_answer_message(q), []
        else:
            answer = response.text
            cited = cited_numbers(answer)
            # LLM quên trích nguồn -> tạm trả mọi nguồn trong context (Phase 7 xử lý chặt hơn)
            sources = [s for s in context.sources if s.number in cited] or list(context.sources)

        result = RAGAnswer(
            question=q, application_id=application_id, answer=answer, sources=sources,
            abstained=abstain_reason is not None, abstain_reason=abstain_reason,
            retrieval=retrieval, context=context, llm_ms=llm_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            input_tokens=getattr(response, "input_tokens", None),
            output_tokens=getattr(response, "output_tokens", None), 
        )
        logger.info("rag_answer app=%s abstained=%s reason=%s sourced=%d prompt=%s "
                    "retrieval_ms=%.0f rerank_ms=%.0f llm_ms=%.0f total_ms=%.0f",
                     application_id, result.abstained, abstain_reason, len(sources), ANSWER_PROMPT_VERSION,
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
        context_max_chars=s.context_max_chars
    )