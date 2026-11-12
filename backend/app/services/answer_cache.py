"""
MỤC ĐÍCH (Phase 12c): CACHE CÂU TRẢ LỜI — cùng câu hỏi thì không gọi lại embedding + rerank + LLM.

Vì sao đáng làm với project này: trang chat có sẵn 3 câu gợi ý ("Bạn ấy đã làm những dự án nào?"...).
Nhà tuyển dụng nào cũng bấm -> cùng 1 câu hỏi lặp lại hàng chục lần. Mỗi lần tốn 3-4 lời gọi API
(~5-30 giây với model miễn phí). Cache: lần 2 trở đi trả lời trong vài ms, 0 lời gọi API.

Phần khó nhất của cache là BIẾT KHI NÀO CÂU TRẢ LỜI CŨ KHÔNG CÒN ĐÚNG (cache invalidation).
Ở đây không xoá cache bằng tay, mà đưa mọi thứ ảnh hưởng tới câu trả lời VÀO KEY:

    key = sha256( application_id
                + câu hỏi đã chuẩn hoá           ("Bạn ấy học ở đâu?" == "bạn ấy  học ở đâu")
                + các document_id được chọn      (chế độ paper: hỏi paper A khác hỏi paper B)
                + dấu vân tay TÀI LIỆU           (danh sách id document READY: upload/xoá -> đổi)
                + dấu vân tay CẤU HÌNH           (model, ngưỡng, top_k, phiên bản prompt, mô tả app...) )

Upload thêm CV mới -> dấu vân tay tài liệu đổi -> key đổi -> không trúng cache cũ -> hỏi lại từ đầu.
Dòng cache cũ không ai dùng nữa, hết hạn (ANSWER_CACHE_TTL_SECONDS) thì bị xoá.

Câu hỏi nối tiếp ("còn dự án đó dùng gì?") dùng câu ĐÃ VIẾT LẠI (standalone) làm key, vì
retrieval + LLM chỉ nhìn câu đó.
"""
import hashlib
import json
import logging
import re
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import AnswerCacheEntry, Document, DocumentStatus
from app.repositories import application_repository as app_repo
from app.services.rag_service import RAGAnswer

logger = logging.getLogger(__name__)

# Mọi cấu hình làm thay đổi câu trả lời. Thêm cấu hình RAG mới -> thêm tên vào đây.
CONFIG_FIELDS = (
    # embedding + chunking
    "embedding_provider", "embedding_model", "cohere_embedding_model",
    "chunk_strategy", "chunk_size", "chunk_overlap", "chunk_min_tokens", "chunk_section_header",
    # retrieval
    "retrieval_mode", "retrieval_candidate_k", "rrf_k", "rrf_keyword_weight", "keyword_ranking",
    "bm25_k1", "bm25_b", "retrieval_top_k",
    # rerank + ngưỡng từ chối
    "reranker_provider", "reranker_model", "cohere_rerank_model", "reranker_top_k", "reranker_max_chars",
    "reranker_min_score",
    # sinh câu trả lời (answer_prompt: v3 | v4 | v5)
    "llm_provider", "llm_model", "llm_temperature", "llm_max_output_tokens", "context_max_chars", "answer_prompt",
    "grounding_check", "grounding_model", "grounding_provider",
)


@dataclass(frozen=True)
class AnswerSnapshot:
    """Phần của RAGAnswer cần để trả lời /chat và ghi lịch sử. Toàn kiểu đơn giản -> lưu được thành JSON."""
    answer: str
    sources: list[dict]
    abstained: bool
    abstain_reason: str | None
    grounding_verdict: str | None
    retrieval_count: int
    reranked_count: int
    relevant_count: int
    chunk_ids: list[str] = field(default_factory=list)   # chunk reranker đã chấm (cho retrieval_logs)
    scores: list[float] = field(default_factory=list)
    rag_ms: float = 0.0
    cached: bool = False                                  # True = lấy từ cache, không gọi API nào

    @classmethod
    def from_rag(cls, r: RAGAnswer) -> "AnswerSnapshot":
        reranked = r.retrieval.reranked
        return cls(
            answer=r.answer, sources=[s.to_dict() for s in r.sources], abstained=r.abstained,
            abstain_reason=r.abstain_reason, grounding_verdict=r.grounding.verdict if r.grounding else None,
            retrieval_count=len(r.retrieval.candidates), reranked_count=len(reranked),
            relevant_count=len(r.retrieval.relevant),
            chunk_ids=[str(x.chunk.chunk_id) for x in reranked], scores=[round(x.score, 4) for x in reranked],
            rag_ms=round(r.total_ms, 1),
        )

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("cached")
        return data

    @classmethod
    def from_json(cls, data: dict) -> "AnswerSnapshot":
        return replace(cls(**data), cached=True)


def normalize_question(question: str) -> str:
    """Chuẩn hoá để 2 câu chỉ khác hoa/thường, dấu cách, dấu ? cuối câu dùng chung 1 key.
    NFC: chữ "ấ" gõ bằng 2 kiểu bàn phím (1 ký tự hoặc a + dấu) trở thành cùng 1 chuỗi."""
    q = unicodedata.normalize("NFC", question).lower()
    q = re.sub(r"\s+", " ", q).strip()
    return q.rstrip(" ?!.。？")


def _sha256(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    def corpus_fingerprint(self, application_id: str) -> str:
        """Đổi mỗi khi có document READY mới hoặc bị xoá (1 câu SQL nhỏ, dùng index application_id)."""
        ids = self.db.scalars(
            select(Document.id).where(Document.application_id == application_id,
                                      Document.status == DocumentStatus.READY).order_by(Document.id)
        ).all()
        return _sha256(*(str(i) for i in ids))

    def config_fingerprint(self, application_id: str) -> str:
        config = {name: getattr(self.settings, name) for name in CONFIG_FIELDS}
        app = app_repo.get(self.db, application_id)
        # tên + mô tả application nằm trong prompt của reranker và LLM
        config.update(app_name=app.name if app else None, app_description=app.description if app else None)
        return _sha256(json.dumps(config, sort_keys=True, default=str))

    def key_for(self, application_id: str, question: str,
                document_ids: Sequence[uuid.UUID] | None = None) -> str:
        # Chế độ paper: sắp xếp id -> chọn [A, B] hay [B, A] vẫn là 1 key. Không chọn gì = "all"
        scope = ",".join(sorted(str(d) for d in document_ids)) if document_ids else "all"
        return _sha256(application_id, normalize_question(question), scope,
                       self.corpus_fingerprint(application_id), self.config_fingerprint(application_id))

    def get(self, key: str, now: datetime | None = None) -> AnswerSnapshot | None:
        now = now or datetime.now(timezone.utc)
        entry = self.db.get(AnswerCacheEntry, key)
        if entry is None or entry.expires_at <= now:
            return None
        # đếm số lần trúng (để xem cache có đáng không). Không commit: commit cùng lượt chat
        self.db.execute(update(AnswerCacheEntry).where(AnswerCacheEntry.key == key)
                        .values(hits=AnswerCacheEntry.hits + 1))
        return AnswerSnapshot.from_json(entry.payload)

    def put(self, key: str, application_id: str, question: str, snapshot: AnswerSnapshot,
            now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.settings.answer_cache_ttl_seconds)
        # 2 request cùng câu hỏi chạy song song -> cả 2 cùng ghi: ON CONFLICT ghi đè, không lỗi trùng khoá
        stmt = insert(AnswerCacheEntry).values(
            key=key, application_id=application_id, question=question[:1000],
            payload=snapshot.to_json(), expires_at=expires, hits=0,
        )
        self.db.execute(stmt.on_conflict_do_update(
            index_elements=[AnswerCacheEntry.key],
            set_={"payload": stmt.excluded.payload, "expires_at": stmt.excluded.expires_at, "hits": 0},
        ))
        # dọn dòng hết hạn của application này (cache cũ sau khi đổi tài liệu / cấu hình)
        self.db.execute(delete(AnswerCacheEntry).where(AnswerCacheEntry.application_id == application_id,
                                                       AnswerCacheEntry.expires_at <= now))

    def stats(self, application_id: str) -> tuple[int, int]:
        """(số câu trong cache còn hạn, tổng số lần trúng)."""
        row = self.db.execute(
            select(func.count(), func.coalesce(func.sum(AnswerCacheEntry.hits), 0))
            .where(AnswerCacheEntry.application_id == application_id,
                   AnswerCacheEntry.expires_at > datetime.now(timezone.utc))
        ).one()
        return int(row[0]), int(row[1])
