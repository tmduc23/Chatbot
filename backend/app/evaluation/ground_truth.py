"""
MỤC ĐÍCH: Tìm "chunk đúng" của từng câu hỏi trong DB (Phase 11).

Đọc MỌI chunk của 1 application (CV chỉ vài chục chunk -> đọc hết vẫn nhanh), rồi so đoạn chữ
expected_texts. Kết quả dùng để tính Recall@k: cần biết có TỔNG CỘNG bao nhiêu chunk đúng,
kể cả những chunk retriever không tìm ra.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.dataset import EvalCase, chunk_matches
from app.models import Chunk, Document



@dataclass(frozen=True)
class GroundTruth:
    relevant: dict[str, set[uuid.UUID]]         # case.id -> chunk_id đúng (chỉ câu có đáp án và tìm thấy) - {id câu hỏi: {chunk_id, chunk_id, ...}}
    missing: list[str]                          # case.id có đáp án nhưng KHÔNG chunk nào khớp -> lỗi dataset, list id câu hỏi mà không chunk nào khớp
    total_chunks: int


def build_ground_truth(db: Session, application_id: str, cases: list[EvalCase]) -> GroundTruth:
    rows = db.execute(
        select(Chunk.id, Chunk.text, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.application_id == application_id)
    ).all()

    relevant: dict[str, set[uuid.UUID]] = {}
    missing: list[str] = []
    for case in cases:
        if not case.answerable:
            continue
        ids = {row.id for row in rows if chunk_matches(case, row.text, row.filename)}       # text: nội dung của chunk
        if ids:
            relevant[case.id] = ids
        else: 
            missing.append(case.id)
    return GroundTruth(relevant, missing, len(rows))
