"""
MỤC ĐÍCH: Phạm vi tìm khi đánh giá (chế độ paper).

    all     tìm trong CẢ application (mặc định)
    paper   mỗi câu chỉ tìm trong paper của nó (expected_document), giống người dùng mở 1 paper rồi hỏi.
            Đây là cách QASPER / PeerQA gốc chấm, nên số liệu so được với bài gốc.

Ở chế độ paper:
  - Bỏ tiền tố 'In the paper "Tên": ' khỏi câu hỏi: người dùng đã chọn paper thì không gõ lại tên.
  - Paper của câu hỏi: expected_documents; nếu không có thì đoán từ tên paper trong tiền tố.
    Câu bẫy THẬT của QASPER/PeerQA (paper có, nhưng không nói điều được hỏi) không có expected_documents
    nên đi đường này: "In the paper "Tên": ..." -> file "NN-<slug của Tên>.txt" (cách import_* đặt tên file).
  - Không tìm được paper trong app -> BỎ QUA, liệt kê trong "skipped". Đó là câu holdout (hỏi về paper
    KHÔNG có trong app: ở chế độ paper người dùng không thể chọn nó) hoặc dataset lệch với app.
"""

import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.evaluation.dataset import EvalCase
from app.evaluation.paper_import import slugify
from app.rag.retrieval.keyword_retriever import _PAPER_PREFIX, strip_paper_prefix
from app.repositories import document_repository as doc_repo


SCOPES = ("all", "paper")

@dataclass(frozen=True)
class ScopedQuestion:
    question: str                   # câu gửi cho retriever / RAGService
    document_ids: list[uuid.UUID]   # None = cả application

    def kwargs(self) -> dict:
        """Chỉ truyền document_ids khi có lọc: code cũ (và fake trong test) không phải biết tham số này."""
        return {"document_ids": self.document_ids} if self.document_ids else {}



@dataclass(frozen=True)
class ScopePlan:
    mode: str
    questions: dict[str, ScopedQuestion]                # case.id -> câu hỏi sẽ chạy
    skipped: list[str]                                  # case.id câu hỏi bị bỏ qua (chỉ ở chế độ paper)
    by_title: list[str] = field(default_factory=list)   # case.id tìm ra paper nhờ tên trong câu hỏi

    # giữ những câu hỏi sẽ chạy
    def keep(self, cases: list[EvalCase]) -> list[EvalCase]:
        return [c for c in cases if c.id in self.questions]

    def summary(self) -> dict:
        return {"mode": self.mode, "skipped": self.skipped, "by_title": self.by_title}


def plan_scope(db: Session, application_id: str, cases: list[EvalCase], mode: str = "all") -> ScopePlan:
    if mode not in SCOPES:
        raise ValueError(f"scope phải là một trong {SCOPES}, nhận '{mode}'")
    if mode == "all":
        return ScopePlan(mode, {c.id: ScopedQuestion(c.question, None) for c in cases}, [])

    # tên file -> id, CHỈ trong application này (Phase 8: không bao giờ nhìn sang app khác)
    by_name: dict[str, list[uuid.UUID]] = {}                   
    for doc in doc_repo.list_documents(db, application_id):
        by_name.setdefault(doc.filename, []).append(doc.id)

    questions: dict[str, ScopedQuestion] = {}
    skipped: list[str] = []
    by_title: list[str] = []
    for case in cases:
        ids = [i for name in case.expected_documents for i in by_name.get(name, [])]        # lấy id
        if not ids and not case.expected_documents:         
            ids = docs_by_title(case.question, by_name)         # lấy id thông qua tên paper trong câu hỏi
            if ids:                                             # nếu có thì ghi vào câu tìm ra nhờ tên
                by_title.append(case.id)
        if not ids:                                             # không có id -> bỏ qua
            skipped.append(case.id)
            continue
        questions[case.id] = ScopedQuestion(strip_paper_prefix(case.question), ids)     
    return ScopePlan(mode, questions, skipped, by_title)            # trả về kết quả


"""
questions["q1"] = ScopedQuestion(question="How many layers?", document_ids=[UUID_A])
"""

def docs_by_title(question: str, by_name: dict[str, list[uuid.UUID]]) -> list[uuid.UUID]:
    """'In the paper "Deep Nets": ...' -> id của file "07-deep-nets.txt" (hoặc "deep-nets.txt"). Không có -> []."""
    m = _PAPER_PREFIX.match(question)
    if m is None:
        return []
    pattern = re.compile(rf"(\d+-)?{re.escape(slugify(m['title']))}\.\w+")
    return [i for name, ids in by_name.items() if pattern.fullmatch(name) for i in ids]