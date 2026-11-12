"""
MỤC ĐÍCH: Phần dùng chung khi biến bộ hỏi đáp trên PAPER KHOA HỌC (QASPER, PeerQA) thành
tài liệu .txt để nạp + eval/dataset_*.json, đúng định dạng app.evaluation.dataset.

Paper khoa học cần xử lý riêng ở 3 chỗ:
  1. Tài liệu là cả bài báo, có MỤC: ghi "## Tên mục" trước mỗi mục, để chunk và câu trả lời
     còn biết đoạn văn thuộc phần nào (Method, Results...).
  2. Câu hỏi viết cho 1 paper ("What baselines do they compare to?"). Khi nạp nhiều paper vào 1 app,
     "they" là paper nào? -> mặc định gắn tiêu đề vào câu hỏi: 'In the paper "X": ...'.
  3. Chấm theo ĐOẠN VĂN như bài báo QASPER/PeerQA: evidence_paragraphs = NGUYÊN các đoạn văn
     bằng chứng; chunk đúng = chunk "thuộc" 1 đoạn đó (xem chunk_matches trong dataset.py).
"""

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.dataset import normalize

MIN_EVIDENCE_CHARS = 25    # ngắn hơn ("Table 2.", "Yes.") thì dễ khớp nhiều chỗ, không chỉ ra được chunk đúng


@dataclass
class Paper:
    paper_id: str
    title: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)         # (tên mục, các đoạn văn)


    # `Paper("p1", "My Paper", [("Abstract", ["We study X."])]).render()` → `"My Paper\n\n## Abstract\n\nWe study X."`
    def render(self) -> str:
        parts = [self.title]
        for name, paragraphs in self.sections:
            body = [p.strip() for p in paragraphs if p and p.strip()]
            if body:
                parts.append(f"## {name}" if name else "")
                parts.extend(body)
        return "\n\n".join(p for p in parts if p)


# `"BERT: Pre-training of Deep..."` → `"bert-pre-training-of-deep"`.
def slugify(title: str, max_len: int = 60) -> str:
    title = title.replace("đ", "d").replace("Đ", "D")
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "paper"

def pick_paragraphs(candidates: list[str], body: str) -> list[str]:
    """
    CÁC ĐOẠN VĂN bằng chứng (giữ nguyên, không cắt) có trong bài, đủ dài, không trùng nhau.
    Không bắt "khớp đúng 1 chỗ": đoạn văn dài nên gần như luôn chỉ có 1 chỗ,
    và nếu bài có 2 đoạn giống hệt nhau thì chunk nào chứa đoạn đó cũng đúng thật.

    candidates: các đoạn bằng chứng thô, gộp từ mọi người chấm 
    """
    norm_body = normalize(body)     # chuẩn hóa file .txt
    picked: list[str] = []
    for raw in candidates:
        text = " ".join((raw or "").split())        #  " ".join(raw.split()): gộp mọi khoảng trắng / xuống dòng thành 1 dấu cách.
        if len(text) < MIN_EVIDENCE_CHARS or text in picked or normalize(text) not in norm_body:
            continue
        picked.append(text)
    return picked           

def with_title(question: str, title: str) -> str:
    return f'In the paper "{title}": {question.strip()}'


def sample_cases(answerable: list[dict], unanswerable: list[dict], n_questions: int,
                 unanswerable_ratio: float, rng: random.Random, prefix: str,
                 holdout: list[dict] | None = None) -> list[dict]:
    """Lấy n câu (có seed), giữ tỉ lệ câu bẫy; id dạng <prefix>a001 / <prefix>u001.

    Câu bẫy thật (bài không trả lời được) được ưu tiên; thiếu thì bù bằng câu `holdout`
    (hỏi về paper KHÔNG được nạp vào app).
    """
    n_unans = round(n_questions * unanswerable_ratio)
    if holdout and len(unanswerable) < n_unans:
        unanswerable = unanswerable + rng.sample(holdout, min(len(holdout), n_unans - len(unanswerable)))
    n_unans = min(len(unanswerable), n_unans)
    n_ans = min(len(answerable), n_questions - n_unans)
    cases = []
    for kind, group in (("a", rng.sample(answerable, n_ans)), ("u", rng.sample(unanswerable, n_unans))):
        for j, case in enumerate(group, start=1):
            cases.append({"id": f"{prefix}{kind}{j:03d}", **case})
    return cases


"""
documents: {tên file: nội dung}, vd. {"06-analysis-of-....txt": "Analysis of...\n\n## Abstract\n\n..."}.
cases: danh sách 200 câu hỏi (các dict đã tạo trong build).
docs_out: thư mục chứa .txt, vd. eval/qasper/docs.
out: đường dẫn file dataset, vd. eval/dataset_qasper.json.
force: có cờ --force hay không.
"""

def write_outputs(documents: dict[str, str], cases: list[dict], docs_out: Path, out: Path, force: bool) -> None:
    """Ghi file .txt + dataset. Đã có dữ liệu mà không --force -> dừng, không ghi đè."""
    if (out.exists() or any(docs_out.glob("*.txt"))) and not force:
        raise SystemExit(f"{out} hoặc {docs_out} đã có dữ liệu. Thêm --force để ghi đè.")
    docs_out.mkdir(parents=True, exist_ok=True)
    for old in docs_out.glob("*.txt"):
        old.unlink()                # xóa các file .txt khi sử dụng --force
    for name, body in documents.items():
        (docs_out / name).write_text(body + "\n", encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
