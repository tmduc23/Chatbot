"""
MỤC ĐÍCH: Đọc và kiểm tra bộ dữ liệu dánh giá eval/dataset.json

Mỗi câu hỏi:
    {
      "id": "q01",
      "question": "Bạn ấy học trường đại học nào?",
      "answerable": true,                          # false = CV không có thông tin -> hệ thống phải từ chối
      "expected_texts": ["Bách khoa Hà Nội (HUST)"],   # chunk CHỨA 1 trong các đoạn này = chunk đúng
      "expected_documents": ["CV.pdf"],            # tuỳ chọn: chunk đúng phải thuộc 1 trong các file này
      "answer_contains": ["Bách khoa"],            # tuỳ chọn: câu trả lời phải chứa MỌI đoạn này
      "tags": ["vi", "education"]                  # tuỳ chọn: để xem điểm theo nhóm
    }


"Chunk đúng" được xác định bằng ĐOẠN CHỮ chứ không bằng chunk_id: chunk_id đổi mỗi lần nạp lại tài liệu
(đổi chunk_size, đổi embedding...), còn đoạn chữ trong CV thì không đổi -> dataset dùng được lâu dài.
"""

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

PARAGRAPH_MATCH = 0.5       # TỈ LỆ ĐO TRÙNG TỐI THIỂU, TÍNH TRÊN PHẦN NGẮN HƠN (CHUNK HOẶC ĐOẠN BẰNG CHỨNG)
NGRAM = 5                   # so theo cụm 5 từ liên tiếp



class DatasetError(ValueError):
    """dataset.json sai định dạng - báo rõ câu nào, trường nào."""



@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    answerable: bool
    expected_texts: list[str] = field(default_factory=list)
    expected_documents: list[str] = field(default_factory=list)
    evidence_paragraphs: list[str] = field(default_factory=list)   # có -> chấm theo đoạn văn (kiểu bài báo)
    answer_contains: list[str] = field(default_factory=list)
    reference_answers: list[str] = field(default_factory=list)   # nhiều đáp án mẫu: lấy điểm CAO NHẤT
    tags: list[str] = field(default_factory=list)



def normalize(text: str) -> str:
    """
    So khớp chữ không phụ thuộc cách viết.

    NFC: chữ "ệ" có thể được lưu thành 1 ký tự hoặc "e" + 2 dấu rời (PDF hay làm vậy);
         2 cách trông giống hệt nhau nhưng `in` của Python coi là khác -> phải đưa về cùng 1 dạng.
    casefold: như lower() nhưng đầy đủ hơn cho Unicode.
    gộp khoảng trắng: PDF hay xuống dòng giữa câu ("Bách Khoa \\nHà Nội").
    """
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=16384)    
def words(text: str) -> tuple[str, ...]:
    """Các từ sau khi normalize, bỏ dấu câu.  lru_cache: cùng 1 chunk được so với nhiều câu hỏi."""
    return tuple(re.findall(r"\w+", normalize(text)))

@lru_cache(maxsize=16384)
def shingles(text: str, n: int = NGRAM) -> frozenset[tuple[str, ...]]:
    """Tập các cụm n từ liên tiếp.

    >>> sorted(shingles("A b, c d e f", 5))
    [('a', 'b', 'c', 'd', 'e'), ('b', 'c', 'd', 'e', 'f')]
    """
    w = words(text)
    return frozenset(w[i:i + n] for i in range(len(w) - n + 1))


def paragraph_overlap(chunk_text: str, paragraph: str) -> float:
    """
    Số cụm 5 từ chung / số cụm của phần NGẮN hơn. 1.0 = cái này nằm trọn trong cái kia.

    Đoạn quá ngắn (≤ 5 từ, vd. "Table 1: Data Statistics.") không có đủ cụm -> chỉ cần nằm trong chunk.

    >>> para = "We train on the Penn Treebank corpus with forty thousand sentences in total."
    >>> paragraph_overlap("Title > Data\\n" + para + " Next paragraph starts here.", para)
    1.0
    >>> paragraph_overlap("A different paragraph about the test set and the metrics we use.", para)
    0.0
    >>> paragraph_overlap("Results.\\n\\nTable 1: Data statistics.", "Table 1: Data Statistics.")
    1.0
    """
    pw = words(paragraph)           # tách thành các từ riêng lẻ
    if not pw:
        return 0.0

    if len(pw) <= NGRAM:
        return 1.0 if f" {' '.join(pw)} " in f" {' '.join(words(chunk_text))} " else 0.0
    c, p = shingles(chunk_text), shingles(paragraph)
    if not c: 
        return 0.0

    return len(c & p) / min(len(c), len(p))


def chunk_matches(case: EvalCase, text: str, filename: str) -> bool:
    """
    Chunk này có phải chunk đúng của câu hỏi không.
    """
    if case.expected_documents and filename not in case.expected_documents:
        return False
    if case.evidence_paragraphs:
        return any(paragraph_overlap(text, p) >= PARAGRAPH_MATCH for p in case.evidence_paragraphs)
    body = normalize(text)
    return any(normalize(t) in body for t in case.expected_texts)




# Câu trả lời của LLM có chứa đủ các "từ khóa bắt buộc" không -> facts_ok_rate
# answer_has_facts -> chấm câu trả lời (generation): nói đủ mọi chi tiết bắt buộc -> dùng all
# VD: answer_contains = ["875", "TOEIC"]
# - **Mục đích:** kiểm tra **câu trả lời của LLM** (không phải chunk) có chứa đủ các sự kiện bắt buộc không.
#  Kết quả là chỉ số `facts_ok_rate`.

def answer_has_facts(case: EvalCase, answer: str) -> bool | None:
    """Câu trả lời có chứa MỌI doạn trong answer_contains không. None = câu này không khai báo"""
    if not case.answer_contains:
        return None
    body = normalize(answer)
    return all(normalize(t) in body for t in case.answer_contains)

"""Lấy raw[key] và bảo đảm nó là list các chuỗi có chữ; thiếu key -> []; sai kiểu hoặc có chuỗi rỗng -> DatasetError."""
def _str_list(raw: dict, key: str, where: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise DatasetError(f"{where}: '{key}' phải là danh sách chuỗi không rỗng")
    return value


"""Kiểm tra và chuyển 1 object JSON thành EvalCase; sai thì raise DatasetError kèm vị trí câu."""
def parse_case(raw: object, index: int) -> EvalCase:
    where = f"câu #{index + 1}"         # ban đầu chưa biết id thì ghi theo index
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: phải là object JSON")
    case_id = raw.get("id")
    question = raw.get("question")
    answerable = raw.get("answerable")
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetError(f"{where}: thiếu 'id'")
    where = f"câu '{case_id}'"          # chắc có id -> chuyển về id
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{where}: thiếu 'question'")
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: 'answerable' phải là true hoặc false")

    case = EvalCase(
        id = case_id,
        question=question.strip(),
        answerable=answerable,
        expected_texts=_str_list(raw, "expected_texts", where),
        expected_documents=_str_list(raw, "expected_documents", where),
        evidence_paragraphs=_str_list(raw, "evidence_paragraphs", where),
        answer_contains=_str_list(raw, "answer_contains", where),
        reference_answers=_str_list(raw, "reference_answers", where),
        tags = _str_list(raw, "tags", where),  
    )
    if case.answerable and not (case.expected_texts or case.evidence_paragraphs):
        raise DatasetError(f"{where}: câu có đáp án phải có ít nhất 1 'expected_texts' hoặc 'evidence_paragraphs'")
    if not case.answerable and (case.expected_texts or case.evidence_paragraphs or case.answer_contains or case.reference_answers):
        raise DatasetError(f"{where}: câu không có đáp án không được có 'expected_texts' / 'evidence_paragraphs' / 'answer_contains' / 'reference_answers'")
    return case


def load_dataset(path: Path) -> list[EvalCase]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: JSON sai ở dòng {exc.lineno}: {exc.msg}") from exc
    if not isinstance(raw, list) or not raw:
        raise DatasetError(f"{path}: phải là 1 danh sách câu hỏi, không rỗng")
    cases = [parse_case(item, i) for i, item in enumerate(raw)]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise DatasetError(f"id '{c.id}' bị trùng")
        seen.add(c.id)
    return cases


def stratified_sample(cases: list[EvalCase], n: int, seed: int = 42) -> list[EvalCase]:
    """Lấy ngẫu nhiên (có seed) n câu, GIỮ tỉ lệ có đáp án / không có đáp án của cả bộ.

    n: số câu muốn lấy ra để chấm

    Không dùng "n câu đầu": file của import_squad / import_mteb_retrieval xếp mọi câu có đáp án
    lên trước, nên 40 câu đầu sẽ không có câu bẫy nào -> hallucination_rate không đo được.
    """
    if n >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    unans = [c for c in cases if not c.answerable]      # 30 câu bẫy
    ans = [c for c in cases if c.answerable]            # 70 câu có đáp án
    n_unans = round(n * len(unans) / len(cases))        # 
    picked = set(id(c) for c in rng.sample(ans, n - n_unans) + rng.sample(unans, n_unans))      # bốc câu có đáp án / tổng số câu có đáp án + câu không đáp án / tổng số câu bẫy
    return [c for c in cases if id(c) in picked]       # giữ thứ tự gốc của file để duyệt