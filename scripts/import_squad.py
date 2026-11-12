"""
MỤC ĐÍCH: Biến 1 bộ dữ liệu chuẩn định dạng SQuAD 2.0 thành tài liệu để nạp + eval/dataset_*.json

SQuAD 2.0 (Stanford, CC BY-SA 4.0) là benchmark hỏi đáp nổi tiếng: các đoạn văn Wikipedia,
câu hỏi do người viết, và ~1/3 câu KHÔNG có đáp án trong đoạn văn (được viết để "gài bẫy").
-> Đúng thứ ta cần đo: tìm đúng chunk (retrieval) và biết từ chối (hallucination).

    python scripts/import_squad.py --input eval/squad/dev-v2.0.json \
        --docs-out eval/squad/docs --out eval/dataset_squad.json --articles 6 --questions 100

Script làm 3 việc:
  1. Chọn ngẫu nhiên (có seed -> chạy lại ra y hệt) N bài Wikipedia, mỗi bài -> 1 file .txt.
  2. Câu có đáp án: expected_texts = câu văn chứa đáp án (đủ dài để chỉ khớp 1 chỗ),
    answer_contains = đáp án ngắn nhất do người chấm SQuAD ghi.
  3. Câu không có đáp án: answerable=false.


Nhận 3 định dạng:
  - file gốc SQuAD: {"data": [{"title", "paragraphs": [{"context", "qas": [...]}]}]}
  - dạng phẳng (JSON list hoặc JSONL, như bản trên HuggingFace):
    {"id", "title", "context", "question", "answers": {"text": [...], "answer_start": [...]}}
  - file .parquet của HuggingFace (cùng các cột như dạng phẳng; cần pip install pyarrow)

Bản tiếng Việt UIT-ViQuAD 2.0 (VLSP 2021) có cùng định dạng, kể cả câu không có đáp án:
    python scripts/import_squad.py --input eval/viquad2/validation.parquet --language vi --source viquad2 \
        --docs-out eval/viquad2/docs --out eval/dataset_viquad2.json

1. _answers(raw)        : lấy danh sách (text, answer_start) của 1 câu, chịu được 2 file
2. parse_squad(raw)     : đọc dạng lồng gốc ({"data": [...]}) hoặc dạng phẳng của HuggingFace(1 dòng = 1 câu) thành list[Article]
3. read_input(path)     : đọc file .json hoặc .jsonl rồi chuyển cho parse_squad.
4. evidence for(context, start, answer): Chọn câu chứa đáp án làm expected_text. 
                                        Câu dài hơn 220 ký tự thì cắt quanh đáp án, không cắt giữa từ 
5. slugify(title)       : tên bài thành tên file an toàn   "Super_Bowl_50" →"super-bowl-50"
6. build(...)           : chọn ngẫu nhiên n_articles bài (có seed), ghi nội dung, tạo câu có đáp án (sa001...) và câu bẫy (su001) theo answerable_ratio
7. main                 : đọc tham số, chặn ghi đè khi thiếu --force, xoá .txt cũ, ghi file, in thống kê và ước tính số lần gọi API.

# Dạng data
data[]                    mỗi phần tử = 1 bài Wikipedia
├─ title                  tên bài
└─ paragraphs[]           các đoạn văn của bài
   ├─ context             nội dung đoạn văn
   └─ qas[]               các câu hỏi về ĐÚNG đoạn đó
      ├─ id, question, is_impossible
      ├─ answers[]        {text, answer_start}; rỗng nếu là câu bẫy
      └─ plausible_answers[]   chỉ có ở câu bẫy
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.dataset import normalize

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MAX_EVIDENCE_CHARS = 220   # câu văn dài hơn thì cắt quanh đáp án, để không vắt qua 2 chunk


@dataclass
class Question:
    id: str
    question: str
    answers: list[tuple[str, int]]          # (text, answer_start); rỗng = không có đáp án


@dataclass
class Paragraph:
    context: str
    questions: list[Question] = field(default_factory=list)


@dataclass
class Article:
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)


# ---------- đọc file ----------

def _answers(raw: dict) -> list[tuple[str, int]]:
    if raw.get("is_impossible"):
        return []
    answers = raw.get("answers") or []
    if isinstance(answers, dict):                       # dạng HuggingFace: {"text": [...], "answer_start": [...]}
        return list(zip(answers.get("text", []), answers.get("answer_start", [])))
    return [(a["text"], a["answer_start"]) for a in answers]


def parse_squad(raw: object) -> list[Article]:
    if isinstance(raw, dict) and "data" in raw:
        return [
            Article(
                title=a["title"],
                paragraphs=[
                    Paragraph(p["context"], [Question(q["id"], q["question"], _answers(q)) for q in p["qas"]])
                    for p in a["paragraphs"]
                ],
            )
            for a in raw["data"]
        ]
    if isinstance(raw, list):                           # dạng phẳng: 1 dòng = 1 câu hỏi
        articles: dict[str, Article] = {}
        paragraphs: dict[tuple[str, str], Paragraph] = {}
        for row in raw:
            article = articles.setdefault(row["title"], Article(row["title"]))
            key = (row["title"], row["context"])
            if key not in paragraphs:
                paragraphs[key] = Paragraph(row["context"])
                article.paragraphs.append(paragraphs[key])
            paragraphs[key].questions.append(Question(row["id"], row["question"], _answers(row)))
        return list(articles.values())
    raise ValueError("Không nhận ra định dạng: cần {'data': [...]} hoặc danh sách câu hỏi")


def read_input(path: Path) -> list[Article]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq                 # chỉ cần khi đọc parquet -> không thêm vào requirements
        except ImportError:
            sys.exit("Đọc .parquet cần pyarrow: docker compose exec backend pip install pyarrow")
        return parse_squad(pq.read_table(path).to_pylist())
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return parse_squad([json.loads(line) for line in text.splitlines() if line.strip()])
    return parse_squad(json.loads(text))


# ---------- chọn đoạn chữ làm bằng chứng ----------

def locate_answer(context: str, text: str, start: int) -> int | None:
    """Vị trí thật của đáp án; answer_start lệch thì tìm lại (chỉ khi đáp án xuất hiện đúng 1 lần)."""
    if context[start:start + len(text)] == text:
        return start
    if context.count(text) == 1:
        return context.index(text)
    return None

def evidence_for(context: str, start: int, answer: str) -> str:
    """Câu văn chứa đáp án; câu quá dài thì cắt quanh đáp án (theo ranh giới từ)."""
    sentence_start = 0
    for m in _SENTENCE_END.finditer(context):
        if m.end() > start:
            break
        sentence_start = m.end()
    m = _SENTENCE_END.search(context, start + len(answer))
    sentence_end = m.start() if m else len(context)
    if sentence_end - sentence_start <= MAX_EVIDENCE_CHARS:
        return context[sentence_start:sentence_end].strip()

    pad = max(0, (MAX_EVIDENCE_CHARS - len(answer)) // 2)
    lo = max(sentence_start, start - pad)
    hi = min(sentence_end, start + len(answer) + pad)
    while lo > sentence_start and not context[lo - 1].isspace():   # không cắt giữa từ
        lo += 1
    while hi < sentence_end and not context[hi].isspace():
        hi -= 1
    return context[lo:hi].strip()


def slugify(title: str) -> str:
    title = title.replace("đ", "d").replace("Đ", "D")    # "đ" không tách được dấu như "ệ" -> không có bước này sẽ bị xoá mất
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-") or "article"


# ---------- dựng dataset ----------

def build(
    articles: list[Article],
    n_articles: int,
    n_questions: int,
    unanswerable_ratio: float,
    max_paragraphs: int,
    seed: int,
    language: str = "en",
    source: str = "squad",
) -> tuple[dict[str, str], list[dict], dict]:
    """Trả về (tên file -> nội dung, danh sách câu hỏi, thống kê)."""
    rng = random.Random(seed)
    chosen = rng.sample(articles, min(n_articles, len(articles)))

    documents: dict[str, str] = {}
    answerable: list[dict] = []
    unanswerable: list[dict] = []
    skipped = 0
    for i, article in enumerate(chosen, start=1):
        filename = f"{i:02d}-{slugify(article.title)}.txt"
        paragraphs = article.paragraphs[:max_paragraphs]
        body = article.title.replace("_", " ") + "\n\n" + "\n\n".join(p.context.strip() for p in paragraphs)
        documents[filename] = body
        norm_body = normalize(body)
        topic = slugify(article.title)

        for p in paragraphs:
            for q in p.questions:
                tags = [language, source, topic]
                if not q.answers:
                    unanswerable.append({"source_id": q.id, "question": q.question.strip(), "answerable": False,
                                         "tags": tags + ["unanswerable"]})
                    continue
                text, start = q.answers[0]
                start = locate_answer(p.context, text, start)
                evidence = evidence_for(p.context, start, text) if start is not None else ""
                # bằng chứng phải chỉ khớp ĐÚNG 1 chỗ trong bài, nếu không "chunk đúng" sẽ bị đếm sai
                if not evidence or text not in evidence or norm_body.count(normalize(evidence)) != 1:
                    skipped += 1
                    continue
                shortest = min((a for a, _ in q.answers if a.strip() and a.strip() in evidence), key=len)
                answerable.append({
                    "source_id": q.id,
                    "question": q.question.strip(),
                    "answerable": True,
                    "expected_texts": [evidence],
                    "expected_documents": [filename],
                    "answer_contains": [shortest.strip()],
                    "reference_answers": list(dict.fromkeys(a.strip() for a, _ in q.answers if a.strip())),
                    "tags": tags,
                })

    n_unans = min(len(unanswerable), round(n_questions * unanswerable_ratio))
    n_ans = min(len(answerable), n_questions - n_unans)
    picked_ans = rng.sample(answerable, n_ans)
    picked_unans = rng.sample(unanswerable, n_unans)

    cases = []
    for prefix, group in (("sa", picked_ans), ("su", picked_unans)):
        for j, case in enumerate(group, start=1):
            cases.append({"id": f"{prefix}{j:03d}", **case})
    stats = {
        "articles": len(documents),
        "paragraphs": sum(len(a.paragraphs[:max_paragraphs]) for a in chosen),
        "words": sum(len(b.split()) for b in documents.values()),
        "answerable_pool": len(answerable),
        "unanswerable_pool": len(unanswerable),
        "skipped_ambiguous": skipped,
        "answerable": n_ans,
        "unanswerable": n_unans,
    }
    return documents, cases, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a SQuAD 2.0-format file into documents + an eval dataset")
    parser.add_argument("--input", type=Path, required=True, help="dev-v2.0.json, .json/.jsonl dạng phẳng, hoặc .parquet")
    parser.add_argument("--docs-out", type=Path, default=Path("eval/squad/docs"))
    parser.add_argument("--out", type=Path, default=Path("eval/dataset_squad.json"))
    parser.add_argument("--articles", type=int, default=6, help="Số bài Wikipedia (mỗi bài 1 file)")
    parser.add_argument("--paragraphs", type=int, default=30, help="Số đoạn tối đa mỗi bài")
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--unanswerable-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="en", help="Tag ngôn ngữ cho câu hỏi (en, vi...)")
    parser.add_argument("--source", default="squad", help="Tag tên bộ dữ liệu (squad, viquad2...)")
    parser.add_argument("--force", action="store_true", help="Ghi đè nếu đã có")
    args = parser.parse_args()

    if not 0 <= args.unanswerable_ratio <= 1:
        sys.exit("--unanswerable-ratio phải trong khoảng 0..1")
    if (args.out.exists() or any(args.docs_out.glob("*.txt"))) and not args.force:
        sys.exit(f"{args.out} hoặc {args.docs_out} đã có dữ liệu. Thêm --force để ghi đè.")

    documents, cases, stats = build(
        read_input(args.input), args.articles, args.questions,
        args.unanswerable_ratio, args.paragraphs, args.seed, args.language, args.source,
    )
    args.docs_out.mkdir(parents=True, exist_ok=True)
    for old in args.docs_out.glob("*.txt"):
        old.unlink()
    for name, body in documents.items():
        (args.docs_out / name).write_text(body + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Tài liệu: {stats['articles']} file, {stats['paragraphs']} đoạn, ~{stats['words']} từ -> {args.docs_out}")
    for name in documents:
        print(f"  {name}")
    print(f"Câu hỏi : {stats['answerable']} có đáp án + {stats['unanswerable']} không có đáp án -> {args.out}")
    print(f"          (nguồn: {stats['answerable_pool']} / {stats['unanswerable_pool']} câu; "
          f"bỏ {stats['skipped_ambiguous']} câu vì không lấy được bằng chứng khớp đúng 1 chỗ)")
    print(f"Ước tính API cho 1 lần evaluate_retrieval: {len(cases)} embed + {len(cases)} rerank")


if __name__ == "__main__":
    main()