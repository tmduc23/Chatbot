"""
MỤC ĐÍCH: Biến QASPER thành tài liệu để nạp + eval/dataset_qasper.json (hỏi đáp trên paper khoa học).
Mọi quy tắc bám theo bài báo QASPER và bộ chấm chính thức qasper_evaluator.py.

QASPER (Dasigi và cộng sự, NAACL 2021; Allen AI; CC BY 4.0): câu hỏi do người CHỈ đọc tiêu đề + abstract
đặt ra, người khác đọc cả bài để trả lời. Mỗi câu có 1-3 người trả lời, mỗi người ghi:
    unanswerable (bài không trả lời được) | extractive_spans | free_form_answer | yes_no,
    evidence (các ĐOẠN VĂN / bảng / hình làm bằng chứng) và highlighted_evidence (câu được bôi).

    python scripts/import_qasper.py --input eval/qasper/qasper-dev-v0.3.json        # 60 paper, 200 câu

Quy tắc:
  - Tài liệu: 1 paper = 1 file .txt, chữ giữ NGUYÊN như QASPER (kể cả mã BIBREF19, INLINEFORM0...).
  - Câu BẪY (answerable=false): MỌI người trả lời đều ghi unanswerable.
    Người bảo có, người bảo không -> bỏ câu (không có 1 đáp án đúng duy nhất để chấm từ chối).
  - Câu có đáp án:
      evidence_paragraphs = NGUYÊN các đoạn trong trường "evidence" (bảng/hình bỏ tiền tố "FLOAT SELECTED: ")
        -> chunk đúng = chunk thuộc 1 đoạn bằng chứng (chấm theo đoạn văn như Evidence F1 của bài báo).
        Không đoạn nào tìm thấy trong bài -> bỏ câu.
      reference_answers = mỗi người 1 đáp án, viết y như qasper_evaluator.py
        (span nối ", " | câu tự viết | "Yes"/"No") -> Answer F1 lấy điểm cao nhất trong các đáp án.
  - Câu bẫy thật của QASPER rất ít (~7% số câu), nên mặc định bù thêm câu "holdout": câu hỏi về một
    paper KHÔNG được nạp vào app (tiêu đề nằm trong câu hỏi) -> hệ thống phải nói không có.
    Tag "unanswerable" (bẫy thật) và "holdout" (thiếu paper) để báo cáo riêng.
  - tags: en, qasper, tên paper, loại đáp án như bộ chấm chính thức (extractive | abstractive | boolean | none).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from app.evaluation.paper_import import Paper, pick_paragraphs, sample_cases, with_title, write_outputs, slugify

_FLOAT = re.compile(r"^FLOAT SELECTED:\s*")    # bằng chứng là bảng/hình: "FLOAT SELECTED: Table 1: ..."


def read_input(path: Path) -> dict[str, dict]:
    """File gốc qasper-*-v0.3.json: {paper_id: {title, abstract, full_text, qas, figures_and_tables}}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(v, dict) and "qas" in v for v in raw.values()):
        sys.exit(f"{path}: không phải file QASPER gốc (cần dạng {{paper_id: {{title, full_text, qas}}}})")
    return raw

# section là list(tuple) chứa các phần
def to_paper(paper_id: str, raw: dict) -> Paper:
    sections = [("Abstract", [raw.get("abstract") or ""])]
    for sec in raw.get("full_text") or []:
        name = (sec.get("section_name") or "").replace(" ::: ", " > ")   # mục con: "Method ::: Encoder"
        sections.append((name, sec.get("paragraphs") or []))    
    captions = [f["caption"] for f in raw.get("figures_and_tables") or [] if f.get("caption")]
    sections.append(("Figures and tables", captions))
    return Paper(paper_id, raw["title"].strip(), sections)


def answer_text(ans: dict) -> str:
    # Đáp án của 1 người, viết y như get_answers_and_evidence() trong qasper.evaluator.py, chuyển đáp án thành 1 chuỗi
    if ans["extractive_spans"]:
        return ", ".join(ans["extractive_spans"])
    if ans["free_form_answer"]:
        return ans["free_form_answer"]
    if ans["yes_no"] is not None:
        return "Yes" if ans["yes_no"] else "No"
    return ""

def answer_type(ans: dict) -> str:
    """Loại đáp án, cùng tên với "Answer F1 by type" của bộ chấm chính thức."""
    if ans["unanswerable"]:
        return "none"
    if ans["extractive_spans"]:
        return "extractive"
    if ans["free_form_answer"]:
        return "abstractive"
    return "boolean" if ans["yes_no"] is not None else "none"

def holdout_questions(raw: dict[str, dict], papers: list[str]) -> list[dict]:
    """Câu hỏi (có đáp án) của các paper KHÔNG nạp -> trong app này là câu không có đáp án."""
    pool = []
    for paper_id in papers:
        title = raw[paper_id]["title"].strip()
        for qa in raw[paper_id]["qas"]:
            if qa["answers"] and not any(a["answer"]["unanswerable"] for a in qa["answers"]):
                pool.append({"source_id": qa["question_id"], "question": with_title(qa["question"], title),
                             "answerable": False, "tags": ["en", "qasper", slugify(title, 40), "none", "holdout"]})
    return pool


def build(raw: dict[str, dict], n_papers: int, n_questions: int, unanswerable_ratio: float,
          seed: int, title_in_question: bool = True, holdout: bool = True) -> tuple[dict[str, str]]:
    rng = random.Random(seed)
    chosen = rng.sample(sorted(raw), min(n_papers, len(raw)))

    documents: dict[str, str] = {}
    answerable: list[dict] = []
    unanswerable: list[dict] = []
    skipped = {"disagree": 0, "no_evidence": 0}
    
    for i, paper_id in enumerate(chosen, start=1):
        paper = to_paper(paper_id, raw[paper_id])
        filename = f"{i:02d}-{slugify(paper.title)}.txt"
        body = paper.render()
        documents[filename] = body

        for qa in raw[paper_id]["qas"]:
            answers = [a["answer"] for a in qa["answers"]]
            if not answers:
                continue
            question = with_title(qa["question"], paper.title) if title_in_question else qa["question"].strip()
            tags = ["en", "qasper", slugify(paper.title, 40)]
            flags = {a["unanswerable"] for a in answers}
            if flags == {True}:             # cả nhóm đồng ý không trả lời được
                unanswerable.append({"source_id": qa["question_id"], "question": question, "answerable": False,
                                    "tags": tags + ["none", "unanswerable"]})
                continue
            if True in flags:               # Còn True thì flags {True, False} là bất đồng
                skipped["disagree"] += 1
                continue

            paragraphs = pick_paragraphs([_FLOAT.sub("", e) for a in answers for e in a["evidence"]], body)     # bỏ tiền tố FLOAT SELECTED
            if not paragraphs:
                skipped["no_evidence"] += 1
                continue
            answerable.append({
                "source_id": qa["question_id"],
                "question": question,
                "answerable": True,
                "evidence_paragraphs": paragraphs,
                "expected_documents": [filename],
                # map(answer_text, answers) là danh sách đáp án của mọi người cho câu, answer_text(ans) chuyển đáp án thành 1 chuỗi
                # chuyển từ dict sang list thì chỉ lấy các key, không lấy value
                "reference_answers": list(dict.fromkeys(t for t in map(answer_text, answers) if t.strip())),        # lấy các đáp án của mọi người
                "tags": tags + [answer_type(answers[0])],
            })


    # holdout chỉ có nghĩa khi câu hỏi có tiêu đề: câu trần "What baselines do they use?" có thể
    # được 1 paper khác trong app trả lời -> không còn là câu không có đáp án
    others = [p for p in sorted(raw) if p not in chosen]
    extra = holdout_questions(raw, others) if holdout and title_in_question else []
    cases = sample_cases(answerable, unanswerable, n_questions, unanswerable_ratio, rng, prefix="q", holdout=extra)
    stats = {
        "paper": len(documents),
        "words": sum(len(b.split()) for b in documents.values()),
        "answerable_pool": len(answerable),
        "unanswerable_pool": len(unanswerable),
        "answerable": sum(c["answerable"] for c in cases),
        "unanswerable": sum(not c["answerable"] for c in cases),
        "holdout": sum("holdout" in c["tags"] for c in cases),
        **{f"skipped_{k}": v for k, v in skipped.items()},
    }

    return documents, cases, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert QASPER into documents + an eval dataset")
    parser.add_argument("--input", type=Path, required=True, help="qasper-dev-v0.3.json (hoặc -test-)")
    parser.add_argument("--docs-out", type=Path, default=Path("eval/qasper/docs"))
    parser.add_argument("--out", type=Path, default=Path("eval/dataset_qasper.json"))
    parser.add_argument("--papers", type=int, default=60, help="Số paper (mỗi paper 1 file)")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--unanswerable-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-questions", action="store_true",
                        help="Giữ nguyên câu hỏi, không gắn tiêu đề paper (chỉ hợp khi app có 1 paper)")
    parser.add_argument("--no-holdout", action="store_true",
                        help="Không bù câu bẫy bằng câu hỏi về paper không được nạp")
    parser.add_argument("--force", action="store_true", help="Ghi đè nếu đã có")
    args = parser.parse_args()

    if not 0 <= args.unanswerable_ratio <= 1:
        sys.exit("--unanswerable-ratio phải trong khoảng 0..1")

    documents, cases, stats = build(read_input(args.input), args.papers, args.questions, args.unanswerable_ratio, 
                                    args.seed, not args.raw_questions, not args.no_holdout)
    write_outputs(documents, cases, args.docs_out, args.out, args.force)

    print(f"Tài lệu: {stats['paper']} paper, ~{stats['words']} từ -> {args.docs_out}")
    print(f"Câu hỏi : {stats['answerable']} có đáp án + {stats['unanswerable']} không có đáp án "
          f"({stats['holdout']} trong đó là holdout) -> {args.out}")

    print(f"          (nguồn: {stats['answerable_pool']} / {stats['unanswerable_pool']} câu; bỏ "
          f"{stats['skipped_disagree']} câu người chấm bất đồng, {stats['skipped_no_evidence']} câu không tìm thấy đoạn bằng chứng trong bài)")
    print(f"Ước tính API cho 1 lần evaluate_retrieval: {len(cases)} embed + {len(cases)} rerank")



if __name__ == "__main__":
    main()

