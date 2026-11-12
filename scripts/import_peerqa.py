"""
MỤC ĐÍCH: Biến PeerQA thành tài liệu để nạp + eval/dataset_peerqa.json (hỏi đáp trên paper khoa học).

Câu hỏi lấy từ NHẬN XÉT CỦA REVIEWER,
chính TÁC GIẢ paper đánh dấu câu bằng chứng trong PDF, viết câu trả lời, hoặc ghi "bài không trả lời được".
Văn bản paper được trích từ PDF bằng GROBID -> gần với tình huống thật hơn QASPER.

Cần 2 file trong thư mục data của PeerQA:
    papers.jsonl: 1 dòng = 1 câu / tiêu đề/ chú thích hình của 1 paper
    {paper_id, idx, pidx, sidx, type, content, last_heading}

    qa.jsonl:     1 dòng = 1 câu hỏi
                  {paper_id, question_id, question, answer_evidence_mapped: [{idx: [...]}], answer_free_form,
                  answerable, ...}
    
    python scripts/import_peerqa.py --data eval/peerqa/data \
        --docs-out eval/peerqa/docs --out eval/dataset_peerqa.json --papers 20 --questions 60

Quy tắc:
    - answerable = true + coscaau bằng chứng tìm lại được trong bài -> câu có đáp án
     evidence_paragraphs = NGUYÊN các ĐOẠN VĂN chứa câu bằng chứng (chấm theo đoạn văn như bài báo
     PeerQA và như QASPER): câu bằng chứng idx -> đoạn có cùng pidx. Bằng chứng là chú thích hình/bảng
     -> chính chú thích đó.

     - Mặc định CHỈ dùng vị trí gốc (idx trong answer_evidecne_mapped), giống bài báo PeerQA: câu không có
     idx nào dùng được (answerable_mapped != true) bị bỏ.
     --text-fallback: thử thêm tìm đoạn chứa NGUYÊN VĂN câu trong answer_evidence_sentences( cách riêng của
     project, không phải cách chuẩn: dòng tóm tắt in ra số câu được cứu theo cách này).
     - answerable=false -> câu bẫy THẬT; reviewer hỏi điều bài không nói, answerable=null -> bỏ.
     - reference_answers = câu trả lời của tác giả (tự viết, nên không có answer_contains).
     - Chỉ dùng paper CÓ trong papers.jsonl: bỏ bước tải OpenReview thì chỉ còn paper EGU / NLPeer.
     - tags: en, peerqa, nguồn (openreview | egu | nlpeer), tên paper.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from app.evaluation.dataset import normalize
from app.evaluation.paper_import import Paper, pick_paragraphs, sample_cases, slugify, with_title, write_outputs

_TEXT_TYPES = {"sentence", "list_item", "formula"}
_CAPTION_TYPES = {"figure", "table"}

_F1000_TITLE = re.compile(r"\[Version \d+;\s*Peer Review", re.IGNORECASE)

# Dấu hiệu BẮT ĐẦU phần review. Cố ý KHÔNG khớp câu trong thân bài
# "Any futher responses from the reviewers can be found at the end of the article"

_REVIEW_START = re.compile(
    r"^Reviewer$"
    r"|^Reviewer (Report|Status)\b"
    r"|Open Peer Review|Current Peer Review Status"
    r"|Is the rotionale for developing the new"
    r"|Reviewer Expertises:"
    r"|I confirm that I have read this submission"
    r"|Competing Interests: No competing interest were disclosed"
    r"|^Our response\b|In their response, the authors|we concur with the reviewer"
)

# Dấu hiệu còn sót (chỉ để CẢNH BÁO sau khi cắt, không dùng để cắt).
_REVIEW_LEFTOVER = re.compile(r"Our response|the reviewer\b|Reviewer Report|Reviewer Expertise", re.IGNORECASE)


def strip_peer_review(rows: list[dict]) -> tuple[list[dict], dict | None]:
    """
    Paper F1000: bỏ mọi dòng chữ/tiêu đề TỪ dòng review đầu tiên trở đi. Paper khác: giữ nguyên.

    Chú thích hình/bảng được giữ (thuộc bài). Trả về (rows còn lại, thông tin cắt hoặc None).
    """

    rows = sorted(rows, key=lambda r: r["idx"])
    title = next((r["content"] for r in rows if r["type"] == "title"), "")      # lấy phần tử đầu tiên là title
    if not _F1000_TITLE.search(title):          # kiểm tra có phải f1000 không 
        return rows, None                       # Không phải -> trả về rows, None

    # cut là dict đầu tiên trong rows thỏa mãn cả 3 điều kiện, nếu không thấy, cut = None
    cut = next((r for r in rows if r["type"] not in _CAPTION_TYPES and r["type"] != "title"
                and _REVIEW_START.search(r["content"].strip())), None)

    # Không tìm thấy điểm cắt -> trả về rows, {}
    if cut is None:
        return rows, {"cut_idx": None, "dropped": 0, "cut_text": "", "before": "", "leftover": []}

    # giữ các hàng trước điểm cắt or toàn bộ caption or giữ title 
    kept = [r for r in rows if r["idx"] < cut["idx"] or r["type"] in _CAPTION_TYPES or r["type"] == "title"]

    # Tạo list body để kiểm tra leftover, bỏ caption
    body = [r for r in kept if r["type"] not in _CAPTION_TYPES]

    # Tìm các dòng review còn sót lại
    leftover = [r["idx"] for r in body if _REVIEW_LEFTOVER.search(r["content"])]

    # before: 90 ký tự đầu của phần tử cuối cùng trong body sau khi cắt
    return kept, {"cut_idx": cut["idx"], "dropped": len(rows) - len(kept), "cut_text": cut["content"][:90],
                  "before": body[-1]["content"][:90] if body else "", "leftover": leftover}

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Không thấy {path}. Tải dữ liệu PeerQA.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_paper(paper_id: str, rows: list[dict]) -> tuple[Paper, dict[int, str]]:
    """
    Ghép các câu thành đoạn (cùng pidx), thành mục (theo heading).

    Trả về paper + {idx: ĐOẠN VĂN chứa dòng idx}, đúng chuỗi được ghi vào file .txt
    (chú thích hình/bảng: chính chú thích đó; tiêu đề mục: chính tên mục).    
    """
    rows = sorted(rows, key=lambda r: r["idx"])
    title = next((r["content"] for r in rows if r["type"] == "title"), paper_id)        # lấy title, nếu không có title thì lấy paper_id
    sections: list[tuple[str, list[str]]] = [("", [])]              # (tên mục, [các đoạn văn])
    captions:dict[int, tuple[list[str], list[int]]] = {}            # pidx -> (các câu của caption, idx của chúng)
    current: list[str] = []                                         # đoạn văn đang dở cùng pidx chờ nối lại
    current_idx: list[int] = []                                     # các idx của những câu trong current, để biết câu nào thuộc đoạn văn nào
    paragraph_of: dict[int, str] = {}                              # {idx: đoạn văn chứa câu idx} (để biết câu nào thuộc đoạn văn nào), 
                                                                    # kết quả phụ để chấm điểm: tra từ idx ra nguyên đoạn văn chứa câu đó
    last_pidx = None


    def flush() -> None:
        if current:
            paragraph = " ".join(current)
            sections[-1][1].append(paragraph)
            for i in current_idx:
                paragraph_of[i] = paragraph
            current.clear()
            current_idx.clear()

    for r in rows:
        if r["type"] == "heading":
            flush()
            sections.append((r["content"], []))
            paragraph_of[r["idx"]] = r["content"]  # heading cũng là 1 đoạn văn, để chấm điểm câu hỏi trích dẫn heading
        elif r["type"] in _CAPTION_TYPES:
            texts, idxs = captions.setdefault(r["pidx"], ([], []))          # "pidx": ([các câu của caption], [idx của chúng])
            texts.append(r["content"])
            idxs.append(r["idx"])   
        elif r["type"] in _TEXT_TYPES:
            if r["pidx"] != last_pidx:
                flush()
            current.append(r["content"])
            current_idx.append(r["idx"])
            last_pidx = r["pidx"]                   # chỉ dòng chữ mới cập nhật: chú thích không cắt đoạn văn
    flush()
    if captions:
        captions_paragraphs = []
        for texts, idxs in captions.values():
            caption = " ".join(texts)                   # ghép các câu của 1 caption
            captions_paragraphs.append(caption)         # mỗi caption 1 đoạn
            for i in idxs:
                paragraph_of[i] = caption               # idx nào của caption cũng tra ra nguyên caption đó
        sections.append(("Figures and tables", captions_paragraphs))  # thêm 1 mục cuối là tất cả các caption
    return Paper(paper_id=paper_id, title=title, sections=sections), paragraph_of


def evidence_paragraphs(qa: dict, paragraph_of: dict[int, str], text_fallback: bool = False) -> list[str]:
    """
    Đoạn văn chứa từng câu bằng chứng, theo thứ tự, không lặp.

    1. Có idx (answer_evidence_mapped) -> đoạn của idx đó (cùng pidx) trong paper.
    2. Có text_fallback và không idx nào dùng được -> đoạn có chứa NGUYÊN VĂN câu trong answer_evidence_sent.
    """
    found: list[str] = []       # toàn bộ đoạn văn chứa câu bằng chứng của 1 câu hỏi

    # Xét từng câu bằng chứng ra idx -> đoạn văn chứa câu đó (cùng pidx) trong paper
    for ev in qa.get("answer_evidence_mapped") or []:
        for idx in ev.get("idx") or []:
            if idx is not None and idx in paragraph_of and paragraph_of[idx] not in found:
                found.append(paragraph_of[idx])         
    if found or not text_fallback:
        return found

    # Chạy khi bước tra theo idx không tìm được đoạn nào
    paragraphs = list(dict.fromkeys(paragraph_of.values()))  # các đoạn văn theo idx, không lặp, theo thứ tự xuất hiện trong paper
    for sent in qa.get("answer_evidence_sent") or []:
        norm = normalize(sent or "")                # chuẩn hóa từng câu đáp án
        if len(norm) < 10:
            continue
        for p in paragraphs:
            if norm in normalize(p) and p not in found:
                found.append(p)
                break
    return found

def build(papers_rows: list[dict], qa_rows: list[dict], n_papers: int, n_questions: int,
          unanswerable_ratio: float, seed: int, title_in_question: bool = True,
          text_fallback: bool = False, keep_review: bool = False) -> tuple[dict[str, str], list[dict], dict]:
    rng = random.Random(seed)
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in papers_rows:
        by_paper[r["paper_id"]].append(r)
    with_questions = sorted({q["paper_id"] for q in qa_rows} & set(by_paper))       # set(by_paper) cho ra tập các paper có chữ -> sử dụng được
    chosen = rng.sample(with_questions, min(n_papers, len(with_questions)))

    documents: dict[str, str] = {}
    answerable: list[dict] = []
    unanswerable: list[dict] = []
    skipped = {"unlabeled": 0, "no_evidence": 0}
    rescued_by_text = 0
    review_cuts: dict[str, dict] = {}
    qa_by_paper: dict[str, list[dict]] = defaultdict(list)

    for q in qa_rows:
        qa_by_paper[q["paper_id"]].append(q)

    for i, paper_id in enumerate(chosen, start=1):
        rows = by_paper[paper_id]
        if not keep_review:
            rows, cut = strip_peer_review(rows)         # các row cắt, kèm đoạn cut    
            if cut is not None:
                review_cuts[paper_id] = cut
        paper, paragraph_of = to_paper(paper_id, rows)

        filename = f"{i:02d}-{slugify(paper.title)}.txt"
        body = paper.render()
        documents[filename] = body
        tags = ["en", "peerqa", paper_id.split("/")[0], slugify(paper.title, 40)]

        for qa in qa_by_paper[paper_id]:
            question = with_title(qa["question"], paper.title) if title_in_question else qa["question"].strip()
            if qa.get("answerable") is False:
                unanswerable.append({"source_id": qa["question_id"], "question": question, "answerable": False,
                                     "tags": tags + ["unanswerable"]})
                continue

            if qa.get("answerable") is not True:
                skipped["unlabeled"] += 1
                continue
            evidence = pick_paragraphs(evidence_paragraphs(qa, paragraph_of, text_fallback), body)
            if text_fallback and evidence and not pick_paragraphs(evidence_paragraphs(qa, paragraph_of), body):
                rescued_by_text += 1
            if not evidence:
                skipped["no_evidence"] += 1
                continue

            case = {"source_id": qa["question_id"], "question": question, "answerable": True,
                    "evidence_paragraphs": evidence, "expected_documents": [filename], "tags": tags}

            if (qa.get("answer_free_form") or "").strip():
                case["reference_answers"] = [qa["answer_free_form"].strip()]
            answerable.append(case)

    cases = sample_cases(answerable, unanswerable, n_questions, unanswerable_ratio, rng, prefix="p") 
    stats = {
        "papers": len(documents),
        "papers_available": len(with_questions),
        "words": sum(len(b.split()) for b in documents.values()),
        "answerable_pool": len(answerable),
        "unanswerable_pool": len(unanswerable),
        "answerable": sum(c["answerable"] for c in cases),
        "unanswerable": sum(not c["answerable"] for c in cases),
        **{f"skipped_{k}": v for k, v in skipped.items()},
        "rescued_by_text": rescued_by_text,
        "review_cuts": review_cuts,
    }
    return documents, cases, stats




def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PeerQA into documents + an eval dataset")
    parser.add_argument("--data", type=Path, default=Path("eval/peerqa/data"), help="Thư mục có papers.jsonl + qa.jsonl")
    parser.add_argument("--docs-out", type=Path, default=Path("eval/peerqa/docs"))
    parser.add_argument("--out", type=Path, default=Path("eval/dataset_peerqa.json"))
    parser.add_argument("--papers", type=int, default=20, help="Số paper (mỗi paper 1 file)")
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--unanswerable-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-questions", action="store_true",
                        help="Giữ nguyên câu hỏi, không gắn tiêu đề paper (chỉ hợp khi app có 1 paper)")
    parser.add_argument("--text-fallback", action="store_true",
                        help="Câu không có idx: thử tìm đoạn chứa nguyên văn answer_evidence_sent "
                             "(mặc định tắt = chỉ dùng vị trí gốc như bài báo PeerQA)")
    parser.add_argument("--keep-review", action="store_true", 
                        help="Giữ phần Open Peer Review của paper F1000 (mặc định: cắt bỏ, vì nhãn chỉ xét thân bài)")
    parser.add_argument("--force", action="store_true", help="Ghi đè nếu đã có")

    args = parser.parse_args()
    if not 0 <= args.unanswerable_ratio <= 1:
        sys.exit("--unanswerable-ratio phải trong khoảng 0..1")

    documents, cases, stats = build(read_jsonl(args.data / "papers.jsonl"), read_jsonl(args.data / "qa.jsonl"),
                                    args.papers, args.questions, args.unanswerable_ratio, args.seed,
                                    not args.raw_questions, args.text_fallback)
    write_outputs(documents, cases, args.docs_out, args.out, args.force)

    print(f"Tài liệu: {stats['papers']} paper (có sẵn {stats['papers_available']}), ~{stats['words']} từ -> {args.docs_out}")
    print(f"Câu hỏi : {stats['answerable']} có đáp án + {stats['unanswerable']} không có đáp án -> {args.out}")
    print(f"          (nguồn: {stats['answerable_pool']} / {stats['unanswerable_pool']} câu; bỏ "
          f"{stats['skipped_unlabeled']} câu chưa gán nhãn, {stats['skipped_no_evidence']} câu không có bằng chứng dùng được)")
    mode = (f"có --text-fallback: cứu thêm {stats['rescued_by_text']} câu bằng khớp chữ" if args.text_fallback
            else "chỉ dùng vị trí bằng chứng gốc (answer_evidence_mapped)")
    print(f"Bằng chứng: {mode}")

    cuts = stats["review_cuts"]
    if cuts:
        print(f"Paper F1000: cắt phần peer review ở {sum(c['cut_idx'] is not None for c in cuts.values())}/{len(cuts)} paper")
        for pid, c in sorted(cuts.items()):
            if c["cut_idx"] is None:
                print(f" {pid}: KHÔNG thấy dấu hiệu review -> giữ nguyên")
                continue
            print(f" {pid}: bỏ {c['dropped']} dòng từ idx {c['cut_idx']}")
            print(f"        dòng cuối giữ lại: {c['before']!r}")
            print(f"        dòng đầu bị cắt  : {c['cut_text']!r}")
            if c["leftover"]:
                print(f"    CẢNH BÁO: còn dấu hiệu ở idx {c['leftover'][:10]} -> xem lại")
    print(f"Ước tính API cho 1 lần evaluate_retrieval: {len(cases)} embed + {len(cases)} rerank")


if __name__ == "__main__":
    main()