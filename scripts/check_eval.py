"""
MỤC ĐÍCH: Giúp viết eval/retrieval_smoke.json ĐÚNG với CV thật. Không gọi   API nào (chỉ đọc DB, không tốn quota)

    #1) Xuất toàn bộ chunk của application ra fie để đọc và chọn expected_text
    python scripts/check_eval.py --app cv --dump
        -> ghi eval/chunks_cv.txt   
    
    # 2) Kiểm ttra từng  expected_text trong eval/retrieval_smoke.json
    python scripts/check_eval.py --app cv


Kết quả cho mỗi câu: 
    OK          expected_text có trong 1-2 chunk    -> tốt
    KHÔNG CÓ    không chunk nào chứa                -> sai chỉnh tả, hoặc CV không có thông tin
    QUÁ CHUNG   có trong >= 3 chunk                 -> script tune_threshold dễ bắt nhầm chunk, chọn cụm cụ thể hơn

    
SELECT
    chunks.chunk_index,
    chunks.page,
    chunks.text,
    documents.filename
FROM chunks
INNER JOIN documents
    ON documents.id = chunks.document_id
WHERE chunks.application_id = 'cv'
ORDER BY documents.filename, chunks.chunk_index;
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Chunk, Document

MAX_MATCHES = 2


def classify(expected_text: str, texts: list[str]) -> tuple[str, list[int]]:
    """Trả (trạng thái, vị trí các chunk chứa expected_text). So không phân biệt hoa thường, giống tune_threshold."""
    needle = expected_text.lower()
    hits = [i for i, t in enumerate(texts) if needle in t.lower()]      # hits lưu vị trí các chunk chứa từ cần tìm

    # Xác định trạng thái
    if not hits:
        return "KHÔNG CÓ", hits
    if len(hits) > MAX_MATCHES:
        return "QUÁ CHUNG", hits
    return "OK",  hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval/retrieval_smoke.json against the real chunks")
    parser.add_argument("--app", required=True)
    parser.add_argument("--eval", type=Path, default=Path("eval/retrieval_smoke.json"))
    parser.add_argument("--dump", action="store_true", help="Ghi mọi chunk ra eval/chunks_<app>.txt")   # xuất file ra để đọc
    args = parser.parse_args()


    with SessionLocal() as db:
        rows = db.execute(
            select(Chunk.chunk_index, Chunk.page, Chunk.text, Document.filename)
            .join(Document, Document.id == Chunk.document_id)   # gộp thêm bảng Document, để lấy Document.id -> Document.fillename
            .where(Chunk.application_id == args.app)            # điều kiện lọc
            .order_by(Document.filename, Chunk.chunk_index)
        ).all()

    if not rows:
        print(f"Application '{args.app}' không có chunk nào -> kiểm tra --app hoặc chạy lại ingest")
        return

    # Nhãn và nội dung
    labels = [f"{r.filename} #{r.chunk_index}" + (f" (page {r.page})" if r.page is not None else "") for r in rows]
    texts = [r.text for r in rows]

    # Xuất toàn bộ chunk
    # Ghép nhãn và nội dung
    if args.dump:
        out = Path("eval") / f"chunks_{args.app}.txt"
        out.write_text("\n\n".join(f"===== {label} =====\n{text}" for label, text in zip(labels, texts)),
                       encoding="utf-8")
        print(f"Đã ghi {len(rows)} chunk vào {out}")            # ghi file và kết thúc
        return                                                  # dừng, không đọc file JSON và không kiểm tra expected text

    cases = json.loads(args.eval.read_text(encoding="utf-8"))   # cases tổng số câu hỏi trong file đánh giá
    print(f"{len(rows)} chunk | {len(cases)} câu trong {args.eval}\n")
    bad = 0
    for case in cases:
        status, hits = classify(case["expected_text"], texts)   # trạng thái, label chứa expected trong hit
        bad += status != "OK"
        where = ", ".join(labels[i] for i in hits[:3]) + (" ..." if len(hits) > 3 else "")    # tra cứu labels, ghép thành chuỗi để in  
        print(f"{status:<10} {case['expected_text'][:30]!r:<34} {case['query'][:45]}")      # expected, query
        if where:
            print(f"{'':<10} -> {where}")
    print(f"\n{len(cases) - bad}/{len(cases)} câu OK" + ("" if bad == 0 else " -> sửa các câu còn lại rồi chạy lại"))

if __name__ == "__main__":
    main()