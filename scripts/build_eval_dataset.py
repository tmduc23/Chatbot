"""
MỤC ĐÍCH: Gộp 2 file eval cũ (Phase 4-6) thành bộ dữ liệu Phase 11: eval/dataset.json.

    python scripts/build_eval_dataset.py

eval/retrieval_smoke.json  [{"query", "expected_text"}]  -> answerable=true,  id a01, a02...
eval/unanswerable.json     [{"query"}]                   -> answerable=false, id u01, u02...

Chạy 1 lần rồi SỬA TAY eval/dataset.json: thêm answer_contains, thêm câu hỏi mới...
Đã có file thì không ghi đè (thêm --force nếu thật sự muốn tạo lại).
"""
import argparse
import json
import sys
from pathlib import Path


def language_tag(text: str) -> str:
    # có ký tự ngoài ASCII (dấu tiếng Việt) -> "vi"; đủ dùng để xem điểm theo ngôn ngữ câu hỏi
    return "vi" if any(ord(ch) > 127 for ch in text) else "en"


def convert(answerable: list[dict], unanswerable: list[dict]) -> list[dict]:
    cases = []
    for i, item in enumerate(answerable, start=1):
        cases.append({
            "id": f"a{i:02d}",
            "question": item["query"],
            "answerable": True,
            "expected_texts": [item["expected_text"]],
            "answer_contains": [],
            "tags": [language_tag(item["query"])],
        })
    for i, item in enumerate(unanswerable, start=1):
        cases.append({
            "id": f"u{i:02d}",
            "question": item["query"],
            "answerable": False,
            "tags": [language_tag(item["query"])],
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval/dataset.json from the old eval files")
    parser.add_argument("--answerable", type=Path, default=Path("eval/retrieval_smoke.json"))
    parser.add_argument("--unanswerable", type=Path, default=Path("eval/unanswerable.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--force", action="store_true", help="Ghi đè nếu file đã tồn tại")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} đã tồn tại (có thể bạn đã sửa tay). Thêm --force để ghi đè.")
    cases = convert(
        json.loads(args.answerable.read_text(encoding="utf-8")),
        json.loads(args.unanswerable.read_text(encoding="utf-8")),
    )
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_ans = sum(c["answerable"] for c in cases)
    print(f"Đã ghi {args.out}: {n_ans} câu có đáp án, {len(cases) - n_ans} câu không có đáp án")


if __name__ == "__main__":
    main()