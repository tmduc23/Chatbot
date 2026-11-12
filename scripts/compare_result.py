"""
MỤC ĐÍCH: So 2 file kết quả đánh giá cùng loại (A/B test), vd. trước và sau khi đổi embedding model.

    python scripts/compare_results.py eval/results/A-retrieval.json eval/results/B-retrieval.json

In ra: cấu hình nào khác nhau, chỉ số nào tăng/giảm, câu nào đổi kết quả.
"""
import argparse
import json
import sys
from pathlib import Path

from app.evaluation.report import compare_reports, config_diff, fmt


def changed_cases(a: dict, b: dict) -> list[str]:
    lines = []
    cases_b = {c["id"]: c for c in b["cases"]}
    for ca in a["cases"]:
        cb = cases_b.get(ca["id"])
        if cb is None:
            continue
        if a["kind"] == "retrieval":
            for mode, ranks in ca["ranks"].items():
                rb = cb["ranks"].get(mode)
                if rb is not None and ranks[:1] != rb[:1]:
                    lines.append(f"{ca['id']} {mode}: hạng đầu {ranks[:1] or '-'} -> {rb[:1] or '-'}  {ca['question'][:45]}")
        elif "error" not in ca and "error" not in cb and ca["abstained"] != cb["abstained"]:
            state = lambda c: "từ chối" if c["abstained"] else "trả lời"
            lines.append(f"{ca['id']}: {state(ca)} -> {state(cb)}  {ca['question'][:50]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation result files")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))

    try:
        rows = compare_reports(a, b)
    except ValueError as exc:
        sys.exit(str(exc))
    if a.get("dataset", {}).get("sha256_12") != b.get("dataset", {}).get("sha256_12"):
        print("⚠ 2 lần chạy dùng dataset KHÁC nhau: so sánh chỉ mang tính tham khảo\n")

    print(f"A = {args.a.name}\nB = {args.b.name}\n")
    diff = config_diff(a, b)
    print("Cấu hình khác nhau:" if diff else "Cấu hình giống nhau (khác biệt chỉ do dao động của API/LLM)")
    for key, (va, vb) in diff.items():
        print(f"  {key}: {va} -> {vb}")

    print(f"\n{'chỉ số':<32}{'A':>9}{'B':>9}{'B-A':>9}")
    for name, va, vb, delta in rows:
        is_ms = name.endswith("_ms")
        show = (lambda v: "-" if v is None else f"{v:.0f}") if is_ms else fmt
        d = "-" if delta is None else (f"{delta:+.0f}" if is_ms else f"{delta:+.2f}")
        print(f"{name:<32}{show(va):>9}{show(vb):>9}{d:>9}")

    lines = changed_cases(a, b)
    print(f"\nCâu đổi kết quả ({len(lines)}):")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()