"""
MỤC ĐÍCH: Xuất lại file Excel cho báo cáo CŨ (chạy trước khi có tính năng tự xuất).

    python scripts/export_excel.py                       # báo cáo mới nhất trong eval/results
    python scripts/export_excel.py eval/results/*.json   # nhiều file 1 lúc

Từ nay evaluate_retrieval.py / evaluate_generation.py tự xuất .xlsx sau mỗi lần chạy, không cần script này.
"""
import argparse
import sys
from pathlib import Path

from app.evaluation.excel import export_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất báo cáo eval (.json) ra Excel (.xlsx)")
    parser.add_argument("reports", type=Path, nargs="*", help="File .json; bỏ trống = file mới nhất")
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    paths = args.reports
    if not paths:
        files = sorted(args.results.glob("*.json"))   # tên bắt đầu bằng thời gian -> cuối cùng = mới nhất
        if not files:
            sys.exit(f"Không có file .json trong {args.results}")
        paths = [files[-1]]
    for path in paths:
        print(f"Đã lưu {export_excel(path)}")


if __name__ == "__main__":
    main()
