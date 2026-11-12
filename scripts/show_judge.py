"""
MỤC ĐÍCH: Xem giám khảo faithfulness đã chấm từng ý (YES / NO) trong 1 file kết quả.

    python scripts/show_judge.py eval/results/<file>.json            # in mọi câu đã chấm
    python scripts/show_judge.py eval/results/<file>.json --only-no  # chỉ câu có ý bị chấm NO / bỏ sót
    python scripts/show_judge.py eval/results/<file>.json --id qa023 # 1 câu, kèm câu hỏi + câu trả lời
    python scripts/show_judge.py eval/results/<file>.json --excel    # thêm file <file>-judge.xlsx cạnh file JSON

Chỉ đọc file, không gọi API.
"""
import argparse
import json
import sys
from pathlib import Path

MARK = {True: "YES", False: "NO ", None: "?? "}   # ?? = giám khảo bỏ sót ý này


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases") or data.get("metrics", {}).get("cases", [])


def write_excel(cases: list[dict], out: Path) -> None:
    """Sheet 'Từng ý': 1 dòng / 1 ý. Sheet 'Từng câu': 1 dòng / 1 câu (điểm, số ý YES/NO)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    fills = {"YES": PatternFill("solid", fgColor="C6EFCE"), "NO": PatternFill("solid", fgColor="FFC7CE"),
             "??": PatternFill("solid", fgColor="FFEB9C")}
    wrap = Alignment(wrap_text=True, vertical="top")
    wb = Workbook()

    ws = wb.active
    ws.title = "Từng ý"
    ws.append(["id", "Câu hỏi", "Câu trả lời", "Điểm câu", "Ý số", "Ý (claim)", "Giám khảo", "Lỗi"])
    for case in cases:
        f = case["faithfulness"]
        claims = f["claims"] or [{"text": "", "supported": None}]
        for i, cl in enumerate(claims, start=1):
            verdict = MARK[cl["supported"]].strip() if f["claims"] else ""
            ws.append([case["id"], case.get("question", "") if i == 1 else "",
                       case.get("answer", "") if i == 1 else "", f["score"] if i == 1 else None,
                       i if f["claims"] else None, cl["text"], verdict, f.get("error") or ""])
            if verdict:
                ws.cell(ws.max_row, 7).fill = fills[verdict]

    summary = wb.create_sheet("Từng câu")
    summary.append(["id", "Câu hỏi", "Câu trả lời", "Điểm", "YES", "NO", "Bỏ sót", "Lỗi"])
    for case in cases:
        f = case["faithfulness"]
        v = [cl["supported"] for cl in f["claims"]]
        summary.append([case["id"], case.get("question", ""), case.get("answer", ""), f["score"],
                        v.count(True), v.count(False), v.count(None), f.get("error") or ""])

    for sheet, widths in ((ws, [8, 45, 60, 9, 6, 60, 10, 20]), (summary, [8, 45, 70, 8, 6, 6, 8, 20])):
        for col, width in enumerate(widths, start=1):
            sheet.column_dimensions[chr(64 + col)].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "B2"
    wb.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xem giám khảo faithfulness chấm từng ý")
    parser.add_argument("result", type=Path, help="file JSON của evaluate_generation.py --faithfulness")
    parser.add_argument("--only-no", action="store_true", help="chỉ lấy câu có ý NO hoặc bỏ sót")
    parser.add_argument("--id", help="chỉ lấy 1 câu (vd. qa023), in kèm câu hỏi và câu trả lời")
    parser.add_argument("--excel", action="store_true", help="xuất thêm <file>-judge.xlsx cạnh file JSON")
    args = parser.parse_args()

    judged = [c for c in load_cases(args.result) if c.get("faithfulness")]
    if not judged:
        sys.exit("File không có kết quả giám khảo (chạy evaluate_generation.py với --faithfulness).")

    shown: list[dict] = []
    n_yes = n_no = n_miss = 0
    for case in judged:
        f = case["faithfulness"]
        verdicts = [cl["supported"] for cl in f["claims"]]
        n_yes += verdicts.count(True)
        n_no += verdicts.count(False)
        n_miss += verdicts.count(None)
        if args.id and case["id"] != args.id:
            continue
        if args.only_no and all(v is True for v in verdicts) and not f.get("error"):
            continue
        shown.append(case)
        score = "-" if f["score"] is None else f"{f['score']:.2f}"
        print(f"\n== {case['id']}  score={score}  {f.get('error') or ''}")
        if args.id:
            print(f"Q: {case.get('question', '')}\nA: {case.get('answer', '')}\n")
        for cl in f["claims"]:
            print(f"   {MARK[cl['supported']]} {cl['text']}")

    print(f"\n{len(shown)}/{len(judged)} câu được in | tổng ý: YES={n_yes} NO={n_no} bỏ sót={n_miss}")
    if args.excel and shown:
        out = args.result.with_name(args.result.stem + "-judge.xlsx")
        write_excel(shown, out)
        print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
