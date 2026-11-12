"""
MỤC ĐÍCH: Xuất 1 báo cáo đánh giá (eval/results/*.json) ra Excel cho dễ đọc (Phase 11, tiện ích).

evaluate_retrieval.py và evaluate_generation.py gọi export_excel() ngay sau save_report(),
nên mỗi lần chạy có cả .json (để máy so bằng compare_results.py) lẫn .xlsx (để người đọc).

Retrieval:  Tổng quan | Theo tag | Từng câu (hạng chunk đúng, tô màu) | Cấu hình
Generation: Tổng quan | Từng câu (câu trả lời nguyên văn + kết luận, tô màu) | Cấu hình
"""
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER = PatternFill("solid", fgColor="1F4E78")
GREEN = PatternFill("solid", fgColor="C6EFCE")
YELLOW = PatternFill("solid", fgColor="FFEB9C")
RED = PatternFill("solid", fgColor="FFC7CE")
GREY = PatternFill("solid", fgColor="D9D9D9")
WRAP = Alignment(wrap_text=True, vertical="top")


# ---------- tiện ích chung ----------

def _header(ws, titles: list[str], row: int = 1) -> None:
    for col, title in enumerate(titles, start=1):
        cell = ws.cell(row=row, column=col, value=title)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _widths(ws, widths: dict[int, float]) -> None:
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width


def _ci_text(ci: list[float] | None) -> str:
    return "-" if ci is None else f"{ci[0]:.2f} – {ci[1]:.2f}"


def _scope_text(scope: dict | None) -> str:
    """Báo cáo cũ không có "scope" -> tìm trong cả application."""
    if not scope or scope.get("mode", "all") == "all":
        return "cả application"
    return f"từng paper (bỏ qua {len(scope.get('skipped', []))} câu không có paper)"


def _info_block(ws, report: dict, extra: list[tuple[str, object]]) -> int:
    """Vài dòng thông tin đầu sheet Tổng quan; trả về dòng trống kế tiếp."""
    rows = [
        ("Nhãn", report.get("label") or "-"),
        ("Thời gian", report.get("created_at", "-")),
        ("Application", report["application_id"]),
        ("Dataset", f"{report['dataset']['path']} (sha {report['dataset']['sha256_12']})"),
        ("Phạm vi tìm", _scope_text(report.get("scope"))),
        *extra,
    ]
    for r, (key, value) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=key).font = Font(bold=True)
        ws.cell(row=r, column=2, value=value)
    return len(rows) + 2


def _legend(ws, row: int, items: list[tuple[PatternFill, str]]) -> None:
    for offset, (fill, text) in enumerate(items):
        ws.cell(row=row + offset, column=1).fill = fill
        ws.cell(row=row + offset, column=2, value=text)


def _sheet_config(wb: Workbook, report: dict) -> None:
    ws = wb.create_sheet("Cấu hình")
    _header(ws, ["Thiết lập", "Giá trị"])
    for r, (key, value) in enumerate(report.get("config", {}).items(), start=2):   # config không có key
        ws.cell(row=r, column=1, value=key)
        ws.cell(row=r, column=2, value=str(value))
    _widths(ws, {1: 22, 2: 60})


def _load_dataset(report: dict) -> dict[str, dict]:
    """Câu bằng chứng / đáp án mẫu lấy từ dataset; không thấy file thì để trống các cột đó."""
    path = Path(report["dataset"]["path"])
    if not path.exists():
        return {}
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}


def _reference(case: dict) -> str:
    return " | ".join(case.get("reference_answers") or case.get("answer_contains") or [])


# ---------- retrieval ----------

def rank_fill(first_rank: int | None) -> PatternFill:
    if first_rank is None:
        return RED                                   # không có trong top-k
    return GREEN if first_rank == 1 else YELLOW      # đứng đầu / có nhưng không đứng đầu


def _retrieval_overview(wb: Workbook, report: dict) -> None:
    ws = wb.active
    ws.title = "Tổng quan"
    counts = report["counts"]
    top = _info_block(ws, report, [
        ("Số chunk trong app", counts["chunks_in_app"]),
        ("Số câu được chấm", f"{counts['scored']}/{counts['answerable']} câu có đáp án"),
    ])
    modes = list(report["modes"])
    titles = ["Chỉ số"]
    for m in modes:
        titles += [m, f"{m} 95% CI"]
    _header(ws, titles, row=top)

    names = list(report["modes"][modes[0]]["metrics"])
    for r, name in enumerate(names, start=top + 1):
        ws.cell(row=r, column=1, value=name).font = Font(bold=True)
        for i, m in enumerate(modes):
            ws.cell(row=r, column=2 + 2 * i, value=report["modes"][m]["metrics"][name]).number_format = "0.00"
            ws.cell(row=r, column=3 + 2 * i, value=_ci_text(report["modes"][m]["ci"].get(name)))
    r = top + 1 + len(names)
    for stat in ("p50_ms", "p95_ms"):
        ws.cell(row=r, column=1, value=f"latency {stat}").font = Font(bold=True)
        for i, m in enumerate(modes):
            ws.cell(row=r, column=2 + 2 * i, value=report["modes"][m]["latency"][stat]).number_format = "0"
        r += 1
    emb = report["embedding_latency"]
    ws.cell(row=r, column=1, value="embedding p50 / p95 ms").font = Font(bold=True)
    ws.cell(row=r, column=2, value="-" if emb["p50_ms"] is None else f"{emb['p50_ms']:.0f} / {emb['p95_ms']:.0f}")

    _widths(ws, {1: 24, 2: 40, **{c: 14 for c in range(3, len(titles) + 1)}})
    ws.freeze_panes = ws.cell(row=top + 1, column=2)


def _retrieval_by_tag(wb: Workbook, report: dict) -> None:
    ws = wb.create_sheet("Theo tag")
    modes = list(report["modes"])
    by_tag = report["modes"][modes[0]]["by_tag"]
    if not by_tag:
        ws["A1"] = "Không có tag"
        return
    names = [n for n in next(iter(by_tag.values())) if n != "n"]
    titles = ["Tag", "Số câu"] + [f"{m} {name}" for name in names for m in modes]
    _header(ws, titles)
    for r, tag in enumerate(by_tag, start=2):
        ws.cell(row=r, column=1, value=tag)
        ws.cell(row=r, column=2, value=by_tag[tag]["n"])
        col = 3
        for name in names:
            for m in modes:
                ws.cell(row=r, column=col, value=report["modes"][m]["by_tag"][tag][name]).number_format = "0.00"
                col += 1
    _widths(ws, {1: 30, 2: 9, **{c: 16 for c in range(3, len(titles) + 1)}})
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def _retrieval_cases(wb: Workbook, report: dict, dataset: dict[str, dict]) -> None:
    ws = wb.create_sheet("Từng câu")
    modes = list(report["modes"])
    max_k = max(report["ks"])
    titles = ["id", "Câu hỏi", "Câu bằng chứng", "Đáp án mẫu", "File đúng", "Số chunk đúng"] + modes
    _header(ws, titles)
    for r, case in enumerate(report["cases"], start=2):
        src = dataset.get(case["id"], {})
        values = [case["id"], case["question"], "\n".join(src.get("expected_texts", [])), _reference(src),
                  ", ".join(src.get("expected_documents", [])), case["n_relevant"]]
        for col, value in enumerate(values, start=1):
            ws.cell(row=r, column=col, value=value).alignment = WRAP
        for col, m in enumerate(modes, start=len(values) + 1):
            ranks = case["ranks"][m]
            # hạng của MỌI chunk đúng tìm được, vd "1, 4"
            cell = ws.cell(row=r, column=col, value=", ".join(map(str, ranks)) if ranks else f"không thấy (>{max_k})")
            cell.fill = rank_fill(ranks[0] if ranks else None)
            cell.alignment = Alignment(horizontal="center", vertical="top")
    _widths(ws, {1: 8, 2: 45, 3: 60, 4: 30, 5: 22, 6: 9, **{c: 15 for c in range(7, len(titles) + 1)}})
    ws.freeze_panes = "C2"   # cột id + câu hỏi và dòng tiêu đề đứng yên khi cuộn
    ws.auto_filter.ref = ws.dimensions
    _legend(ws, len(report["cases"]) + 3, [(GREEN, "chunk đúng đứng hạng 1"),
                                          (YELLOW, f"có trong top-{max_k} nhưng không đứng đầu"),
                                          (RED, f"không có trong top-{max_k}")])


# ---------- generation ----------

def verdict(row: dict) -> tuple[str, PatternFill]:
    """Kết luận 1 câu, giống phần "Cần đọc lại bằng mắt" của evaluate_generation.py."""
    if "error" in row:
        return "lỗi API (không tính điểm)", GREY
    if not row["answerable"]:
        return ("từ chối đúng", GREEN) if row["abstained"] else ("BỊA: trả lời câu không có đáp án", RED)
    if row["abstained"]:
        return "từ chối nhầm", YELLOW
    if row.get("facts_ok") is False:
        return "trả lời nhưng thiếu đáp án", RED
    if row.get("citation_hit") is False:
        return "đúng đáp án, trích sai chunk", YELLOW
    return "đúng", GREEN


def _generation_overview(wb: Workbook, report: dict) -> None:
    ws = wb.active
    ws.title = "Tổng quan"
    c = report["counts"]
    top = _info_block(ws, report, [
        ("Số câu", f"{c['cases']} ({c['answerable']} có đáp án, {c['unanswerable']} không, {c['errors']} lỗi API)"),
        ("Tokens vào / ra", f"{report['tokens']['input']} / {report['tokens']['output']}"),
    ])
    _header(ws, ["Chỉ số", "Giá trị", "95% CI"], row=top)
    r = top + 1
    for name, value in report["metrics"].items():
        ws.cell(row=r, column=1, value=name).font = Font(bold=True)
        cell = ws.cell(row=r, column=2, value=value)
        if not isinstance(value, int):
            cell.number_format = "0.00"
        ws.cell(row=r, column=3, value=_ci_text(report["ci"].get(name)))
        r += 1
    r += 1
    _header(ws, ["Latency (ms)", "p50", "p95"], row=r)
    for stage, lat in report["latency"].items():
        r += 1
        ws.cell(row=r, column=1, value=stage).font = Font(bold=True)
        ws.cell(row=r, column=2, value=lat["p50_ms"]).number_format = "0"
        ws.cell(row=r, column=3, value=lat["p95_ms"]).number_format = "0"
    r += 2
    ws.cell(row=r, column=1, value="Lý do từ chối").font = Font(bold=True)
    ws.cell(row=r, column=2, value=", ".join(f"{k}: {v}" for k, v in report["abstain_reasons"].items()) or "-")
    _widths(ws, {1: 26, 2: 40, 3: 16})
    ws.freeze_panes = ws.cell(row=top + 1, column=2)


def _generation_cases(wb: Workbook, report: dict, dataset: dict[str, dict]) -> None:
    ws = wb.create_sheet("Từng câu")
    titles = ["id", "Kết luận", "Câu hỏi", "Có đáp án?", "Đáp án mẫu", "Câu trả lời của chatbot", "Nguồn trích",
              "Lý do từ chối", "squad_f1", "faithfulness", "top rerank", "total ms"]
    _header(ws, titles)
    for r, row in enumerate(report["cases"], start=2):
        text, fill = verdict(row)
        faith = (row.get("faithfulness") or {}).get("score")
        values = [
            row["id"], text, row["question"], "có" if row["answerable"] else "không",
            _reference(dataset.get(row["id"], {})),
            row.get("answer") or row.get("error", ""),
            ", ".join(f"[{s['number']}] {s['filename']}" for s in row.get("sources", [])),
            row.get("abstain_reason") or "",
            row.get("squad_f1"), faith, row.get("top_rerank_score"), row.get("total_ms"),
        ]
        for col, value in enumerate(values, start=1):
            cell = ws.cell(row=r, column=col, value=value)
            cell.alignment = WRAP
            if col >= 9:
                cell.number_format = "0" if col == 12 else "0.00"
        ws.cell(row=r, column=2).fill = fill
    _widths(ws, {1: 8, 2: 26, 3: 45, 4: 9, 5: 30, 6: 70, 7: 28, 8: 16, 9: 9, 10: 11, 11: 10, 12: 9})
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions   # lọc cột "Kết luận" để chỉ xem câu sai
    _legend(ws, len(report["cases"]) + 3, [(GREEN, "đúng / từ chối đúng"),
                                          (YELLOW, "từ chối nhầm / trích sai chunk"),
                                          (RED, "bịa hoặc thiếu đáp án"), (GREY, "lỗi API")])


# ---------- điểm vào ----------

def export_excel(json_path: Path) -> Path:
    """Đọc 1 báo cáo .json, ghi file .xlsx cùng tên bên cạnh, trả về đường dẫn .xlsx."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    dataset = _load_dataset(report)
    wb = Workbook()
    if report["kind"] == "retrieval":
        _retrieval_overview(wb, report)
        _retrieval_by_tag(wb, report)
        _retrieval_cases(wb, report, dataset)
    elif report["kind"] == "generation":
        _generation_overview(wb, report)
        _generation_cases(wb, report, dataset)
    else:
        raise ValueError(f"Không biết xuất báo cáo loại '{report['kind']}'")
    _sheet_config(wb, report)
    out = json_path.with_suffix(".xlsx")
    wb.save(out)
    return out
