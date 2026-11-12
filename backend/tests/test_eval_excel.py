"""MỤC ĐÍCH: Test xuất báo cáo eval ra Excel (Phase 11, tiện ích). Không cần DB, không gọi API."""
import json
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from app.evaluation.excel import GREEN, RED, YELLOW, export_excel, rank_fill, verdict  # noqa: E402

DATASET = [{"id": "a1", "question": "Q1", "answerable": True, "expected_texts": ["câu bằng chứng 1"],
            "expected_documents": ["doc.txt"], "answer_contains": ["x"], "reference_answers": ["x", "xy"]}]


def _write(tmp_path: Path, report: dict) -> Path:
    ds = tmp_path / "dataset.json"
    ds.write_text(json.dumps(DATASET), encoding="utf-8")
    base = {"created_at": "2026-10-27T03:20:00", "label": "t", "application_id": "squad",
            "dataset": {"path": str(ds), "sha256_12": "abc"}, "config": {"embedding": "fake/fake"}}
    path = tmp_path / "20261027-032000-x.json"
    path.write_text(json.dumps({**base, **report}), encoding="utf-8")
    return path


def test_rank_and_verdict_colors() -> None:
    assert rank_fill(1) is GREEN and rank_fill(3) is YELLOW and rank_fill(None) is RED
    assert verdict({"answerable": False, "abstained": False})[1] is RED          # trả lời câu bẫy = bịa
    assert verdict({"answerable": True, "abstained": True})[1] is YELLOW         # từ chối nhầm
    assert verdict({"answerable": True, "abstained": False, "facts_ok": True, "citation_hit": True})[0] == "đúng"


def test_export_retrieval_report(tmp_path) -> None:
    m = {"hit@1": 0.5, "mrr@5": 0.75}
    mode = {"metrics": m, "ci": {"hit@1": [0.0, 1.0], "mrr@5": None}, "by_tag": {"en": {"n": 2, "hit@1": 0.5}},
            "latency": {"p50_ms": 10.0, "p95_ms": 20.0}, "unscored": 0}
    path = _write(tmp_path, {
        "kind": "retrieval", "ks": [1, 5],
        "counts": {"cases": 2, "answerable": 2, "scored": 2, "chunks_in_app": 9},
        "embedding_latency": {"p50_ms": None, "p95_ms": None},
        "modes": {"vector": mode, "hybrid": mode},
        "cases": [{"id": "a1", "question": "Q1", "tags": [], "n_relevant": 1, "ranks": {"vector": [1], "hybrid": []}}],
    })
    out = export_excel(path)
    assert out == path.with_suffix(".xlsx")
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Tổng quan", "Theo tag", "Từng câu", "Cấu hình"]
    cases = wb["Từng câu"]
    assert [c.value for c in cases[2]] == ["a1", "Q1", "câu bằng chứng 1", "x | xy", "doc.txt", 1, "1", "không thấy (>5)"]
    assert cases["H2"].fill.fgColor.rgb.endswith(RED.fgColor.rgb[-6:])


def test_export_generation_report_without_dataset_file(tmp_path) -> None:
    path = _write(tmp_path, {
        "kind": "generation",
        "counts": {"cases": 2, "errors": 1, "answerable": 1, "unanswerable": 0},
        "metrics": {"answer_rate": 1.0, "invalid_citation_count": 0}, "ci": {"answer_rate": [1.0, 1.0]},
        "abstain_reasons": {}, "latency": {"total": {"p50_ms": 900.0, "p95_ms": 900.0}},
        "tokens": {"input": 100, "output": 20},
        "cases": [{"id": "a1", "question": "Q1", "answerable": True, "answer": "x [Source 1]", "abstained": False,
                   "abstain_reason": None, "sources": [{"number": 1, "filename": "doc.txt", "page": None}],
                   "facts_ok": True, "citation_hit": True, "squad_f1": 1.0, "top_rerank_score": 0.9, "total_ms": 900.0},
                  {"id": "a2", "question": "Q2", "answerable": True, "error": "LLMError: 429"}],
    })
    (tmp_path / "dataset.json").unlink()        # dataset đã bị xoá: vẫn xuất được, chỉ trống cột đáp án mẫu
    wb = openpyxl.load_workbook(export_excel(path))
    assert wb.sheetnames == ["Tổng quan", "Từng câu", "Cấu hình"]
    rows = list(wb["Từng câu"].iter_rows(min_row=2, max_row=3, values_only=True))
    assert rows[0][:3] == ("a1", "đúng", "Q1") and rows[0][4] is None and rows[0][6] == "[1] doc.txt"
    assert rows[1][1] == "lỗi API (không tính điểm)" and rows[1][5] == "LLMError: 429"
