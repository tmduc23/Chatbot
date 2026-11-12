"""MỤC ĐÍCH: Test phần dùng chung paper_import.py + script chuyển bộ QASPER -> tài liệu + dataset đánh giá."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from app.evaluation.dataset import parse_case
from app.evaluation.paper_import import Paper, pick_paragraphs, sample_cases


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qasper = _load("import_qasper")

# ---------- phần dùng chung (paper_import.py) ----------

def test_render_writes_sections_as_headings() -> None:
    paper = Paper("p1", "My Paper", [("Abstract", ["We study X."]), ("Empty", []), ("Results", ["It works.", " "])])
    assert paper.render() == "My Paper\n\n## Abstract\n\nWe study X.\n\n## Results\n\nIt works."


def test_pick_paragraphs_keeps_whole_paragraphs_found_in_the_paper() -> None:
    para = "We train on the Penn Treebank corpus with 40k sentences and evaluate on section 23."
    body = f"Title\n\n## Data\n\n{para}\n\nShort one."
    picked = pick_paragraphs([para, "  " + para,                    # trùng (khác khoảng trắng) -> 1 lần
                              "Short one.",                         # quá ngắn
                              "Not in the paper at all, sorry about that."], body)
    assert picked == [para]      


def test_sample_fills_traps_with_holdout_only_when_needed() -> None:
    ans = [{"question": f"a{i}", "answerable": True} for i in range(10)]
    real = [{"question": "u1", "answerable": False}]
    hold = [{"question": f"h{i}", "answerable": False} for i in range(5)]
    cases = sample_cases(ans, real, 10, 0.3, random.Random(1), prefix="t", holdout=hold)
    traps = [c["question"] for c in cases if not c["answerable"]]
    assert len(cases) == 10 and len(traps) == 3 and "u1" in traps      # bẫy thật luôn được lấy trước
    assert {c["id"] for c in cases} >= {"ta001", "tu001"}


# ---------- QASPER ----------

# dạng câu trả lời, không có thì là null
def _ans(unanswerable=False, spans=(), free="", yes_no=None, evidence=(), highlighted=()):
    return {"answer": {"unanswerable": unanswerable, "extractive_spans": list(spans), "free_form_answer": free,
                       "yes_no": yes_no, "evidence": list(evidence), "highlighted_evidence": list(highlighted)}}


P1_TEXT = ("We evaluate on the Waseem dataset and the Davidson dataset of tweets. "
           "Our model improves F1 by 4 points over the SVM baseline BIBREF3.")
QASPER = {
    "p1": {"title": "Hate Speech Detection", "abstract": "We detect hate speech with BERT.",
           "full_text": [{"section_name": "Experiments ::: Data", "paragraphs": [P1_TEXT]}],
           "figures_and_tables": [{"file": "t1.png", "caption": "Table 1: F1 scores of every model on both datasets."}],
           "qas": [
               {"question_id": "q1", "question": "Which datasets do they use?", "answers": [
                   _ans(spans=["Waseem dataset", "Davidson dataset,"],
                        highlighted=["We evaluate on the Waseem dataset and the Davidson dataset of tweets."]),
                   _ans(free="Two tweet datasets", evidence=[P1_TEXT])]},
               {"question_id": "q2", "question": "Do they beat the baseline?", "answers": [
                   _ans(yes_no=True, evidence=["FLOAT SELECTED: Table 1: F1 scores of every model on both datasets."])]},
               {"question_id": "q3", "question": "What GPU did they use?", "answers": [_ans(True), _ans(True)]},
               {"question_id": "q4", "question": "Is the code public?", "answers": [_ans(True), _ans(yes_no=False)]},
           ]},
    "p2": {"title": "Another Paper", "abstract": "Something else.", "full_text": [], "figures_and_tables": [],
           "qas": [{"question_id": "q5", "question": "What is the main result?",
                    "answers": [_ans(free="It is good")]}]},
}


def _qasper_build(**kw):
    params = dict(n_papers=1, n_questions=10, unanswerable_ratio = 0.5, seed=1)
    params.update(kw)
    raw = {"p1": QASPER["p1"], "p2": QASPER["p2"]}
    return qasper.build(raw, **params)


def test_qasper_cases_follow_the_rules() -> None:
    documents, cases, stats = _qasper_build(seed=3)
    doc_name, body = next(iter(documents.items()))
    if "hate-speech" not in doc_name:                       # seed chọn p2 -> chọn lại cho chắc
        documents, cases, stats = qasper.build({"p1": QASPER["p1"]}, 1, 10, 0.5, 1)
        doc_name, body = next(iter(documents.items()))
    assert "## Experiments > Data" in body and "## Figures and tables" in body
    by_src = {c["source_id"]: c for c in cases}
    q1 = by_src["q1"]
    assert q1["question"] == 'In the paper "Hate Speech Detection": Which datasets do they use?'
    assert q1["evidence_paragraphs"] == [P1_TEXT]
    assert "expected_texts" not in q1 and "answer_contains" not in q1
    assert q1["reference_answers"] == ["Waseem dataset, Davidson dataset,", "Two tweet datasets"]   # y như bộ chấm gốc
    assert "extractive" in q1["tags"]
    q2 = by_src["q2"]
    assert q2["evidence_paragraphs"] == ["Table 1: F1 scores of every model on both datasets."]   # bỏ "FLOAT SELECTED:"
    assert q2["reference_answers"] == ["Yes"] and "boolean" in q2["tags"]
    assert by_src["q3"]["answerable"] is False and "unanswerable" in by_src["q3"]["tags"]
    assert "q4" not in by_src and stats["skipped_disagree"] == 1                # người chấm bất đồng -> bỏ
    for i, raw in enumerate(cases):
        parse_case(raw, i)              # chuyển thành evalcase

def test_qasper_drops_question_without_evidence_paragraph() -> None:
    paper = {"title": "T", "abstract": "We use BERT.", "full_text": [], "figures_and_tables": [],
             "qas": [{"question_id": "q1", "question": "Which model?", "answers": [
                 _ans(spans=["BERT"], highlighted=["We use BERT."])]}]}          # chỉ có câu bôi, không có đoạn
    _, cases, stats = qasper.build({"p": paper}, 1, 5, 0.0, 1, holdout=False)
    assert cases == [] and stats["skipped_no_evidence"] == 1



def test_qasper_holdout_traps_come_from_papers_not_loaded() -> None:
    documents, cases, stats = qasper.build(QASPER, 1, 10, 0.5, 1)
    loaded = next(iter(documents.values())).splitlines()[0]
    holdout = [c for c in cases if "holdout" in c["tags"]]
    assert holdout and all(loaded not in c["question"] and not c["answerable"] for c in holdout)
    _, cases_raw, _ = qasper.build(QASPER, 1, 10, 0.5, 1, title_in_question=False)
    assert not any("holdout" in c["tags"] for c in cases_raw) 

def test_qasper_main_writes_files_and_same_seed_same_output(tmp_path, monkeypatch) -> None:
    src = tmp_path / "qasper-dev-v0.3.json"
    src.write_text(json.dumps(QASPER), encoding="utf-8")
    docs, out = tmp_path / "docs", tmp_path / "dataset.json"
    argv = ["import_qasper.py", "--input", str(src), "--docs-out", str(docs), "--out", str(out), "--papers", "2"]
    monkeypatch.setattr("sys.argv", argv)
    qasper.main()
    first = out.read_text(encoding="utf-8")
    assert len(list(docs.glob("*.txt"))) == 2
    monkeypatch.setattr("sys.argv", argv)                   # không --force -> dừng, không ghi đè
    with pytest.raises(SystemExit):
        qasper.main()
    monkeypatch.setattr("sys.argv", argv + ["--force"])
    qasper.main()
    assert out.read_text(encoding="utf-8") == first