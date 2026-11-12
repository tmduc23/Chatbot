"""MỤC ĐÍCH: Test script chuyển bộ PeerQA -> tài liệu + dataset đánh giá."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import parse_case


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


peerqa = _load("import_peerqa")


# ---------- PeerQA ----------

def _row(idx, pidx, type_, content, heading=None, paper="egu/ESD/1"):
    return {"paper_id": paper, "idx": idx, "pidx": pidx, "sidx": 0, "type": type_, "content": content,
            "last_heading": heading}


PAPERS = [
    _row(0, 0, "title", "Glacier Melt Under Warming"),
    _row(1, 1, "heading", "Abstract"),
    _row(2, 2, "sentence", "We model glacier melt in the Alps from 1990 to 2020.", "Abstract"),
    _row(3, 2, "sentence", "Melt doubled over three decades in our simulations.", "Abstract"),
    _row(4, 3, "heading", "Methods"),
    _row(5, 4, "sentence", "We use a degree-day model calibrated on 12 stake sites.", "Methods"),
    _row(6, 4, "sentence", "Calibration used daily air temperature records.", "Methods"),
    _row(7, 5, "figure", "Figure 1: Map of the twelve stake sites in the Alps."),
]
METHODS_PARAGRAPH = "We use a degree-day model calibrated on 12 stake sites. Calibration used daily air temperature records."
QA = [
    {"paper_id": "egu/ESD/1", "question_id": "r1", "question": "How was the model calibrated?",
     "answer_evidence_mapped": [{"sentence": "x", "idx": [5, None]}], "answer_evidence_sent": [],
     "answer_free_form": "On 12 stake sites.", "answerable": True},
    {"paper_id": "egu/ESD/1", "question_id": "r2", "question": "Did you validate against satellite data?",
     "answer_evidence_mapped": None, "answer_free_form": "", "answerable": False},
    {"paper_id": "egu/ESD/1", "question_id": "r3", "question": "Why the Alps?",
     "answer_evidence_mapped": None, "answer_free_form": "", "answerable": None},
    {"paper_id": "openreview/ICLR-2022-conf/9", "question_id": "r4", "question": "Missing paper?",
     "answer_evidence_mapped": None, "answerable": False},
]


def test_peerqa_paper_text_keeps_paragraphs_and_sections() -> None:
    paper, sentence_of = peerqa.to_paper("egu/ESD/1", PAPERS)
    body = paper.render()
    assert body.startswith("Glacier Melt Under Warming\n\n## Abstract\n\n")
    assert ("We model glacier melt in the Alps from 1990 to 2020. Melt doubled over three decades in our simulations."
            in body)                                       # 2 câu cùng pidx -> 1 đoạn
    assert "## Methods" in body and "## Figures and tables" in body
    assert sentence_of[5] == sentence_of[6] == METHODS_PARAGRAPH        # idx -> NGUYÊN đoạn chứa nó
    assert sentence_of[7] == "Figure 1: Map of the twelve stake sites in the Alps."   # chú thích -> chính nó


def test_peerqa_cases_follow_the_rules() -> None:
    documents, cases, stats = peerqa.build(PAPERS, QA, 5, 10, 0.5, 1)
    assert stats["papers"] == 1 and stats["papers_available"] == 1   # paper OpenReview chưa tải -> không dùng
    by_src = {c["source_id"]: c for c in cases}
    r1 = by_src["r1"]
    assert r1["evidence_paragraphs"] == [METHODS_PARAGRAPH] and "expected_texts" not in r1
    assert r1["reference_answers"] == ["On 12 stake sites."] and "answer_contains" not in r1
    assert r1["question"].startswith('In the paper "Glacier Melt Under Warming": ')
    assert "egu" in r1["tags"]
    assert by_src["r2"]["answerable"] is False
    assert "r3" not in by_src and stats["skipped_unlabeled"] == 1
    for i, raw in enumerate(cases):
        parse_case(raw, i)


def test_peerqa_main_reads_data_folder(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "papers.jsonl").write_text("\n".join(json.dumps(r) for r in PAPERS), encoding="utf-8")
    (data / "qa.jsonl").write_text("\n".join(json.dumps(q) for q in QA), encoding="utf-8")
    out = tmp_path / "dataset.json"
    monkeypatch.setattr("sys.argv", ["import_peerqa.py", "--data", str(data), "--docs-out", str(tmp_path / "docs"),
                                     "--out", str(out)])
    peerqa.main()
    assert {c["source_id"] for c in json.loads(out.read_text(encoding="utf-8"))} == {"r1", "r2"}
    monkeypatch.setattr("sys.argv", ["import_peerqa.py", "--data", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        peerqa.main()


def test_peerqa_evidence_paragraphs_from_idx_or_sentence_text() -> None:
    _, paragraph_of = peerqa.to_paper("egu/ESD/1", PAPERS)
    # 2 câu cùng 1 đoạn -> đoạn chỉ xuất hiện 1 lần; idx None / không có trong bài -> bỏ qua
    qa = {"answer_evidence_mapped": [{"idx": [5]}, {"idx": [6, None, 999]}, {"idx": [7]}]}
    assert peerqa.evidence_paragraphs(qa, paragraph_of) == [
        METHODS_PARAGRAPH, "Figure 1: Map of the twelve stake sites in the Alps."]
    # không có idx dùng được: mặc định bỏ (như bài báo PeerQA) ...
    qa = {"answer_evidence_mapped": [{"idx": [None]}],
          "answer_evidence_sent": ["Melt doubled over three decades", "Not in the paper at all, sorry.", "Short"]}
    assert peerqa.evidence_paragraphs(qa, paragraph_of) == []
    # ... --text-fallback: tìm đoạn chứa nguyên văn câu bằng chứng
    assert peerqa.evidence_paragraphs(qa, paragraph_of, text_fallback=True) == [
        "We model glacier melt in the Alps from 1990 to 2020. Melt doubled over three decades in our simulations."]


def test_peerqa_question_without_usable_evidence_is_skipped() -> None:
    qa = [{"paper_id": "egu/ESD/1", "question_id": "r9", "question": "Q?", "answerable": True,
           "answer_evidence_mapped": [{"idx": [None]}], "answer_evidence_sent": ["Nothing like this in the paper."]}]
    _, cases, stats = peerqa.build(PAPERS, qa, 5, 10, 0.0, 1)
    assert cases == [] and stats["skipped_no_evidence"] == 1


def test_peerqa_evidence_paragraph_matches_chunks_like_qasper() -> None:
    from app.evaluation.dataset import chunk_matches
    documents, cases, _ = peerqa.build(PAPERS, QA, 5, 10, 0.5, 1)
    case = parse_case(next(c for c in cases if c["source_id"] == "r1"), 0)
    filename = next(iter(documents))
    assert chunk_matches(case, "## Methods\n\n" + METHODS_PARAGRAPH, filename)
    assert not chunk_matches(case, "We model glacier melt in the Alps from 1990 to 2020.", filename)
    assert not chunk_matches(case, METHODS_PARAGRAPH, "99-other-paper.txt")


def test_peerqa_text_fallback_is_off_by_default_and_counted_when_on() -> None:
    q2 = {"paper_id": "egu/ESD/1", "question_id": "r5", "question": "How much did melt increase?",
          "answerable": True, "answer_evidence_mapped": [{"idx": [None]}],
          "answer_evidence_sent": ["Melt doubled over three decades"]}
    _, cases, stats = peerqa.build(PAPERS, QA + [q2], 5, 10, 0.0, 1)
    assert {c["source_id"] for c in cases} == {"r1"}                  # chỉ câu có idx gốc
    assert stats["skipped_no_evidence"] == 1 and stats["rescued_by_text"] == 0
    _, cases, stats = peerqa.build(PAPERS, QA + [q2], 5, 10, 0.0, 1, text_fallback=True)
    assert {c["source_id"] for c in cases} == {"r1", "r5"}
    assert stats["skipped_no_evidence"] == 0 and stats["rescued_by_text"] == 1


def test_peerqa_captions_grouped_by_pidx_like_qasper_and_official_peerqa() -> None:
    # Code gốc PeerQA: mỗi hình/bảng 1 pidx, chú thích dài tách thành nhiều dòng (sidx 0, 1...) cùng pidx.
    rows = PAPERS + [
        _row(8, 6, "table", "Table 1: Melt rate per decade."),
        _row(9, 6, "table", "Values are in metres of water equivalent."),
    ]
    paper, paragraph_of = peerqa.to_paper("egu/ESD/1", rows)
    table = "Table 1: Melt rate per decade. Values are in metres of water equivalent."
    assert paragraph_of[8] == paragraph_of[9] == table                 # bằng chứng = NGUYÊN chú thích
    figures = dict(paper.sections)["Figures and tables"]
    assert figures == ["Figure 1: Map of the twelve stake sites in the Alps.", table]   # mỗi chú thích 1 đoạn
    assert "in the Alps.\n\nTable 1:" in paper.render()                  # cách nhau dòng trống, như QASPER



F1000 = "nlpeer/F1000-22/10-73"
F1000_ROWS = [
    _row(0, 0, "title", "Rnfuzzyapp: An R Shiny App [Version 2; Peer Review: 3 Approved]", paper=F1000),
    _row(1, 1, "heading", "Abstract", paper=F1000),
    _row(2, 2, "sentence", "Any further responses from the reviewers can be found at the end of the article.",
         "Abstract", paper=F1000),
    _row(3, 3, "sentence", "PCA is run on log1p transformed counts in the app.", "Abstract", paper=F1000),
    _row(4, 4, "sentence", "Reviewer Report 28 February 2022 https://doi.org/10.5256/x", paper=F1000),
    _row(5, 5, "sentence", "2. Is the PCA plot calculated on the whole count matrix?", paper=F1000),
    _row(6, 6, "sentence", "Our response: PCA of raw read counts uses all the genes.", paper=F1000),
    _row(7, 7, "figure", "Figure 1: The RNfuzzyApp workflow.", paper=F1000),
]


def test_peerqa_strip_peer_review_cuts_f1000_from_first_review_row() -> None:
    kept, cut = peerqa.strip_peer_review(F1000_ROWS)
    assert [r["idx"] for r in kept] == [0, 1, 2, 3, 7]        # câu "responses ... end of the article" vẫn giữ
    assert cut["cut_idx"] == 4 and cut["dropped"] == 3 and cut["leftover"] == []
    assert peerqa.strip_peer_review(PAPERS) == (sorted(PAPERS, key=lambda r: r["idx"]), None)   # không phải F1000


def test_peerqa_build_drops_review_text_and_keep_review_restores_it() -> None:
    qa = [{"paper_id": F1000, "question_id": "f1", "question": "Is the PCA on the whole matrix?",
           "answer_evidence_mapped": None, "answerable": False}]
    documents, _, stats = peerqa.build(F1000_ROWS, qa, 5, 5, 1.0, 1)
    text = next(iter(documents.values()))
    assert "Our response" not in text and "Reviewer Report" not in text and "log1p" in text
    assert "The RNfuzzyApp workflow" in text
    assert stats["review_cuts"][F1000]["cut_idx"] == 4
    documents, _, stats = peerqa.build(F1000_ROWS, qa, 5, 5, 1.0, 1, keep_review=True)
    assert "Our response" in next(iter(documents.values())) and stats["review_cuts"] == {}