"""
MỤC ĐÍCH: Test cách chấm "chunk đúng" theo ĐOẠN VĂN (evidence_paragraphs), giống bài báo QASPER/PeerQA.

Chunk đúng = phần lớn chunk nằm trong 1 đoạn bằng chứng, HOẶC chunk chứa phần lớn đoạn đó
(so theo cụm 5 từ liên tiếp, ngưỡng PARAGRAPH_MATCH = 0.5).
"""
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import DatasetError, EvalCase, chunk_matches, load_dataset, paragraph_overlap

PARA = " ".join(f"Sentence {i} of the evidence paragraph talks about glacier melt rates." for i in range(10))
OTHER = " ".join(f"Line {i} of an unrelated paragraph describes the hardware setup." for i in range(10))

def _case(**kw) -> EvalCase:
    return EvalCase(id="a", question="q", answerable=True, **kw)

def test_piece_of_a_long_paragraph_matches() -> None:
    words = PARA.split()
    first_third = "Paper > Results\n" + " ".join(words[: len(words) // 3])     # đoạn dài bị cắt làm 3 chunk
    assert paragraph_overlap(first_third, PARA) > 0.9
    assert chunk_matches(_case(evidence_paragraphs=[PARA]), first_third, "p.txt")


def test_chunk_with_only_a_small_tail_of_the_paragraph_does_not_match() -> None:
    tail = " ".join(PARA.split()[-12:])                                      # chunk vắt qua: 12 từ cuối của đoạn
    chunk = f"{OTHER}\n\n{tail}"
    assert paragraph_overlap(chunk, PARA) < 0.5
    assert not chunk_matches(_case(evidence_paragraphs=[PARA]), chunk, "p.txt")
    assert not chunk_matches(_case(evidence_paragraphs=[PARA]), OTHER, "p.txt")

def test_very_short_paragraph_like_a_table_caption() -> None:
    case = _case(evidence_paragraphs=["Table 1: Data Statistics."])
    assert chunk_matches(case, "## Figures and tables\n\ntable 1: data statistics.", "p.txt")
    assert not chunk_matches(case, "Table 2: Results.", "p.txt")


def test_paragraphs_take_priority_over_expected_texts() -> None:
    sentence = PARA.split(". ")[-1]                                          # câu cuối của đoạn
    chunk = f"{OTHER}\n\n{sentence}"                                          # chứa câu, nhưng gần như toàn đoạn khác
    assert chunk_matches(_case(expected_texts=[sentence]), chunk, "p.txt")                       # cách cũ: đúng
    assert not chunk_matches(_case(expected_texts=[sentence], evidence_paragraphs=[PARA]), chunk, "p.txt")
    assert not chunk_matches(_case(evidence_paragraphs=[PARA], expected_documents=["a.txt"]), PARA, "b.txt")

def test_dataset_accepts_paragraphs_only_but_not_on_traps(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps([{"id": "a1", "question": "q", "answerable": True, "evidence_paragraphs": [PARA]}]))
    assert load_dataset(path)[0].evidence_paragraphs == [PARA]
    path.write_text(json.dumps([{"id": "u1", "question": "q", "answerable": False, "evidence_paragraphs": [PARA]}]))
    with pytest.raises(DatasetError, match="không được có"):
        load_dataset(path)