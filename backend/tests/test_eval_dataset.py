"""MỤC ĐÍCH: Test đọc/kiểm tra eval/dataset.json và cách so khớp chunk đúng (Phase 11)."""
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    DatasetError, EvalCase, answer_has_facts, chunk_matches, load_dataset, normalize, stratified_sample,
)

GOOD = [
    {"id": "a01", "question": "Bạn ấy học ở đâu?", "answerable": True,
     "expected_texts": ["Bách khoa Hà Nội"], "tags": ["vi"]},
    {"id": "u01", "question": "Lương mong muốn?", "answerable": False},
]


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_normalize_unicode_case_and_spaces() -> None:
    decomposed = unicodedata.normalize("NFD", "Bách Khoa")    # "a" + dấu rời, như PDF hay xuất ra
    assert decomposed != "Bách Khoa"
    assert normalize(decomposed) == normalize("bách   khoa") == "bách khoa"
    assert normalize("Bách\nkhoa") == "bách khoa"


def test_chunk_matches_text_and_document() -> None:
    case = EvalCase(id="a", question="q", answerable=True, expected_texts=["TOEIC 875"],
                    expected_documents=["CV.pdf"])
    assert chunk_matches(case, "Ngoại ngữ: toeic   875", "CV.pdf")
    assert not chunk_matches(case, "Ngoại ngữ: TOEIC 875", "Other.pdf")    # đúng chữ, sai file
    assert not chunk_matches(case, "IELTS 7.0", "CV.pdf")


def test_any_expected_text_is_enough() -> None:
    case = EvalCase(id="a", question="q", answerable=True, expected_texts=["HUST", "Bách khoa"])
    assert chunk_matches(case, "Đại học Bách khoa Hà Nội", "cv.pdf")


def test_answer_has_facts() -> None:
    base = dict(id="a", question="q", answerable=True, expected_texts=["x"])
    assert answer_has_facts(EvalCase(**base), "gì cũng được") is None          # không khai báo
    case = EvalCase(**base, answer_contains=["875", "TOEIC"])
    assert answer_has_facts(case, "Bạn ấy có toeic 875 [Source 1].") is True
    assert answer_has_facts(case, "Bạn ấy có TOEIC.") is False                  # thiếu "875"


def test_load_good_dataset(tmp_path: Path) -> None:
    cases = load_dataset(_write(tmp_path, GOOD))
    assert [c.id for c in cases] == ["a01", "u01"]
    assert cases[0].expected_texts == ["Bách khoa Hà Nội"] and cases[0].tags == ["vi"]
    assert cases[1].answerable is False and cases[1].expected_texts == []


@pytest.mark.parametrize("bad, message", [
    ([{**GOOD[0], "id": "u01"}, GOOD[1]], "trùng"),
    ([{**GOOD[0], "expected_texts": []}], "ít nhất 1"),
    ([{**GOOD[1], "expected_texts": ["x"]}], "không được có"),
    ([{**GOOD[0], "answerable": "yes"}], "true hoặc false"),
    ([{**GOOD[0], "question": " "}], "question"),
    ([{**GOOD[0], "expected_texts": "Bách khoa"}], "danh sách"),
    ([], "không rỗng"),
])
def test_bad_dataset_is_rejected_with_clear_message(tmp_path: Path, bad: object, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write(tmp_path, bad))


def test_broken_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('[\n  {"id": "a01",}\n]', encoding="utf-8")
    with pytest.raises(DatasetError, match="dòng 2"):
        load_dataset(path)


# Chuyển sang định dạng mới
def test_build_eval_dataset_converts_old_files() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_eval_dataset.py"
    spec = importlib.util.spec_from_file_location("build_eval_dataset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = module.convert(
        [{"query": "Bạn ấy học ở đâu?", "expected_text": "HUST"}, {"query": "GPA?", "expected_text": "3.35"}],
        [{"query": "Favorite food?"}],
    )
    assert [c["id"] for c in cases] == ["a01", "a02", "u01"]
    assert cases[0]["tags"] == ["vi"] and cases[2]["tags"] == ["en"]
    assert cases[2]["answerable"] is False and "expected_texts" not in cases[2]

# test lấy mẫu
def test_stratified_sample_keeps_unanswerable_ratio() -> None:
    cases = [EvalCase(id=f"a{i}", question="q", answerable=True, expected_texts=["x"]) for i in range(70)]
    cases += [EvalCase(id=f"u{i}", question="q", answerable=False) for i in range(30)]
    picked = stratified_sample(cases, 40, seed=1)
    assert len(picked) == 40
    assert sum(not c.answerable for c in picked) == 12               # 30% như cả bộ, không phải 0 như "40 câu đầu"
    assert picked == stratified_sample(cases, 40, seed=1)            # cùng seed -> cùng bộ câu
    assert [c.id for c in picked] == [c.id for c in cases if c in picked]   # giữ thứ tự file
    assert stratified_sample(cases, 500) == cases

# reference_answers được đọc đúng, câu bẫy có reference_answers thì bị từ chối
def test_reference_answers_are_loaded_and_only_for_answerable(tmp_path: Path) -> None:
    good = [{**GOOD[0], "reference_answers": ["Bách khoa Hà Nội", "HUST"]}, GOOD[1]]
    assert load_dataset(_write(tmp_path, good))[0].reference_answers == ["Bách khoa Hà Nội", "HUST"]
    with pytest.raises(DatasetError, match="reference_answers"):
        load_dataset(_write(tmp_path, [GOOD[0], {**GOOD[1], "reference_answers": ["x"]}]))