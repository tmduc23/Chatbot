"""MỤC ĐÍCH: Test script chuyển SQuAD 2.0 -> tài liệu + dataset đánh giá (Phase 11, dataset chuẩn)."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import parse_case

_script = Path(__file__).resolve().parents[1] / "scripts" / "import_squad.py"
_spec = importlib.util.spec_from_file_location("import_squad", _script)
squad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(squad)

CTX_1 = "The Normans were a people in France. They gave their name to Normandy in the 10th century."
CTX_2 = "Rollo was a Viking leader. He became the first ruler of Normandy in 911."
CTX_3 = "Photosynthesis happens in chloroplasts. It needs light, water and carbon dioxide."

NESTED = {"version": "v2.0", "data": [
    {"title": "Normans", "paragraphs": [
        {"context": CTX_1, "qas": [
            {"id": "q1", "question": "In what country is Normandy?", "is_impossible": False,
             "answers": [{"text": "France", "answer_start": CTX_1.index("France")}]},
            {"id": "q2", "question": "Who gave their name to Normandy in the 1000s?", "is_impossible": True,
             "answers": [], "plausible_answers": [{"text": "Normans", "answer_start": 4}]},
        ]},
        {"context": CTX_2, "qas": [
            {"id": "q3", "question": "When did Rollo start ruling?", "is_impossible": False,
             "answers": [{"text": "in 911", "answer_start": CTX_2.index("in 911")},
                         {"text": "911", "answer_start": CTX_2.index("911")}]},
        ]},
    ]},
    {"title": "Photosynthesis", "paragraphs": [
        {"context": CTX_3, "qas": [
            {"id": "q4", "question": "Where does photosynthesis happen?", "is_impossible": False,
             "answers": [{"text": "chloroplasts", "answer_start": CTX_3.index("chloroplasts")}]},
        ]},
    ]},
]}


def _build(articles, **kw):
    params = dict(n_articles=10, n_questions=100, unanswerable_ratio=0.5, max_paragraphs=30, seed=1)
    params.update(kw)
    return squad.build(articles, **params)


def test_parse_nested_and_flat_give_same_questions() -> None:
    flat = []
    for a in NESTED["data"]:
        for p in a["paragraphs"]:
            for q in p["qas"]:
                flat.append({"id": q["id"], "title": a["title"], "context": p["context"], "question": q["question"],
                             "answers": {"text": [x["text"] for x in q["answers"]],
                                         "answer_start": [x["answer_start"] for x in q["answers"]]}})

    def summary(articles):
        return sorted((a.title, q.id, tuple(q.answers)) for a in articles for p in a.paragraphs for q in p.questions)

    assert summary(squad.parse_squad(NESTED)) == summary(squad.parse_squad(flat))


def test_impossible_question_has_no_answers_even_with_plausible_answers() -> None:
    q2 = squad.parse_squad(NESTED)[0].paragraphs[0].questions[1]
    assert q2.id == "q2" and q2.answers == []          # plausible_answers là bẫy, không phải đáp án


def test_cases_are_valid_dataset_entries() -> None:
    documents, cases, stats = _build(squad.parse_squad(NESTED))
    assert len(documents) == 2
    assert stats["answerable"] == 3 and stats["unanswerable"] == 1
    for i, raw in enumerate(cases):
        parse_case(raw, i)                              # load_dataset sẽ không từ chối
    assert len({c["id"] for c in cases}) == len(cases)


def test_evidence_is_the_answer_sentence_in_the_right_document() -> None:
    documents, cases, _ = _build(squad.parse_squad(NESTED))
    q1 = next(c for c in cases if c["source_id"] == "q1")
    assert q1["expected_texts"] == ["The Normans were a people in France."]
    doc = q1["expected_documents"][0]
    assert q1["expected_texts"][0] in documents[doc]
    assert q1["answer_contains"] == ["France"]
    q3 = next(c for c in cases if c["source_id"] == "q3")
    assert q3["answer_contains"] == ["911"]             # lấy đáp án ngắn nhất -> chấm dễ hơn
    assert q3["reference_answers"] == ["in 911", "911"]  # giữ MỌI đáp án để chấm F1 như SQuAD


def test_answer_contains_ignores_answers_outside_the_evidence() -> None:
    # SQuAD thật (câu 573776eec3c5551400e51eec): 1 người chấm ghi "straight down", không có trong câu bằng chứng
    ctx = "A ball dropped from a mast would land behind the foot of the mast. Galileo said it falls straight down."
    data = {"data": [{"title": "Force", "paragraphs": [{"context": ctx, "qas": [
        {"id": "f1", "question": "Where does the ball land according to Aristotle?", "is_impossible": False,
         "answers": [{"text": "behind the foot of the mast", "answer_start": ctx.index("behind")},
                     {"text": "straight down", "answer_start": ctx.index("straight down")}]}]}]}]}
    _, cases, _ = _build(squad.parse_squad(data))
    assert cases[0]["answer_contains"] == ["behind the foot of the mast"]   # không phải "straight down"
    assert cases[0]["reference_answers"] == ["behind the foot of the mast", "straight down"]   # F1 vẫn giữ mọi đáp án

def test_long_sentence_is_cut_around_answer_at_word_boundaries() -> None:
    words = " ".join(f"word{i}" for i in range(80))
    context = f"{words} the treaty was signed in 1648 by many {words}."
    evidence = squad.evidence_for(context, context.index("1648"), "1648")
    assert "1648" in evidence
    assert len(evidence) <= squad.MAX_EVIDENCE_CHARS
    assert evidence.split()[0] in context.split() and evidence.split()[-1] in context.split()


def test_ambiguous_evidence_is_skipped() -> None:
    ctx = "Paris is big."
    data = {"data": [{"title": "T", "paragraphs": [
        {"context": ctx, "qas": [{"id": "x1", "question": "Is Paris big?", "answers": [{"text": "big", "answer_start": 9}]}]},
        {"context": ctx, "qas": []},                    # câu văn y hệt ở đoạn khác -> không biết chunk nào đúng
    ]}]}
    _, cases, stats = _build(squad.parse_squad(data))
    assert cases == [] and stats["skipped_ambiguous"] == 1


def test_same_seed_same_dataset_and_limits_respected() -> None:
    articles = squad.parse_squad(NESTED)
    a = _build(articles, n_articles=1, seed=7)
    b = _build(articles, n_articles=1, seed=7)
    assert a == b
    assert len(a[0]) == 1
    _, cases, _ = _build(articles, n_questions=2, unanswerable_ratio=0.5)
    assert sum(c["answerable"] for c in cases) == 1 and len(cases) == 2


def test_main_writes_files(tmp_path, monkeypatch) -> None:
    src = tmp_path / "dev.json"
    src.write_text(json.dumps(NESTED), encoding="utf-8")
    docs, out = tmp_path / "docs", tmp_path / "dataset.json"
    argv = ["import_squad.py", "--input", str(src), "--docs-out", str(docs), "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    squad.main()
    assert sorted(p.name for p in docs.glob("*.txt")) and len(json.loads(out.read_text(encoding="utf-8"))) == 4
    monkeypatch.setattr("sys.argv", argv)               # chạy lại mà không --force -> dừng, không ghi đè
    with pytest.raises(SystemExit):
        squad.main()


# ---------- UIT-ViQuAD 2.0 (bản tiếng Việt, cùng định dạng) ----------

VI_1 = "Hồ Hoàn Kiếm còn được gọi là hồ Gươm. Hồ nằm ở trung tâm Hà Nội."
VI_2 = "Đà Nẵng là thành phố biển. Cầu Rồng bắc qua sông Hàn."
VI_ROWS = [   # giống 1 dòng của file parquet trên HuggingFace
    {"id": "0001-0001-0001", "title": "Hồ_Hoàn_Kiếm", "context": VI_1, "question": "Hồ Hoàn Kiếm còn được gọi là gì?",
     "answers": {"text": ["hồ Gươm"], "answer_start": [VI_1.index("hồ Gươm")]}, "is_impossible": False},
    {"id": "0001-0001-0002", "title": "Hồ_Hoàn_Kiếm", "context": VI_1,
     "question": "Hồ ở trung tâm Hải Phòng còn được gọi là gì?",          # đổi thực thể -> không trả lời được
     "answers": {"text": [], "answer_start": []}, "is_impossible": True,
     "plausible_answers": {"text": ["hồ Gươm"], "answer_start": [VI_1.index("hồ Gươm")]}},
    {"id": "0002-0001-0001", "title": "Đà_Nẵng", "context": VI_2, "question": "Cầu Rồng bắc qua sông nào?",
     "answers": {"text": ["sông Hàn"], "answer_start": [0]}, "is_impossible": False},   # answer_start lệch
]


def test_slugify_keeps_vietnamese_d() -> None:
    assert squad.slugify("Đà_Nẵng") == "da-nang"           # không có bước đổi "đ" -> "a-nang"
    assert squad.slugify("Hồ_Hoàn_Kiếm") == "ho-hoan-kiem"


def test_wrong_answer_start_is_relocated() -> None:
    assert squad.locate_answer(VI_2, "sông Hàn", VI_2.index("sông Hàn")) == VI_2.index("sông Hàn")   # đúng sẵn
    assert squad.locate_answer(VI_2, "sông Hàn", 0) == VI_2.index("sông Hàn")                        # tìm lại
    assert squad.locate_answer("A và A", "A", 3) is None                  # xuất hiện 2 lần -> không đoán
    assert squad.locate_answer(VI_2, "sông Hương", 0) is None             # không có trong đoạn


def test_vietnamese_rows_build_dataset_with_source_tag() -> None:
    documents, cases, stats = _build(squad.parse_squad(VI_ROWS), language="vi", source="viquad2")
    assert stats["answerable"] == 2 and stats["unanswerable"] == 1 and stats["skipped_ambiguous"] == 0
    assert any(name.endswith("-da-nang.txt") for name in documents)
    by_src = {c["source_id"]: c for c in cases}
    assert by_src["0001-0001-0001"]["expected_texts"] == ["Hồ Hoàn Kiếm còn được gọi là hồ Gươm."]
    assert by_src["0002-0001-0001"]["expected_texts"] == ["Cầu Rồng bắc qua sông Hàn."]   # nhờ locate_answer
    trap = by_src["0001-0001-0002"]
    assert trap["answerable"] is False and "answer_contains" not in trap      # plausible_answers không phải đáp án
    assert all(c["tags"][:2] == ["vi", "viquad2"] for c in cases)


def test_read_parquet_same_as_flat_rows(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(VI_ROWS), path)

    def summary(articles):
        return sorted((a.title, q.id, tuple(q.answers)) for a in articles for p in a.paragraphs for q in p.questions)

    assert summary(squad.read_input(path)) == summary(squad.parse_squad(VI_ROWS))