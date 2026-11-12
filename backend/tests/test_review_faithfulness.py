"""MỤC ĐÍCH: Test script chấm tay giám khảo faithfulness (lấy mẫu, tính tỉ lệ đồng ý, chấm mù)."""
import importlib.util
import json
from pathlib import Path

_script = Path(__file__).resolve().parents[1] / "scripts" / "review_faithfulness.py"
_spec = importlib.util.spec_from_file_location("review_faithfulness", _script)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)

REPORT = {
    "kind": "generation",
    "cases": [
        {"id": "a1", "question": "q1", "answer": "A, B", "context": "ctx1",
         "faithfulness": {"score": 0.5, "claims": [{"text": "A", "supported": True},
                                                   {"text": "B", "supported": False},
                                                   {"text": "C", "supported": None}]}},
        {"id": "a2", "question": "q2", "answer": "từ chối", "abstained": True},
        {"id": "a3", "question": "q3", "answer": "D", "context": "ctx3",
         "faithfulness": {"score": 1.0, "claims": [{"text": "D", "supported": True}]}},
    ],
}


def test_collect_only_claims_the_judge_scored() -> None:
    items = review.collect_claims(REPORT)
    assert [(i["id"], i["claim"], i["judge"]) for i in items] == [("a1", "A", True), ("a1", "B", False), ("a3", "D", True)]
    assert items[0]["context"] == "ctx1"


def test_sample_is_reproducible_and_capped() -> None:
    items = review.collect_claims(REPORT)
    assert review.sample_claims(items, 2, seed=1) == review.sample_claims(items, 2, seed=1)
    assert len(review.sample_claims(items, 50, seed=1)) == 3


def test_agreement_splits_lenient_and_strict_errors() -> None:
    result = review.agreement([
        {"judge": True, "human": True},
        {"judge": True, "human": False},     # giám khảo dễ dãi
        {"judge": False, "human": True},     # giám khảo khó tính
        {"judge": False, "human": False},
    ])
    assert result == {"n": 4, "agree": 2, "rate": 0.5, "judge_yes_human_no": 1, "judge_no_human_yes": 1}
    assert review.agreement([])["rate"] is None


def test_main_asks_blind_then_saves(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "run-generation.json"
    path.write_text(json.dumps(REPORT, ensure_ascii=False), encoding="utf-8")
    answers = iter(["x", "y", "n", "y"])                  # "x" sai cú pháp -> hỏi lại
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.argv", ["x", str(path), "--sample", "3", "--seed", "1"])
    review.main()
    saved = json.loads((tmp_path / "run-generation.review.json").read_text(encoding="utf-8"))
    assert saved["n"] == 3 and len(saved["items"]) == 3
    assert {"claim", "judge", "human"} <= set(saved["items"][0])
    assert "gõ y (có)" in capsys.readouterr().out