"""
MỤC ĐíCH: In chi tiết giám khảo faithfulness đã chấm từng ý (YES/NO) trong 1 file kết quả.

    python scripts/show_judge.py eval/results/<file>.json               # mọi câu đã chấm
    python scripts/show_judge.py eval/results/<file>.json --only-no     # chỉ câu ý bị chấm NO / bỏ sót
    python scripts/show_judge.py eval/results/<file>.json --id qa023    # 1 câu, kèm câu hỏi + câu trả lời

Chỉ đọc file, không gọi API.
"""
import argparse
import json
import sys
from pathlib import Path

MARK = {True: "YES", False: "NO", None: "?? "}  # ?? = giám khảo bỏ sót ý


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases") or data.get("metrics", {}).get("cases", [])



def main() -> None:
    parser = argparse.ArgumentParser(description="Xem giám khảo faithfulness chấm từng ý")
    parser.add_argument("result", type=Path, help="file JSON của evaluate_generation.py --faithfulness")
    parser.add_argument("--only-no", action="store_true", help="chỉ in câu có ý NO hoặc bỏ sót")
    parser.add_argument("--id", help="chỉ in 1 câu (vd. qa023), kèm câu hỏi và câu trả lời")
    args = parser.parse_args()


    judged = [c for c in load_cases(args.result) if c.get("faithfulness")]
    if not judged:
        sys.exit("File không có kết quả giám khảo (chạy evaluate_generation.py với --faithfulness).")

    shown = n_yes = n_no = n_miss = 0
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
        shown += 1
        score = "-" if f["score"] is None else f"{['score']:.2f}"
        print(f"\n== {case['id']} score={score} {f.get('error') or ''}")
        if args.id:
            print(f"Q: {case.get('question', '')}\nA: {case.get('answer', '')}\n")
        for cl in f["claims"]:
            print(f" {MARK[cl['supported']]} {cl['text']}")

    print(f"\n{shown}/{len(judged)} câu được in | tổng ý: YES={n_yes} NO={n_no} bỏ sót={n_miss}")


if __name__ == "__main__":
    main()