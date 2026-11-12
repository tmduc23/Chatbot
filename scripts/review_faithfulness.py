"""
MỤC ĐÍCH: Tự chấm lại giám khảo faithfulness -> biết giám khảo LLM đúng bao nhiêu % (spec mục 23).

    python scripts/review_faithfulness.py eval/results/<file>-generation-<label>.json --sample 20

Script lấy ngẫu nhiên (có seed) N ý mà giám khảo đã chấm, cho bạn xem context + ý, rồi hỏi
"context có chứng minh ý này không?". Bạn trả lời TRƯỚC, sau đó mới thấy giám khảo chấm gì
(chấm "mù" để không bị giám khảo dắt). Cuối cùng in tỉ lệ đồng ý và lưu <file>.review.json.

Một dòng trong README như "giám khảo đồng ý với người chấm 18/20 = 90%" làm con số faithfulness đáng tin hơn hẳn.
"""
import argparse
import json
import random
import sys
from pathlib import Path


def collect_claims(report: dict) -> list[dict]:
    """Mọi ý giám khảo ĐÃ chấm (YES/NO) trong 1 file kết quả generation."""
    items = []
    for row in report.get("cases", []):
        faith = row.get("faithfulness")
        if not faith:
            continue
        for i, claim in enumerate(faith["claims"]):
            if claim["supported"] is not None:
                items.append({"id": row["id"], "index": i, "question": row["question"], "answer": row["answer"],
                              "context": row.get("context", ""), "claim": claim["text"], "judge": claim["supported"]})
    return items


def sample_claims(items: list[dict], n: int, seed: int) -> list[dict]:
    return random.Random(seed).sample(items, min(n, len(items)))


def agreement(reviewed: list[dict]) -> dict:
    """reviewed: các ý có cả 'judge' và 'human' (True/False)."""
    n = len(reviewed)
    agree = sum(r["judge"] == r["human"] for r in reviewed)
    return {
        "n": n,
        "agree": agree,
        "rate": agree / n if n else None,
        # giám khảo dễ dãi: cho YES nhưng người chấm thấy context KHÔNG có -> faithfulness bị thổi phồng
        "judge_yes_human_no": sum(r["judge"] and not r["human"] for r in reviewed),
        # giám khảo khó tính: cho NO nhưng context CÓ -> faithfulness bị đánh thấp
        "judge_no_human_yes": sum(not r["judge"] and r["human"] for r in reviewed),
    }


def ask(prompt: str) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "n", "s", "q"):
            return answer
        print("  gõ y (có), n (không), s (bỏ qua) hoặc q (dừng)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually check the faithfulness judge on a random sample")
    parser.add_argument("result", type=Path, help="File kết quả của evaluate_generation.py --faithfulness")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = json.loads(args.result.read_text(encoding="utf-8"))
    items = collect_claims(report)
    if not items:
        sys.exit("File này không có ý nào được chấm. Chạy evaluate_generation.py với --faithfulness trước.")
    picked = sample_claims(items, args.sample, args.seed)
    print(f"{len(items)} ý đã được giám khảo chấm, bạn sẽ chấm lại {len(picked)} ý.\n")

    reviewed = []
    for i, item in enumerate(picked, start=1):
        print("=" * 80)
        print(f"[{i}/{len(picked)}] {item['id']} | câu hỏi: {item['question']}")
        print(f"\nCONTEXT:\n{item['context']}\n")
        print(f"Ý CẦN CHẤM: {item['claim']}")
        answer = ask("Context có chứng minh ý này không? [y/n/s/q] ")
        if answer == "q":
            break
        if answer == "s":
            continue
        human = answer == "y"
        print(f"  -> giám khảo chấm {'YES' if item['judge'] else 'NO'}: "
              f"{'ĐỒNG Ý' if human == item['judge'] else 'KHÁC bạn'}\n")
        reviewed.append({**item, "human": human})

    result = agreement(reviewed)
    if result["n"]:
        print(f"\nĐồng ý {result['agree']}/{result['n']} = {result['rate'] * 100:.0f}% | "
              f"giám khảo dễ dãi {result['judge_yes_human_no']} | khó tính {result['judge_no_human_yes']}")
    out = args.result.with_suffix(".review.json")
    out.write_text(json.dumps({"source": args.result.name, "seed": args.seed, **result,
                               "items": [{k: r[k] for k in ("id", "index", "claim", "judge", "human")} for r in reviewed]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()