"""
MỤC ĐÍCH: Đo chất lluojwng TRÍCH DẪN và (tùy chọn) grounding trên bộ câu có đáp án (Phase 7)

    docker compose exec backend python scripts/eval_citations.py --app cv --limit 5
    docker compose exec -e GROUNDING_CHECK=true backend python scripts/eval_citations.py --app cv --limit 5

Dùng lại eval/retrieval_smoke.json: [{"query": "...", "expected_text": "đoạn chữ có trong chunk đúng"}]

Chỉ số (không cần LLM chấm, tính bằng code):
    - Trả lời:                  câu KHÔNG bị từ chối (câu có đáp án mà bị từ chối = từ chối nhầm)
    - Có trích nguồn:           câu trả lời có ít nhất 1 [Source n] hợp lệ
    - Trích đúng:               ít nhất 1 nguồn đuộc trích CHỨA expected_text (trích đúng chunk có đáp án)
    - Trích bịa:                số [Souce n] không tồn tại mà LLM viết ra (đã bị xóa)

Chỉ số đo LLM chấm (khi GROUNDING_CHECK=true): số câu SUPPORTED / PARTIAL / UNSUPPORTED / UNKNOWN

Phương pháp LLM-as-judge (spec mục 23 yêu cầu ghi rõ, và kiểm tra bias/consistency của judge):
  judge = GROUNDING_MODEL (mặc định LLM_MODEL), temperature 0, prompt grounding-v1, chỉ thấy SOURCES + ANSWER.
  - bias: judge cùng họ với model trả lời có thể dễ dãi -> script lưu mọi câu trả lời ra eval/results/
    để bạn TỰ ĐỌC và ghi lại số câu bạn đồng ý với judge;
  - consistency: chạy lại cùng --limit lần 2 và so verdict của 2 file kết quả.

Quota: mỗi câu = 1 lần gọi reranker + 1 lần LLM (+1 nếu bật grounding). Free tier gemini-3.8-flash
20 request/ngày -> dùng --limit.
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.rag_service import RAGAnswer, build_rag_service


def cites_expected(result: RAGAnswer, expect_text: str) -> bool:
    """Có nguồn được trích nào chứa expected_text không (so không phân biệt hoa thường)."""
    texts = {r.chunk.chunk_id: r.chunk.text for r in result.retrieval.relevant} # chunk_id:text (các chunk rerank)
    needle = expect_text.lower()
    return any(needle in texts.get(s.chunk_id, "").lower() for s in result.sources)



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citations and grounding")
    parser.add_argument("--app", required=True)
    parser.add_argument("--eval", type=Path, default=Path("eval/retrieval_smoke.json"))
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chayyj N câu đầu (tiết kiệm quota)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)    # trích bịa đã có cột riêng


    settings = get_settings()
    cases = json.load(args.eval.read_text(encoding="utf-8"))[: args.limit]
    calls = len(cases) * (2 if settings.grounding_check else 1)
    print(f"\n{len(cases)} câu | grounding: {'BẬT' if settings.grounding_check else 'tắt'}"
          f"| sẽ gọi LLM tối đa {calls} lần\n")

    rows = []
    with SessionLocal() as db:
        service = build_rag_service(db, settings)
        print(f"{'kết quả':<30}{'nguồn':>6}{'đúng':>6}{'bịa':>5}  {'grounding':<12}câu hỏi")
        print("-" * 90)
        for case in cases:
            result = service.ask(case["query"], args.app)
            row = {
                "query": case["query"],
                "answer": result.answer,
                "abstain_reason": result.abstain_reason,
                "sources": [s.to_dict() for s in result.sources],    # chuyển các [{"filename": "cv.pdf", "page": 1}]
                "cites_expected": cites_expected(result, case["expected_text"]),
                "invalid_citations": result.invalid_citations,
                "grounding": result.grounding.verdict if result.grounding else None,
            }
            rows.append(row)
            outcome = f"từ chối ({result.abstain_reason})" if result.abstained else "trả lời"
            print(f"{outcome:<30}{len(result.sources):>6}{'✓' if row['cites_expected'] else '✗':>6}"
                  f"{len(result.invalid_citations):>5}  {row['grounding'] or '-':<12}{case['query'][:40]}")
      
    n = len(rows)
    answered = [r for r in rows if r["abstain_reason"] is None]
    print(f"\nTrả lời:        {len(answered)}/{n}   (từ chối nhầm {n - len(answered)})")
    print(f"Có trích nguồn: {sum(bool(r['sources']) for r in answered)}/{len(answered)}")
    print(f"Trích đúng:     {sum(r['cites_expected'] for r in answered)}/{len(answered)}")
    print(f"Trích bịa:      {sum(len(r['invalid_citations']) for r in rows)} lần")
    if settings.grounding_check:
        print(f"Grounding:      {dict(Counter(r['grounding'] for r in rows if r['grounding']))}")
    
    out = Path("eval/results") / f"citations_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu từng câu trả lời vào {out} -> đọc lại để kiểm tra judge chấm có đúng không.")


if __name__ == "__main__":
    main()