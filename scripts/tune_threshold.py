"""
MỤC ĐÍCH: Chọn RERANKER_MIN_SCORE bằng SỐ ĐO thay vì đoán (không chọn tùy ý)

Ý tưởng: ngưỡng tốt phải:
    - GIỮ chunnk đúng của câu hỏi CÓ đáp án         (eval/retrieval_smoke.json)
    - CHẶN mọi chunk của câu hỏi không có đáp án    (eval/unanswerable.json)

    câu có đáp án: điểm rerank của chunk ĐÚNG (chunk chuuawx expected_text)
    câu không có đáp án: điểm rerank CAO NHẤT (chỉ 1 chunk lọt ngưỡng là LLML đã bị gọi)

Nếu mọi điểm "không đáp án" < mọi điểm "có đáp án" -> tách được hoàn toàn, chọn ngưỡng ở giữa.
Nếu không tách được (nghĩa là trùng ) -> chọn ngưỡng vẫn giữ 100% câu có đáp án; câu lọt qua sẽ do LLM tự từ chối 
(INSUFFICIENT_CONTEXT, lớp bảo vệ thứ 2). Thêm --llm để đo cả lớp đó.

Chạy cho TỪNG reranker vì thang điểm khác nhau:
    docker compose exec -e RERANKER_PROVIDER=gemini backend python scripts/tune_threshold.py --app cv
    docker compose exec -e RERANKER_PROVIDER=cohere backend python scripts/tune_threshold.py --app cv
    ... --llm      # chạy thêm RAGService đầy đủ (tốn thêm 1 lần gọi LLM mỗi câu)
    ... --sleep 7  # nghỉ 7 giây giữa các câu (Cohere trial key chỉ cho 10 request/phút)


Mỗi câu hỏi tốn 1 lần gọi reranker (không gọi LLM nếu không có --llm).


loại         điểm  câu hỏi
--------------------------------------------------------------------------------
có           0.85  Bạn có kinh nghiệm Docker không?
có           0.72  Bạn học trường đại học nào?
có              -  Bạn biết Kubernetes không? (lỗi reranker / chunk đúng không có trong top-k)
không        0.25  What is the expected salary?
không        0.10  Tình trạng hôn nhân?
không           -  Bạn có bằng lái xe không?


# giúp nhìn để chọn ngưỡng thay vì con số gợi ý
ngưỡng     giữ câu có đáp án     chặn câu không đáp án
0.00                10/10                      0/6
0.10                10/10                      1/6
0.30                10/10                      5/6
0.35                10/10                      6/6
...
0.70                10/10                      6/6
0.75                 8/10                      6/6
0.95                 1/10                      6/6


phase 2 
[("Bạn đã làm dự án RAG chatbot nào?", True),
 ("Bạn có kinh nghiệm với Docker không?", True),
 ...
 ("Mức lương mong muốn của bạn là bao nhiêu?", False),
 ("Bạn đã kết hôn chưa?", False)]

RAGService đầy đủ, ngưỡng 0.45:
loại      kết quả                         câu hỏi
có        trả lời                         Bạn đã làm dự án RAG chatbot nào?
có        trả lời                         Bạn có kinh nghiệm với Docker không?
có        từ chối (no_relevant_chunks)    Bạn học trường đại học nào?
có        trả lời                         Bạn biết những ngôn ngữ lập trình nào?
có        trả lời                         Kể về kinh nghiệm làm việc với FastAPI và PostgreS
không     từ chối (no_relevant_chunks)    Mức lương mong muốn của bạn là bao nhiêu?
không     từ chối (llm_insufficient)      Bạn đã kết hôn chưa?
không     trả lời                         Bạn có bằng lái xe không?

Từ chối đúng (câu không có đáp án): 2/3
Từ chối nhầm (câu có đáp án):   1/5
"""

import argparse
import json
import logging
from pathlib import Path
import time

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.rag.reranking import get_min_score
from app.rag.retrieval_pipeline import RetrievalResult, build_retrieval_pipeline
from app.services.rag_service import build_rag_service

THRESHOLDS = [round(i * 0.05, 2) for i in range(20)]            # 0.00, 0.05, ..., 0.95


def judged_score(result: RetrievalResult, expected_text: str | None = None) -> float | None:
    """
    expected_text có -> điểm của chunk ĐÚNG đầu tiên (None nếu chunk đúng không có trong top-k)
    expected_text None -> điểm cao nhất
    None nếu reranker lỗi (fail-open, không có điểm nào để so)
    """
    judged = [r for r in result.reranked if r.judged]
    if not judged:
        return None
    if expected_text is None:
        return max(r.score for r in judged)
    needle = expected_text.lower()
    return next((r.score for r in judged if needle in r.chunk.text.lower()), None)  # lấy chunk có score đúng đầu tiên


def suggest_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, bool]:
    """Trả (ngưỡng gợi ý, tách được hoàn toàn hay không)."""
    lo = min(answerable)                            # chunk đúng có điểm THẤP nhất
    hi = max(unanswerable) if unanswerable else 0.0 # câu không đáp án có điểm CAO nhất
    if hi < lo:
        return round((lo + hi) / 2, 2), True
    return max(t for t in THRESHOLDS if t <= lo), False         # 2 lớp chồng lên nhau thì chọn cái max <= lo

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RERANKER_MIN_SCORE on an eval test")
    parser.add_argument("--app", required=True)
    parser.add_argument("--answerable", type=Path, default=Path("eval/retrieval_smoke.json"))
    parser.add_argument("--unanswerable", type=Path, default=Path("eval/unanswerable.json"))
    parser.add_argument("--llm", action="store_true", help="Chạy thêm RAGService đầy đủ với ngưỡng hiện tại")
    parser.add_argument("--sleep", type=float, default=0.0, help="Nghỉ giữa các câu, giây (Cohere trial: 7)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    answerable = json.loads(args.answerable.read_text(encoding="utf-8"))
    unanswerable = json.loads(args.unanswerable.read_text(encoding="utf-8"))
    print(f"\nReranker: {settings.reranker_provider} | ngưỡng hiện tại: {get_min_score(settings):.2f}\n")

    ans_scores: list[float] = []
    unans_scores: list[float] = []
    with SessionLocal() as db:

        # phần 1: mặc định không có LM -> chỉ hỏi "reranker chám điểm thế nào"
        # Kết quả à các con số điểm, bảng ngưỡng và ngưỡng hợp lý
        pipeline = build_retrieval_pipeline(db, settings, min_score=0.0)        # lấy điểm thô, chưa lọc
        print(f"{'loại':<10}{'điểm':>7}  câu hỏi")
        print("-" * 80)
        for case in answerable:
            score = judged_score(pipeline.run(case["query"], args.app), case["expected_text"])
            label = " (lỗi reranker / chunk đúng không có trong top-k)" if score is None else ""
            print(f"{'có':<10}{'-' if score is None else f'{score:.2f}':>7}  {case['query'][:55]}{label}")
            if score is not None:
                ans_scores.append(score)

        for case in unanswerable:
            score = judged_score(pipeline.run(case["query"], args.app))
            print(f"{'không':<10}{'-' if score is None else f'{score:.2f}':>7}  {case['query'][:55]}")
            if score is not None:
                unans_scores.append(score)


        if not ans_scores:
            print("\nKhông có điểm nào của câu có đáp án -> kiểm tra reranker và eval/retrieval_smoke.json")
            return


        print(f"\n{'ngưỡng':>7}  {'giữ câu có đáp án':>20}  {'chặn câu không đáp án':>24}")
        for t in THRESHOLDS:
            kept = sum(s >= t for s in ans_scores)
            blocked = sum(s < t for s in unans_scores)
            print(f"{t:>7.2f} {f'{kept}/{len(ans_scores)}':>20} {f'{blocked}/{len(unans_scores)}':>24}")

        value, separable = suggest_threshold(ans_scores, unans_scores)          # tuple (điểm phân tách, có chồng lấn hay không)
        print(f"\n ý RERANKER_MIN_SCORE={value:.2f}"
              + "tách được hoàn toàn" if separable
              else "(KHÔNG tách được hoàn toàn: câu lọt qua sẽ nhờ LLM tự từ chối, chạy thêm --llm để kiểm tra)")

        # Phần 2: dùng LLM hỏi chat bot trả lời
        # Kiểm tra cả hệ thông
        if args.llm:
            service = build_rag_service(db, settings)                                   # tạo chatbot thật
            print(f"\nRAGService đầy đủ, ngưỡng {service.pipeline.min_score:.2f}:")     # in ra ngưỡng
            print(f"{'loại':<10}{'kết quả':<32}câu hỏi")

            # Gộp 2 bộ câu hỏi thành 1 danh sách.
            # Mỗi phần tử là tuple (câu hỏi, có đáp án không)
            rows = [(c["query"], True) for c in answerable] + [(c["query"], False) for c in unanswerable]
            false_abstain = correct_abstain = 0
            for query, has_answer in rows:
                result = service.ask(query, args.app)       # hỏi chatbot, đi lần lượt 2 lớp ở lớp 1 và lớp 2
                time.sleep(args.sleep)
                outcome = f"từ chối ({result.abstain_reason})" if result.abstained else "trả lời"
                if has_answer and result.abstained:
                    false_abstain += 1
                if not has_answer and result.abstained:
                    correct_abstain += 1
                print(f"{'có' if has_answer else 'không':<10}{outcome:<32}{query[:50]}")        # in ra 1 dòng, câu, kết quả

            print(f"\nTừ chối đúng (câu không có đáp án): {correct_abstain}/{len(unanswerable)}")
            print(f"Từ chối nhầm (câu có đáp án):   {false_abstain}/{len(answerable)}")

if __name__ == "__main__":
    main()