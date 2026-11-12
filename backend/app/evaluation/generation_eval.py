"""
MỤC ĐÍCH: Đánh giá generation cơ bản (spec Phase 11): chạy RAGService.ask trên từng câu và đếm.

Câu CÓ đáp án (answerable=true):
    answered        không bị từ chối (bị từ chối = từ chối NHẦM, lỗi nặng nhất với người hỏi)
    cited           có ít nhất 1 [Source n]
    citation_hit    ít nhất 1 nguồn được trích là chunk ĐÚNG (theo expected_texts; câu mà DB
                    không có chunk nào khớp expected_texts thì không chấm, xem missing_ground_truth)
    facts_ok        câu trả lời chứa mọi đoạn trong answer_contains (chỉ câu có khai báo)
    squad_f1        token F1 với đáp án mẫu reference_answers, như HasAns_f1 của SQuAD 2.0
                    (chỉ câu có khai báo; câu bị từ chối = 0, lấy điểm cao nhất nếu có nhiều đáp án mẫu)
    answer_recall   như squad_f1 nhưng chỉ tính recall: không phạt câu trả lời dài
Câu KHÔNG có đáp án (answerable=false):
    abstained       hệ thống từ chối đúng; trả lời = "bịa" từ thông tin không có trong CV

Tùy chọn (judge != None): faithfulness theo kiểu RAGAS cho MỌI câu được trả lời (xem faithfulness.py).
    faithfulness            trung bình điểm 0..1: bao nhiêu phần câu trả lời có trong context
    unfaithful_answer_rate  tỉ lệ câu trả lời có ít nhất 1 ý giám khảo chấm NO

Lỗi API (429, 503...) KHÔNG bị tính là từ chối: ghi riêng vào "errors" và loại khỏi mẫu số,
nếu không thì hết quota sẽ trông như "hệ thống từ chối nhiều".

Mọi câu trả lời được lưu nguyên văn vào file kết quả để bạn TỰ ĐỌC lại: số đếm bằng code chỉ là
lớp kiểm tra đầu, không thay được việc đọc (spec mục 23: phải kiểm tra cả người chấm).
"""

import time
from collections import Counter
from collections.abc import Callable

from app.evaluation.dataset import EvalCase, answer_has_facts
from app.evaluation.faithfulness import FaithfulnessJudge
from app.evaluation.ground_truth import GroundTruth
from app.evaluation.scope import ScopePlan
from app.evaluation.metrics import bootstrap_ci, latency_summary, mean, token_f1, token_recall
from app.services.rag_service import RAGService


def score_vs_references(score: Callable[[str, str], float], case: EvalCase, answer: str,
                         abstained: bool) -> float | None:
    """Điểm cao nhất so với các đáp án mẫu. None = câu này không khai báo reference_answers.

    Từ chối một câu CÓ đáp án = 0 điểm, giống HasAns_f1 của SQuAD: nếu bỏ câu đó ra thì
    hệ thống từ chối càng nhiều, điểm F1 trên phần còn lại càng đẹp.
    """
    if not case.reference_answers:
        return None
    if abstained:
        return 0.0
    return max(score(answer, ref) for ref in case.reference_answers)


def evaluate_generation(
    service: RAGService,
    application_id: str,
    cases: list[EvalCase],
    truth: GroundTruth,
    sleep_s: float = 0.0,
    progress: Callable[[str], None] = lambda _msg: None,
    judge: FaithfulnessJudge | None = None,
    plan: ScopePlan | None = None,
) -> dict:

    """
    plan=None: hỏi cả application. plan từ scope.plan_scope(..., "paper"): mỗi câu chỉ hỏi trong paper của nó,
    câu bị bỏ qua (holdout) không chạy.
    """
    if plan is not None:
        cases = plan.keep(cases)
    rows: list[dict] = []
    for i, case in enumerate(cases, start=1):
        progress(f"[{i}/{len(cases)}] {case.question[:60]}")
        if i > 1 and sleep_s > 0:
            time.sleep(sleep_s)   # giãn cách để không chạm giới hạn request/phút của free tier
        try:
            if plan is None:
                r = service.ask(case.question, application_id)
            else:
                sq = plan.questions[case.id]
                r = service.ask(sq.question, application_id, **sq.kwargs())
        except Exception as exc:  # 1 câu lỗi không làm hỏng cả lượt đánh giá
            rows.append({"id": case.id, "question": case.question, "answerable": case.answerable,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        relevant = truth.relevant.get(case.id)   # None: dataset chưa đúng -> không chấm được citation_hit
        rows.append({
            "id": case.id,
            "question": case.question,
            "answerable": case.answerable,
            "tags": case.tags,
            "answer": r.answer,
            "abstained": r.abstained,
            "abstain_reason": r.abstain_reason,
            "sources": [{"number": s.number, "filename": s.filename, "page": s.page} for s in r.sources],
            "cited": bool(r.sources),
            "citation_hit": None if relevant is None else any(s.chunk_id in relevant for s in r.sources),
            "facts_ok": answer_has_facts(case, r.answer),
            "squad_f1": score_vs_references(token_f1, case, r.answer, r.abstained),
            "answer_recall": score_vs_references(token_recall, case, r.answer, r.abstained),
            "invalid_citations": r.invalid_citations,
            "grounding": r.grounding.verdict if r.grounding else None,
            "top_rerank_score": max((x.score for x in r.retrieval.reranked if x.judged), default=None),
            "retrieval_ms": r.retrieval.retrieval_ms,
            "rerank_ms": r.retrieval.rerank_ms,
            "llm_ms": r.llm_ms,
            "total_ms": r.total_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        })
        if judge is not None and not r.abstained and r.context is not None:
            f = judge.score(r.answer, r.context.text)
            # lưu cả context: scripts/review_faithfulness.py cần nó để bạn tự chấm lại giám khảo
            rows[-1].update(faithfulness=f.to_dict(), context=r.context.text)
    report = summarize_generation(application_id, rows)
    report["missing_ground_truth"] = [c for c in truth.missing if any(x.id == c for x in cases)]
    report["scope"] = plan.summary() if plan is not None else {"mode": "all", "skipped": []}
    return report


def summarize_generation(application_id: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    with_facts = [r for r in answered if r["facts_ok"] is not None]
    with_truth = [r for r in answered if r["citation_hit"] is not None]
    # .get: báo cáo cũ (trước khi có squad_f1) không có trường này
    with_refs = [r for r in ans if r.get("squad_f1") is not None]   # ans, không phải answered: từ chối = 0
    faith = [r["faithfulness"]["score"] for r in ok
             if r.get("faithfulness") and r["faithfulness"]["score"] is not None]

    # điểm TỪNG câu (True = 1, False = 0) của mỗi chỉ số: trung bình = tỉ lệ, bootstrap = khoảng tin cậy
    samples: dict[str, list[float]] = {
        # câu có đáp án
        "answer_rate": [not r["abstained"] for r in ans],
        "false_abstain_rate": [r["abstained"] for r in ans],
        "citation_rate": [r["cited"] for r in answered],
        "citation_hit_rate": [r["citation_hit"] for r in with_truth],
        "facts_ok_rate": [r["facts_ok"] for r in with_facts],
        "squad_f1": [r["squad_f1"] for r in with_refs],
        "answer_recall": [r["answer_recall"] for r in with_refs],
        # câu không có đáp án
        "correct_abstain_rate": [r["abstained"] for r in unans],
        "hallucination_rate": [not r["abstained"] for r in unans],
        # faithfulness (chỉ có khi chạy với giám khảo; None = không chấm)
        "faithfulness": faith,
        "unfaithful_answer_rate": [score < 1.0 for score in faith],
    }
    metrics = {name: mean(values) for name, values in samples.items()}
    metrics["invalid_citation_count"] = sum(len(r["invalid_citations"]) for r in ok)   # số đếm, không có khoảng
    return {
        "kind": "generation",
        "application_id": application_id,
        "counts": {
            "cases": len(rows), "errors": len(rows) - len(ok),
            "answerable": len(ans), "unanswerable": len(unans), "facts_checked": len(with_facts),
            "squad_scored": len(with_refs),
            "faithfulness_judged": len(faith),
            "faithfulness_failed": sum(1 for r in ok if r.get("faithfulness") and r["faithfulness"]["score"] is None),
            "judge_llm_calls": sum(r["faithfulness"]["llm_calls"] for r in ok if r.get("faithfulness")),
        },
        "metrics": metrics,
        "ci": {name: bootstrap_ci(values) for name, values in samples.items()},
        "abstain_reasons": dict(Counter(r["abstain_reason"] for r in ok if r["abstained"])),
        "grounding": dict(Counter(r["grounding"] for r in ok if r["grounding"])),
        "latency": {
            stage: latency_summary([r[f"{stage}_ms"] for r in ok])
            for stage in ("retrieval", "rerank", "llm", "total")
        },
        "tokens": {
            "input": sum(r["input_tokens"] or 0 for r in ok),
            "output": sum(r["output_tokens"] or 0 for r in ok),
        },
        "cases": rows,
    }