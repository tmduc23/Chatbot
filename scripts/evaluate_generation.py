"""
MỤC ĐÍCH: Đánh giá generation cơ bản trên eval/dataset.json (Phase 11).

    python scripts/evaluate_generation.py --app cv --limit 5          # thử 5 câu đầu
    python scripts/evaluate_generation.py --app cv --sleep 4          # chạy hết, nghỉ 4s giữa 2 câu
    python scripts/evaluate_generation.py --app cv --only unanswerable
    python scripts/evaluate_generation.py --app squad --sample 40 --sleep 10 --faithfulness --judge-model <model khác>
    python scripts/evaluate_generation.py --app qasper --scope paper --sample 40 --sleep 6   # chế độ paper

Quota: mỗi câu = 1 embedding + 1 rerank + tối đa 1 LLM (+1 nếu GROUNDING_CHECK=true).
Câu bị chặn ở ngưỡng rerank không gọi LLM.
--faithfulness: thêm tối đa 2 lần gọi LLM (giám khảo) cho mỗi câu ĐƯỢC TRẢ LỜI.
"""
import argparse
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.evaluation.dataset import DatasetError, load_dataset, stratified_sample
from app.evaluation.faithfulness import build_faithfulness_judge, judge_model_name
from app.evaluation.generation_eval import evaluate_generation
from app.evaluation.ground_truth import build_ground_truth
from app.evaluation.report import fmt, fmt_ci, fmt_ms, save_report
from app.evaluation.scope import SCOPES, plan_scope
from app.services.rag_service import build_rag_service

LABELS = [
    ("answer_rate", "Câu có đáp án được trả lời", True),
    ("false_abstain_rate", "  ...bị từ chối nhầm", True),
    ("citation_rate", "  ...có trích nguồn", True),
    ("citation_hit_rate", "  ...trích đúng chunk có đáp án", True),
    ("facts_ok_rate", "  ...chứa đủ answer_contains", True),
    ("squad_f1", "  ...token F1 với đáp án mẫu (SQuAD, 0..1)", False),
    ("answer_recall", "  ...token recall với đáp án mẫu (0..1)", False),
    ("correct_abstain_rate", "Câu không có đáp án bị từ chối đúng", True),
    ("hallucination_rate", "  ...bị trả lời (bịa)", True),
    ("invalid_citation_count", "Số [Source n] bịa đã bị xoá", False),
    ("faithfulness", "Faithfulness (giám khảo LLM, 0..1)", False),
    ("unfaithful_answer_rate", "  ...có ý không có trong context", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answers: abstain, citations, facts, latency")
    parser.add_argument("--app", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--only", choices=["all", "answerable", "unanswerable"], default="all")
    parser.add_argument("--limit", type=int, help="Chỉ chạy N câu đầu (sau khi lọc --only)")
    parser.add_argument("--sample", type=int, help="Lấy ngẫu nhiên N câu, giữ tỉ lệ câu không có đáp án")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho --sample")
    parser.add_argument("--sleep", type=float, default=0.0, help="Số giây nghỉ giữa 2 câu (giới hạn RPM)")
    parser.add_argument("--scope", choices=SCOPES, default="all",
                        help="all: tìm trong cả app; paper: mỗi câu chỉ tìm trong paper của nó (như QASPER/PeerQA gốc)")
    parser.add_argument("--faithfulness", action="store_true", help="Chấm faithfulness kiểu RAGAS (tốn thêm lượt LLM)")
    parser.add_argument("--judge-model", help="Model giám khảo (mặc định JUDGE_MODEL, rồi GROUNDING_MODEL, rồi LLM_MODEL); nên khác model trả lời")
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    try:
        cases = load_dataset(args.dataset)
    except DatasetError as exc:
        sys.exit(f"Dataset lỗi: {exc}")
    if args.only != "all":
        cases = [c for c in cases if c.answerable == (args.only == "answerable")]

    settings = get_settings()
    judge = build_faithfulness_judge(settings, args.judge_model) if args.faithfulness else None

    def llm_calls(run: list) -> int:
        return len(run) * ((2 if settings.grounding_check else 1) + (2 if judge else 0))

    if judge:
        judge_model = judge_model_name(settings, args.judge_model)
        same = " (⚠ TRÙNG model trả lời: giám khảo dễ dãi với chính mình)" if judge_model == settings.llm_model else ""
        print(f"Giám khảo faithfulness: {settings.llm_provider}/{judge_model}, temperature 0{same}")

    with SessionLocal() as db:
        # chế độ paper: bỏ câu không có paper (holdout) TRƯỚC khi lấy mẫu, để --sample 40 vẫn ra 40 câu chạy được
        plan = plan_scope(db, args.app, cases, args.scope)
        if plan.skipped:
            print(f"Chế độ paper: bỏ qua {len(plan.skipped)} câu không có paper trong app "
                  f"({', '.join(plan.skipped[:10])}{' ...' if len(plan.skipped) > 10 else ''})")
        cases = plan.keep(cases)
        if args.sample:
            cases = stratified_sample(cases, args.sample, args.seed)
        cases = cases[: args.limit]
        print(f"{len(cases)} câu | LLM {settings.llm_provider}/{settings.llm_model} | "
              f"grounding {'BẬT' if settings.grounding_check else 'tắt'} | gọi LLM tối đa {llm_calls(cases)} lần")
        truth = build_ground_truth(db, args.app, cases)
        service = build_rag_service(db, settings)
        report = evaluate_generation(service, args.app, cases, truth, sleep_s=args.sleep,
                                     progress=lambda msg: print(f"  {msg}", flush=True), judge=judge,
                                     plan=plan if args.scope == "paper" else None)

    counts, m = report["counts"], report["metrics"]
    print(f"\nĐã chạy {counts['cases']} câu: {counts['answerable']} có đáp án, "
          f"{counts['unanswerable']} không có đáp án, {counts['errors']} lỗi API")
    if report["missing_ground_truth"]:
        print(f"⚠ Không chunk nào chứa expected_texts của: {', '.join(report['missing_ground_truth'])} "
              f"-> các câu này không được chấm 'trích đúng chunk' (sửa dataset)")
    print(f"  {'':<40}{'':>8}  khoảng tin cậy 95%")
    for key, label, pct in LABELS:
        value = m[key] if isinstance(m[key], int) else fmt(m[key], pct)   # số đếm in nguyên, tỉ lệ/điểm qua fmt
        ci = fmt_ci(report["ci"][key], pct) if key in report["ci"] else ""
        print(f"  {label:<40}{value:>8}  {ci}")
    if report["abstain_reasons"]:
        print(f"  Lý do từ chối: {report['abstain_reasons']}")
    if report["grounding"]:
        print(f"  Grounding (LLM chấm): {report['grounding']}")
    if judge:
        print(f"  Faithfulness: chấm {counts['faithfulness_judged']} câu, không chấm được {counts['faithfulness_failed']}, "
              f"giám khảo gọi LLM {counts['judge_llm_calls']} lần")
    lat = report["latency"]
    print("\nLatency (ms)      p50     p95")
    for stage in ("retrieval", "rerank", "llm", "total"):
        print(f"  {stage:<12}{fmt_ms(lat[stage]['p50_ms']):>8}{fmt_ms(lat[stage]['p95_ms']):>8}")
    print(f"Tokens: vào {report['tokens']['input']}, ra {report['tokens']['output']}")

    print("\nCần đọc lại bằng mắt:")
    for row in report["cases"]:
        if "error" in row:
            print(f"  {row['id']}: LỖI {row['error'][:90]}")
        elif row["answerable"] and row["abstained"]:
            print(f"  {row['id']}: TỪ CHỐI NHẦM ({row['abstain_reason']}, top rerank "
                  f"{fmt(row['top_rerank_score'])}) {row['question'][:50]}")
        elif not row["answerable"] and not row["abstained"]:
            print(f"  {row['id']}: TRẢ LỜI câu không có đáp án: {row['answer'][:70]!r}")
        elif row["answerable"] and row["citation_hit"] is False:
            print(f"  {row['id']}: không trích chunk đúng: {row['answer'][:70]!r}")
        elif row["facts_ok"] is False:
            print(f"  {row['id']}: thiếu answer_contains: {row['answer'][:70]!r}")
        faith = row.get("faithfulness") or {}
        for claim in faith.get("claims", []):
            if claim["supported"] is False:
                print(f"  {row['id']}: giám khảo chấm NO (faithfulness {fmt(faith['score'])}): {claim['text'][:70]!r}")

    path = save_report(report, args.out, settings, args.dataset, args.label)
    print(f"\nĐã lưu {path} (có nguyên văn mọi câu trả lời)")
    try:   # cần openpyxl (requirements.txt); thiếu thì vẫn giữ .json, chỉ bỏ qua Excel
        from app.evaluation.excel import export_excel
        print(f"Đã lưu {export_excel(path)} (mở bằng Excel)")
    except ImportError:
        print("⚠ Chưa có openpyxl nên không xuất Excel: docker compose up -d --build backend")


if __name__ == "__main__":
    main()
