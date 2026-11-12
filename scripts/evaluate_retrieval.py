"""
MỤC ĐÍCH: Đánh giá retrieval trên eval/dataset.json (Phase 11, spec mục 23).

    python scripts/evaluate_retrieval.py --app cv
    python scripts/evaluate_retrieval.py --app cv -k 1 3 5 --label cohere-embed
    python scripts/evaluate_retrieval.py --app cv --no-rerank        # không tốn quota reranker
    python scripts/evaluate_retrieval.py --app cv -k 1 2 3 --sleep 6 # CV ít chunk + trial key Cohere
    python scripts/evaluate_retrieval.py --app squad --rerank-from vector hybrid   # thêm cột vector+rerank (x2 quota)
    python scripts/evaluate_retrieval.py --app qasper --scope paper -k 1 3 5 10 --rerank-from vector   # chế độ paper


Quota: mỗi câu có đáp án = 1 lần gọi embedding + 1 lần gọi reranker cho MỖI nguồn ở --rerank-from
Kết quả lưu ở eval/results/, so 2 lần chạy bằng scripts/compare_results.py.
"""
import argparse
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.report import fmt, fmt_ci, fmt_ms, save_report
from app.evaluation.retrieval_eval import evaluate_retrieval
from app.evaluation.scope import SCOPES
from app.rag.embeddings import get_embedding_provider
from app.rag.reranking import get_reranker
from app.rag.retrieval import RETRIEVAL_MODES


def print_report(report: dict) -> None:
    modes = list(report["modes"])
    width = 16
    print(f"\n{'chỉ số':<16}" + "".join(f"{m:>{width}}" for m in modes))
    print("-" * (16 + width * len(modes)))
    for name in report["modes"][modes[0]]["metrics"]:
        print(f"{name:<16}" + "".join(f"{fmt(report['modes'][m]['metrics'][name]):>{width}}" for m in modes))
        if name.startswith(("hit@", "ndcg@", "mrr@")):   # khoảng tin cậy 95% cho các chỉ số chính
            print(f"{'  95% CI':<16}" + "".join(f"{fmt_ci(report['modes'][m]['ci'][name]):>{width}}" for m in modes))
    for stat in ("p50_ms", "p95_ms"):
        print(f"{stat:<16}" + "".join(f"{fmt_ms(report['modes'][m]['latency'][stat]):>{width}}" for m in modes))
    tags = list(report["modes"][modes[0]]["by_tag"])
    if tags:
        print("\nTheo tag:")
        for tag in tags:
            first = report["modes"][modes[0]]["by_tag"][tag]
            for name in [n for n in first if n != "n"]:
                cells = "".join(f"{fmt(report['modes'][m]['by_tag'][tag][name]):>{width}}" for m in modes)
                print(f"{(tag + ' ' + name)[:15]:<16}{cells}   (n={first['n']})")
    emb = report["embedding_latency"]
    print(f"\nEmbedding câu hỏi (tính riêng, không nằm trong cột trên): "
          f"p50 {fmt_ms(emb['p50_ms'])} ms, p95 {fmt_ms(emb['p95_ms'])} ms")

    print(f"\n{'id':<6}{'câu hỏi':<42}" + "".join(f"{m:>{width}}" for m in modes) + "   (hạng chunk đúng)")
    for case in report["cases"]:
        cells = "".join(f"{(','.join(map(str, case['ranks'][m])) or '-'):>{width}}" for m in modes)
        print(f"{case['id']:<6}{case['question'][:40]:<42}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval: Hit@k, Recall@k, Precision@k, nDCG@k, MRR, latency")
    parser.add_argument("--app", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--no-rerank", action="store_true", help="Bỏ các cột +rerank")
    parser.add_argument("--rerank-from", nargs="+", choices=RETRIEVAL_MODES, default=["hybrid"],
                        help="Nguồn ứng viên cho reranker, mỗi nguồn 1 cột '<nguồn>+rerank' (vd. vector hybrid)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Số giây nghỉ giữa 2 câu (giới hạn RPM của reranker)")
    parser.add_argument("--scope", choices=SCOPES, default="all",
                        help="all: tìm trong cả app; paper: mỗi câu chỉ tìm trong paper của nó (như QASPER/PeerQA gốc)")
    parser.add_argument("--label", help="Tên ngắn gắn vào file kết quả, vd. gemini-embed")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    try:
        cases = load_dataset(args.dataset)
    except DatasetError as exc:
        sys.exit(f"Dataset lỗi: {exc}")
    if any(k < 1 for k in args.k):
        sys.exit("-k phải >= 1")

    settings = get_settings()
    n_ans = sum(c.answerable for c in cases)
    print(f"{n_ans} câu có đáp án | embedding {settings.embedding_provider} | "
          f"reranker {'tắt' if args.no_rerank else settings.reranker_provider}")
    n_rerank = 0 if args.no_rerank else n_ans * len(args.rerank_from)
    if n_rerank and len(args.rerank_from) > 1 and args.sleep and args.sleep < 6 * len(args.rerank_from):
        print(f"⚠ Mỗi câu gọi reranker {len(args.rerank_from)} lần: trial key Cohere (10 lần/phút) cần "
              f"--sleep {6 * len(args.rerank_from)}")
    print(f"Sẽ gọi API: ~{n_ans} lần embedding" + (f" + ~{n_rerank} lần rerank" if n_rerank else ""))

    with SessionLocal() as db:
        report = evaluate_retrieval(
            db, settings, args.app, cases,
            embedder=get_embedding_provider(settings),
            reranker=None if args.no_rerank else get_reranker(settings),
            ks=args.k,
            rerank_sources=tuple(dict.fromkeys(args.rerank_from)),   # bỏ trùng, giữ thứ tự
            sleep_s=args.sleep,
            scope=args.scope,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )

    counts = report["counts"]
    print(f"\nApplication '{args.app}': {counts['chunks_in_app']} chunk | chấm {counts['scored']}/{counts['answerable']} câu")
    if counts["chunks_in_app"] and max(args.k) >= counts["chunks_in_app"]:
        print(f"⚠ k={max(args.k)} >= số chunk ({counts['chunks_in_app']}): Hit/Recall@{max(args.k)} gần như chắc chắn 100%, "
              f"hãy nhìn k nhỏ hơn và MRR")
    if report["scope"]["mode"] == "paper":
        skipped = report["scope"]["skipped"]
        print(f"Chế độ paper: mỗi câu chỉ tìm trong paper của nó, đã bỏ tiền tố tên paper"
              + (f"; bỏ qua {len(skipped)} câu không có paper trong app: {', '.join(skipped[:10])}"
                 + (" ..." if len(skipped) > 10 else "") if skipped else ""))
    if report["missing_ground_truth"]:
        print(f"⚠ Không chunk nào chứa expected_texts của: {', '.join(report['missing_ground_truth'])} "
              f"-> sửa dataset (gõ sai? đoạn chữ bị cắt giữa 2 chunk?). Các câu này KHÔNG được chấm.")
    for mode, data in report["modes"].items():
        if mode.endswith("+rerank") and data["unscored"]:
            print(f"⚠ Reranker lỗi ở {data['unscored']} câu (fail-open, giữ thứ tự gốc): cột {mode} chưa đáng tin, "
                  f"chạy lại sau ít phút")

    print_report(report)
    path = save_report(report, args.out, settings, args.dataset, args.label)
    print(f"\nĐã lưu {path}")
    try:   # cần openpyxl (requirements.txt); thiếu thì vẫn giữ .json, chỉ bỏ qua Excel
        from app.evaluation.excel import export_excel
        print(f"Đã lưu {export_excel(path)} (mở bằng Excel)")
    except ImportError:
        print("⚠ Chưa có openpyxl nên không xuất Excel: docker compose up -d --build backend")


if __name__ == "__main__":
    main()
