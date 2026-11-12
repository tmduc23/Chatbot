"""
MỤC ĐÍCH: Lưu, in và SO SÁNH kết quả đánh giá (Phase 11).

Mỗi lần chạy lưu 1 file eval/results/<thời gian>-<loại>.json gồm: điểm, latency, chi tiết từng câu
và CẤU HÌNH lúc chạy (provider, model, ngưỡng...). Có cấu hình thì mới biết 2 lần chạy khác nhau ở đâu:
"hôm qua MRR 0.82, hôm nay 0.75" là vô nghĩa nếu không biết hôm nay đã đổi embedding model.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.rag.reranking import get_min_score


def config_snapshot(settings: Settings) -> dict:
    reranker = settings.reranker_provider.lower()
    return {
        "embedding": f"{settings.embedding_provider}/"
                     f"{settings.cohere_embedding_model if settings.embedding_provider == 'cohere' else settings.embedding_model}",
        "retrieval_mode": settings.retrieval_mode,
        "keyword_ranking": settings.keyword_ranking,
        "retrieval_candidate_k": settings.retrieval_candidate_k,
        "rrf_k": settings.rrf_k,
        "rrf_keyword_weight": settings.rrf_keyword_weight,
        "retrieval_top_k": settings.retrieval_top_k,
        "reranker": reranker,
        "reranker_model": {"gemini": settings.reranker_model, "cohere": settings.cohere_rerank_model}.get(reranker),
        "reranker_top_k": settings.reranker_top_k,
        "min_score": get_min_score(settings),
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "answer_prompt": settings.answer_prompt,
        "grounding_check": settings.grounding_check,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_section_header": settings.chunk_section_header,
    }   # không bao giờ đưa API key vào đây: file kết quả có thể được commit


def file_fingerprint(path: Path) -> str:
    """12 ký tự đầu của sha256 — biết 2 lần chạy có dùng CÙNG dataset không."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def save_report(report: dict, out_dir: Path, settings: Settings, dataset: Path, label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "dataset": {"path": str(dataset), "sha256_12": file_fingerprint(dataset)},
        # scope ("all" | "paper") nằm trong config để compare_results.py báo khi 2 lần chạy khác phạm vi
        "config": {**config_snapshot(settings), "scope": report.get("scope", {}).get("mode", "all")},
        **report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{label}" if label else ""
    path = out_dir / f"{stamp}-{report['kind']}{suffix}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def fmt_ci(ci: list[float] | None, pct: bool = False) -> str:
    """Khoảng tin cậy dạng "68–92%" hoặc "0.70–0.88"; "-" khi không tính được (ít hơn 2 câu)."""
    if ci is None:
        return "-"
    lo, hi = ci
    return f"{lo * 100:.0f}–{hi * 100:.0f}%" if pct else f"{lo:.2f}–{hi:.2f}"


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def flatten_metrics(report: dict) -> dict[str, float | None]:
    """Đưa mọi chỉ số về dạng phẳng "tên -> số" để so 2 báo cáo: 'hybrid hit@3', 'false_abstain_rate'..."""
    flat: dict[str, float | None] = {}
    if report["kind"] == "retrieval":
        for mode, data in report["modes"].items():
            for name, value in data["metrics"].items():
                flat[f"{mode} {name}"] = value
            flat[f"{mode} p95_ms"] = data["latency"]["p95_ms"]
    else:
        flat.update(report["metrics"])
        flat["total p95_ms"] = report["latency"]["total"]["p95_ms"]
    return flat


def compare_reports(a: dict, b: dict) -> list[tuple[str, float | None, float | None, float | None]]:
    """[(tên chỉ số, A, B, B - A)] cho mọi chỉ số có ở cả 2 báo cáo cùng loại."""
    if a["kind"] != b["kind"]:
        raise ValueError(f"Không so được báo cáo '{a['kind']}' với '{b['kind']}'")
    fa, fb = flatten_metrics(a), flatten_metrics(b)
    rows = []
    for name in fa:
        if name in fb:
            va, vb = fa[name], fb[name]
            rows.append((name, va, vb, None if va is None or vb is None else vb - va))
    return rows


def config_diff(a: dict, b: dict) -> dict[str, tuple]:
    ca, cb = a.get("config", {}), b.get("config", {})
    return {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
