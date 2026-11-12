"""
MỤC ĐÍCH: Chấm điểm retrieval (spec Phase 11): Hit@k, Recall@k, Precision@k, nDCG@k, MRR, latency.

So 4 cách tìm trên CÙNG bộ câu hỏi:
    vector   keyword   hybrid   hybrid+rerank (đúng pipeline mà /chat đang dùng, chưa lọc ngưỡng)

Embedding của mỗi câu hỏi được tính 1 LẦN trước (đo latency riêng), rồi cache cho mọi cách tìm:
  - không tốn quota embedding 3 lần cho cùng 1 câu;
  - latency của vector/hybrid là thời gian TÌM trong DB, không lẫn thời gian gọi API embedding.

Chỉ chấm câu answerable=true. Câu không có đáp án được chấm ở generation_eval (có từ chối đúng không).
"""
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.evaluation.dataset import EvalCase
from app.evaluation.ground_truth import build_ground_truth
from app.evaluation.scope import plan_scope
from app.evaluation.metrics import (
    bootstrap_ci, hit_at_k, latency_summary, mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranking import Reranker
from app.rag.retrieval import RETRIEVAL_MODES, build_retriever
from app.rag.retrieval_pipeline import RetrievalPipeline, application_domain_resolver

RERANK_MODE = "hybrid+rerank"   # cột mặc định; mỗi nguồn ứng viên khác có cột "<nguồn>+rerank"


class CachedQueryEmbedder(EmbeddingProvider):
    """Bọc embedder thật: embed_query của cùng 1 câu chỉ gọi API 1 lần."""

    def __init__(self, inner: EmbeddingProvider):
        self.inner = inner
        self.dimension = inner.dimension
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.inner.embed_query(text)
        return self._cache[text]


@dataclass
class ModeResult:
    ranks: dict[str, list[int]] = field(default_factory=dict)   # case.id -> hạng các chunk đúng
    latencies_ms: list[float] = field(default_factory=list)
    unscored: int = 0   # chỉ hybrid+rerank: số câu reranker lỗi rồi fail-open (thứ tự = hybrid)


# (query, application_id, k) -> (danh sách chunk_id theo thứ tự, có được reranker chấm không)
Searcher = Callable[[str, str, int], tuple[list[uuid.UUID], bool]]


def ranks_of(result_ids: list[uuid.UUID], relevant: set[uuid.UUID]) -> list[int]:
    return [rank for rank, cid in enumerate(result_ids, start=1) if cid in relevant]


def build_searchers(
    db: Session, settings: Settings, embedder: EmbeddingProvider, reranker: Reranker | None, max_k: int,
    rerank_sources: tuple[str, ...] = ("hybrid",),
    ) -> dict[str, Searcher]:
    searchers: dict[str, Searcher] = {}
    for mode in RETRIEVAL_MODES:
        retriever = build_retriever(db, settings, embedder=embedder, mode=mode)
        searchers[mode] = lambda q, app, k, r=retriever, **scope: ([c.chunk_id for c in r.retrieve(q, app, k, **scope)], True)

    for source in (rerank_sources if reranker is not None else ()):
        pipeline = RetrievalPipeline(
            build_retriever(db, settings, embedder=embedder, mode=source),   # cùng số ứng viên, chỉ khác nguồn
            reranker,
            retrieval_top_k=max(settings.retrieval_top_k, max_k),
            rerank_top_k=max_k,
            domain_resolver=application_domain_resolver(db),
            min_score=0.0,   # đánh giá THỨ HẠNG, không lọc ngưỡng (ngưỡng đánh giá ở tune_threshold / generation)
        )

        def rerank_search(q: str, app: str, k: int, p=pipeline, **scope) -> tuple[list[uuid.UUID], bool]:
            result = p.run(q, app, **scope)
            judged = not result.reranked or any(r.judged for r in result.reranked)
            return [r.chunk.chunk_id for r in result.reranked][:k], judged

        searchers[f"{source}+rerank"] = rerank_search
    return searchers


def evaluate_retrieval(
    db: Session,
    settings: Settings,
    application_id: str,
    cases: list[EvalCase],
    embedder: EmbeddingProvider,
    reranker: Reranker | None,
    ks: list[int],
    sleep_s: float = 0.0,
    progress: Callable[[str], None] = lambda _msg: None,
    rerank_sources: tuple[str, ...] = ("hybrid",),
    scope: str = "all",
) -> dict:
    ks = sorted(set(ks))
    max_k = ks[-1]
    truth = build_ground_truth(db, application_id, cases)
    plan = plan_scope(db, application_id, cases, scope)         # "paper": mỗi câu chỉ tìm trong paper của nó
    scored = [c for c in plan.keep(cases) if c.id in truth.relevant]

    # 1. Embed trước mọi câu hỏi: đo latency embedding riêng, các cách tìm dùng lại từ cache
    cached = CachedQueryEmbedder(embedder)
    embed_ms: list[float] = []
    for case in scored:
        t0 = time.perf_counter()
        cached.embed_query(plan.questions[case.id].question)
        embed_ms.append((time.perf_counter() - t0) * 1000)

    # 2. Chạy từng cách tìm
    searchers = build_searchers(db, settings, cached, reranker, max_k, rerank_sources)
    results = {mode: ModeResult() for mode in searchers}
    for i, case in enumerate(scored, start=1):
        progress(f"[{i}/{len(scored)}] {case.question[:60]}")
        if i > 1 and sleep_s > 0:
            time.sleep(sleep_s)   # trial key Cohere rerank: 10 request/phút -> nên --sleep 6
        sq = plan.questions[case.id]
        for mode, search in searchers.items():
            t0 = time.perf_counter()
            ids, judged = search(sq.question, application_id, max_k, **sq.kwargs())
            results[mode].latencies_ms.append((time.perf_counter() - t0) * 1000)
            results[mode].ranks[case.id] = ranks_of(ids, truth.relevant[case.id])
            if not judged:
                results[mode].unscored += 1

    # 3. Tính điểm
    modes_out = {}
    for mode, res in results.items():
        # điểm TỪNG câu của mỗi chỉ số; trung bình = điểm của bộ, bootstrap = khoảng tin cậy
        samples: dict[str, list[float]] = {}
        for k in ks:
            samples[f"hit@{k}"] = [hit_at_k(res.ranks[c.id], k) for c in scored]
            samples[f"recall@{k}"] = [recall_at_k(res.ranks[c.id], len(truth.relevant[c.id]), k) for c in scored]
            samples[f"precision@{k}"] = [precision_at_k(res.ranks[c.id], k) for c in scored]
            samples[f"ndcg@{k}"] = [ndcg_at_k(res.ranks[c.id], len(truth.relevant[c.id]), k) for c in scored]
        samples[f"mrr@{max_k}"] = [reciprocal_rank(res.ranks[c.id], max_k) for c in scored]
        metrics = {name: mean(values) for name, values in samples.items()}
        ci = {name: bootstrap_ci(values) for name, values in samples.items()}
        # Điểm theo nhóm tag (vd. "vi" / "en"): câu tiếng Anh hỏi CV tiếng Việt có kém hơn không?
        by_tag = {}
        for tag in sorted({t for c in scored for t in c.tags}):
            group = [c for c in scored if tag in c.tags]
            by_tag[tag] = {
                "n": len(group),
                f"hit@{ks[0]}": mean([hit_at_k(res.ranks[c.id], ks[0]) for c in group]),
                f"mrr@{max_k}": mean([reciprocal_rank(res.ranks[c.id], max_k) for c in group]),
            }
        modes_out[mode] = {"metrics": metrics, "ci": ci, "by_tag": by_tag,
                           "latency": latency_summary(res.latencies_ms), "unscored": res.unscored}

    return {
        "kind": "retrieval",
        "application_id": application_id,
        "ks": ks,
        "counts": {
            "cases": len(cases),
            "answerable": sum(c.answerable for c in cases),
            "scored": len(scored),
            "chunks_in_app": truth.total_chunks,
        },
        "missing_ground_truth": truth.missing,
        "scope": plan.summary(),
        "embedding_latency": latency_summary(embed_ms),
        "modes": modes_out,
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "tags": c.tags,
                "n_relevant": len(truth.relevant[c.id]),
                "ranks": {mode: results[mode].ranks[c.id] for mode in results},
            }
            for c in scored
        ],
    }