"""
MỤC ĐÍCH: Test chạy đánh giá retrieval + generation với DB thật và provider GIẢ (Phase 11).

Mỗi tài liệu nhỏ = đúng 1 chunk, nên biết trước chunk nào đúng cho câu nào.
Không gọi Gemini / Cohere / OpenRouter: không tốn quota.
"""
import json
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.db.session import SessionLocal
from app.evaluation.dataset import EvalCase
from app.evaluation.faithfulness import FaithfulnessJudge
from app.evaluation.generation_eval import evaluate_generation, summarize_generation
from app.evaluation.ground_truth import build_ground_truth
from app.evaluation.report import compare_reports, config_diff, save_report
from app.evaluation.retrieval_eval import RERANK_MODE, evaluate_retrieval
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.base import LLMProvider, LLMResponse
from app.rag.llm.fake import FakeLLM
from app.rag.reranking import PassthroughReranker
from app.rag.reranking.fake import FakeReranker
from app.services.document_service import DocumentService
from app.services.rag_service import build_rag_service

SETTINGS = Settings(chunk_size=500, chunk_overlap=50, tokenizer_encoding="whitespace")
EMBEDDER = FakeEmbeddingProvider(EMBEDDING_DIM)
DOCS = {
    "edu.txt": "Học vấn: Đại học Bách khoa Hà Nội (HUST), GPA 3.35/4.0.",
    "skills.txt": "Kỹ năng: Python, Docker, FastAPI, PostgreSQL.",
    "lang.txt": "Ngoại ngữ: TOEIC 875.",
}
CASES = [
    EvalCase(id="a01", question="Bạn ấy học Bách khoa nào?", answerable=True,
             expected_texts=["bách khoa hà nội"], tags=["vi"]),
    EvalCase(id="a02", question="Docker FastAPI", answerable=True, expected_texts=["Docker"], tags=["en"]),
    EvalCase(id="a03", question="Có chứng chỉ AWS không?", answerable=True, expected_texts=["AWS Certified"]),
    EvalCase(id="u01", question="Lương mong muốn bao nhiêu?", answerable=False),
]


@pytest.fixture
def db() -> Iterator[Session]:
    session = SessionLocal()
    yield session
    session.close()


def _make_app(db: Session, tmp_path: Path, docs: dict[str, str]) -> str:
    app_id = f"t_{uuid.uuid4().hex[:8]}"
    db.add(Application(id=app_id, name="eval test"))
    db.commit()
    service = DocumentService(db, settings=SETTINGS, storage_dir=tmp_path / app_id, embedder=EMBEDDER)
    for name, text in docs.items():
        path = tmp_path / f"{app_id}-{name}"
        path.write_text(text, encoding="utf-8")
        service.ingest_file(app_id, path, filename=name)
    return app_id


@pytest.fixture
def app_id(db: Session, tmp_path: Path) -> Iterator[str]:
    app_id = _make_app(db, tmp_path, DOCS)
    yield app_id
    db.rollback()
    db.execute(delete(Application).where(Application.id == app_id))
    db.commit()


def _retrieval(db: Session, app_id: str, reranker=None, ks=(1, 3)) -> dict:
    return evaluate_retrieval(db, SETTINGS, app_id, CASES, embedder=EMBEDDER,
                              reranker=reranker or FakeReranker(), ks=list(ks))


# ---------------- ground truth ----------------

def test_ground_truth_finds_chunks_and_reports_missing(db: Session, app_id: str) -> None:
    truth = build_ground_truth(db, app_id, CASES)
    assert truth.total_chunks == 3
    assert set(truth.relevant) == {"a01", "a02"}          # u01 không có đáp án -> không cần chunk đúng
    assert truth.missing == ["a03"]                        # gõ đoạn chữ không có trong tài liệu


def test_ground_truth_never_reads_other_application(db: Session, app_id: str, tmp_path: Path) -> None:
    other = _make_app(db, tmp_path, DOCS)                  # application khác có ĐÚNG cùng nội dung
    try:
        truth = build_ground_truth(db, app_id, CASES)
        assert len(truth.relevant["a02"]) == 1             # không đếm chunk của application kia
    finally:
        db.execute(delete(Application).where(Application.id == other))
        db.commit()


# ---------------- retrieval ----------------

def test_retrieval_report_shape(db: Session, app_id: str) -> None:
    report = _retrieval(db, app_id)
    assert report["counts"] == {"cases": 4, "answerable": 3, "scored": 2, "chunks_in_app": 3}
    assert report["missing_ground_truth"] == ["a03"]
    assert set(report["modes"]) == {"vector", "keyword", "hybrid", RERANK_MODE}
    metrics = report["modes"]["hybrid"]["metrics"]
    assert set(metrics) == {"hit@1", "recall@1", "precision@1", "ndcg@1",
                            "hit@3", "recall@3", "precision@3", "ndcg@3", "mrr@3"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    ci = report["modes"]["hybrid"]["ci"]
    assert set(ci) == set(metrics)                              # mỗi chỉ số có 1 khoảng tin cậy
    assert all(lo <= metrics[name] <= hi for name, (lo, hi) in ci.items())
    assert [c["id"] for c in report["cases"]] == ["a01", "a02"]
    assert set(report["modes"]["keyword"]["by_tag"]) == {"vi", "en"}


def test_keyword_and_rerank_rank_exact_match_first(db: Session, app_id: str) -> None:
    report = _retrieval(db, app_id)
    a02 = next(c for c in report["cases"] if c["id"] == "a02")
    assert a02["ranks"]["keyword"] == [1]                  # "docker" chỉ có trong skills.txt
    assert a02["ranks"][RERANK_MODE] == [1]
    assert report["modes"][RERANK_MODE]["unscored"] == 0


def test_unscored_rerank_is_counted(db: Session, app_id: str) -> None:
    # PassthroughReranker = giống reranker bị lỗi rồi fail-open: không chấm điểm câu nào
    report = _retrieval(db, app_id, reranker=PassthroughReranker())
    assert report["modes"][RERANK_MODE]["unscored"] == 2


def test_no_reranker_means_no_rerank_column(db: Session, app_id: str) -> None:
    report = evaluate_retrieval(db, SETTINGS, app_id, CASES, embedder=EMBEDDER, reranker=None, ks=[1])
    assert RERANK_MODE not in report["modes"]


def test_each_question_is_embedded_once(db: Session, app_id: str) -> None:
    class CountingEmbedder(FakeEmbeddingProvider):
        calls = 0

        def embed_query(self, text: str) -> list[float]:
            CountingEmbedder.calls += 1
            return super().embed_query(text)

    evaluate_retrieval(db, SETTINGS, app_id, CASES, embedder=CountingEmbedder(EMBEDDING_DIM),
                       reranker=FakeReranker(), ks=[1, 3])
    assert CountingEmbedder.calls == 2                     # 2 câu được chấm, dù có 4 cách tìm


# ---------------- generation ----------------

class BrokenLLM(LLMProvider):
    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        raise LLMError("HTTP 429 quota exceeded")


def _generation(db: Session, app_id: str, llm: LLMProvider) -> dict:
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(), llm=llm, min_score=0.5)
    cases = [c for c in CASES if c.id in ("a02", "u01")]
    return evaluate_generation(service, app_id, cases, build_ground_truth(db, app_id, cases))


def test_generation_counts_answers_and_abstains(db: Session, app_id: str) -> None:
    report = _generation(db, app_id, FakeLLM("Bạn ấy dùng Docker [Source 1]."))
    m = report["metrics"]
    assert report["counts"]["errors"] == 0
    assert m["answer_rate"] == 1.0 and m["false_abstain_rate"] == 0.0
    assert m["citation_rate"] == 1.0 and m["citation_hit_rate"] == 1.0
    assert m["correct_abstain_rate"] == 1.0 and m["hallucination_rate"] == 0.0
    assert report["abstain_reasons"] == {"no_relevant_chunks": 1}
    a02 = next(r for r in report["cases"] if r["id"] == "a02")
    assert a02["answer"] == "Bạn ấy dùng Docker [Source 1]."    # lưu nguyên văn để đọc lại
    assert a02["sources"][0]["filename"] == "skills.txt"


def test_api_errors_are_not_counted_as_abstains(db: Session, app_id: str) -> None:
    report = _generation(db, app_id, BrokenLLM())
    assert report["counts"]["errors"] == 1                      # a02 gọi LLM -> lỗi
    assert report["metrics"]["false_abstain_rate"] is None      # không còn câu có đáp án nào để tính
    assert report["metrics"]["correct_abstain_rate"] == 1.0     # u01 bị chặn trước khi gọi LLM
    assert "quota" in next(r for r in report["cases"] if r["id"] == "a02")["error"]


def test_citation_hit_not_scored_without_ground_truth(db: Session, app_id: str) -> None:
    # a03: expected_texts không có trong tài liệu -> không biết chunk nào đúng -> không chấm
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(),
                                llm=FakeLLM("Có [Source 1]."), min_score=0.0)
    cases = [c for c in CASES if c.id in ("a02", "a03")]
    report = evaluate_generation(service, app_id, cases, build_ground_truth(db, app_id, cases))
    rows = {r["id"]: r for r in report["cases"]}
    assert rows["a03"]["citation_hit"] is None
    assert report["metrics"]["citation_hit_rate"] == 1.0         # chỉ tính a02
    assert report["missing_ground_truth"] == ["a03"]


class ScriptedLLM(LLMProvider):
    """Giám khảo giả: trả lần lượt các câu soạn sẵn, ghi lại prompt."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, system_instruction: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text=self.replies.pop(0), model="judge")


def test_generation_with_faithfulness_judge(db: Session, app_id: str) -> None:
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(),
                                llm=FakeLLM("Bạn ấy dùng Docker và AWS [Source 1]."), min_score=0.5)
    judge_llm = ScriptedLLM("- Bạn ấy dùng Docker\n- Bạn ấy dùng AWS", "1: YES\n2: NO")
    cases = [c for c in CASES if c.id in ("a02", "u01")]
    report = evaluate_generation(service, app_id, cases, build_ground_truth(db, app_id, cases),
                                 judge=FaithfulnessJudge(judge_llm))
    rows = {r["id"]: r for r in report["cases"]}
    assert rows["a02"]["faithfulness"]["score"] == 0.5
    assert "Docker" in rows["a02"]["context"]                  # context đúng là cái LLM trả lời đã đọc
    assert "Docker" in judge_llm.prompts[1]
    assert "faithfulness" not in rows["u01"]                   # bị từ chối -> không có gì để chấm
    assert report["metrics"]["faithfulness"] == 0.5
    assert report["metrics"]["unfaithful_answer_rate"] == 1.0
    assert report["counts"]["faithfulness_judged"] == 1 and report["counts"]["judge_llm_calls"] == 2


def test_summarize_generation_facts_only_over_declared_cases() -> None:
    def row(i: str, facts: bool | None, abstained: bool = False) -> dict:
        return {"id": i, "question": "q", "answerable": True, "abstained": abstained, "abstain_reason": None,
                "cited": True, "citation_hit": True, "facts_ok": facts, "invalid_citations": [], "grounding": None,
                "retrieval_ms": 1.0, "rerank_ms": 1.0, "llm_ms": 1.0, "total_ms": 3.0,
                "input_tokens": 10, "output_tokens": 5}

    report = summarize_generation("cv", [row("a", True), row("b", False), row("c", None)])
    assert report["metrics"]["facts_ok_rate"] == 0.5            # c không khai báo answer_contains -> không tính
    assert report["counts"]["facts_checked"] == 2
    assert report["metrics"]["correct_abstain_rate"] is None    # không có câu không-đáp-án nào
    assert report["tokens"] == {"input": 30, "output": 15}
    assert report["metrics"]["faithfulness"] is None            # không chạy giám khảo -> không có điểm, không phải 0
    assert report["metrics"]["squad_f1"] is None                # không câu nào có reference_answers
    lo, hi = report["ci"]["facts_ok_rate"]                      # 2 câu: 1 đúng, 1 sai
    assert lo <= 0.5 <= hi
    assert report["ci"]["correct_abstain_rate"] is None         # 0 câu -> không có khoảng
    assert "invalid_citation_count" not in report["ci"]         # số đếm, không phải tỉ lệ


def test_generation_squad_f1_against_reference_answers(db: Session, app_id: str) -> None:
    a02 = replace(next(c for c in CASES if c.id == "a02"), reference_answers=["Python", "Docker"])
    service = build_rag_service(db, SETTINGS, embedder=EMBEDDER, reranker=FakeReranker(),
                                llm=FakeLLM("Bạn ấy dùng Docker [Source 1]."), min_score=0.5)
    report = evaluate_generation(service, app_id, [a02], build_ground_truth(db, app_id, [a02]))
    row = report["cases"][0]
    assert row["squad_f1"] == pytest.approx(0.4)    # khớp "Docker": precision 1/4 (bạn ấy dùng docker), recall 1
    assert row["answer_recall"] == 1.0
    assert report["metrics"]["squad_f1"] == pytest.approx(0.4) and report["counts"]["squad_scored"] == 1


def test_summarize_squad_f1_counts_false_abstain_as_zero() -> None:
    def row(i: str, f1: float | None, abstained: bool) -> dict:
        return {"id": i, "question": "q", "answerable": True, "abstained": abstained, "abstain_reason": None,
                "cited": not abstained, "citation_hit": None, "facts_ok": None, "squad_f1": f1, "answer_recall": f1,
                "invalid_citations": [], "grounding": None, "retrieval_ms": 1.0, "rerank_ms": 1.0, "llm_ms": 1.0,
                "total_ms": 3.0, "input_tokens": 0, "output_tokens": 0}

    report = summarize_generation("squad", [row("a", 1.0, False), row("b", 0.0, True), row("c", None, False)])
    assert report["metrics"]["squad_f1"] == 0.5     # b từ chối nhầm = 0 vẫn được tính; c không có đáp án mẫu -> bỏ
    assert report["counts"]["squad_scored"] == 2


# ---------------- report ----------------

def test_save_report_has_config_but_no_secret(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    settings = Settings(cohere_api_key="super-secret-key", gemini_api_key="another-secret")
    path = save_report({"kind": "generation", "metrics": {}}, tmp_path / "out", settings, dataset, label="ab")

    assert path.name.endswith("-generation-ab.json")
    text = path.read_text(encoding="utf-8")
    assert "secret" not in text
    saved = json.loads(text)
    model = settings.cohere_embedding_model if settings.embedding_provider == "cohere" else settings.embedding_model
    assert saved["config"]["embedding"] == f"{settings.embedding_provider}/{model}"
    assert len(saved["dataset"]["sha256_12"]) == 12


def test_compare_reports_and_config_diff() -> None:
    a = {"kind": "generation", "config": {"llm": "x", "min_score": 0.7},
         "metrics": {"answer_rate": 0.8, "hallucination_rate": None}, "latency": {"total": {"p95_ms": 900.0}}}
    b = {"kind": "generation", "config": {"llm": "x", "min_score": 0.6},
         "metrics": {"answer_rate": 0.9, "hallucination_rate": 0.1}, "latency": {"total": {"p95_ms": 1000.0}}}
    rows = {name: (va, vb, d) for name, va, vb, d in compare_reports(a, b)}
    assert rows["answer_rate"] == (0.8, 0.9, pytest.approx(0.1))
    assert rows["hallucination_rate"] == (None, 0.1, None)
    assert rows["total p95_ms"][2] == 100.0
    assert config_diff(a, b) == {"min_score": (0.7, 0.6)}
    with pytest.raises(ValueError):
        compare_reports(a, {**b, "kind": "retrieval"})