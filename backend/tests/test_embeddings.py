"""
MỤC ĐÍCH: Unit test cho tầng embedding — KHÔNG gọi API thật.

- l2_normalize và FakeEmbeddingProvider: test trực tiếp.
- GeminiEmbeddingProvider: thay client thật bằng một "stub" giả lập,
  để kiểm tra logic của mình (chia batch, task_type, normalize, retry)
  mà không toosn quota và không cần mạng.
"""


import math
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.models import EMBEDDING_DIM
from app.rag.embeddings import get_embedding_provider, l2_normalize
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.embeddings.gemini import GeminiEmbeddingProvider



def _cosine(a: list[float], b: list[float]) -> float:
    return sum (x * y for x, y in zip(a, b))               # vector đã normalize -> dot_product = cosine


# l2 normalize
def test_l2_normalizze_gives_unit_length() -> None:
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)


def test_l2_normalize_zero_vector_is_unchanged() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# Fake provider, cùng text cho cùng vector
def test_fake_provider_is_deterministic_and_sized() -> None:
    p = FakeEmbeddingProvider(EMBEDDING_DIM)
    a, b = p.embed_query("Docker and Kubernetes"), p.embed_query("Docker and Kubernetes")
    assert a == b
    assert len(a) == EMBEDDING_DIM

# text liên quan có cosine cao hơn 
def test_fake_provider_similar_text_scores_higher() -> None:
    p = FakeEmbeddingProvider(EMBEDDING_DIM)
    q = p.embed_query("Docker Compose deployment")
    related = p.embed_documents(["Experienced with Docker Compose deployment"])[0]
    unrelated = p.embed_documents(["Playing guitar and jazz music"])[0]
    assert _cosine(q, related) > _cosine(q, unrelated)


# Gemini provier (với stub client)
class _StubModels:
    def __init__(self, fail_first_width: int | None = None):
        self.calls: list[tuple[str, list[str], object]] = []
        self._fail_code = fail_first_width

    def embed_content(self, model: str, contents: list[str], config: object) -> SimpleNamespace:
        if self._fail_code is not None:
            code, self._fail_code = self._fail_code, None
            err = RuntimeError("rate limited")
            err.code = code         # type: ignore[attr-defined]
            raise err
        self.calls.append((model, list(contents), config))
        vec = [3.0, 4.0] + [0.0] * (EMBEDDING_DIM - 2)   # chưa normalize
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec) for _ in contents])


def _gemini(stub_models: _StubModels) -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider(api_key="test", model="gemini-embedding-001",
                                   dimension=EMBEDDING_DIM, client=SimpleNamespace(models=stub_models))


# chia text thành các batch
def test_gemini_batches_and_normalize() -> None:
    stub = _StubModels()
    vectors = _gemini(stub).embed_documents([f"chunk {i}" for i in range(250)])

    assert len(vectors) == 250
    assert [len(c[1]) for c in stub.calls] == [100, 100, 50]     # chia batch 100
    config = stub.calls[0][2]
    assert config.task_type == "RETRIEVAL_DOCUMENT"
    assert config.output_dimensionality == EMBEDDING_DIM
    assert vectors[0][:2] == pytest.approx([0.6, 0.8]) 

# Kiểm tra Gemini dùng đúng query không
def test_gemini_query_uses_query_task_type() -> None:
    stub = _StubModels()
    _gemini(stub).embed_query("Duc biết Docker không?")
    assert stub.calls[-1][2].task_type == "RETRIEVAL_QUERY"

# Kiểm tra retry, 429 thì thử lại
def test_gemini_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.rag.embeddings.gemini.time.sleep", lambda s: None)  # không chờ thật
    stub = _StubModels(fail_first_width=429)
    assert len(_gemini(stub).embed_query("hello")) == EMBEDDING_DIM
    assert len(stub.calls) == 1

# Kiểm tra báo 400 thì có retry không, 400 thì raise luôn
def test_gemini_non_retryable_error_raises_embedding_error() -> None:
    stub = _StubModels(fail_first_width=400)
    with pytest.raises(EmbeddingError):
        _gemini(stub).embed_query("hello")

# ---------- Factory ----------
# chọn gemini mà key rỗng thì raise EmbeddingError
def test_factory_requires_api_key_for_gemini() -> None:
    with pytest.raises(EmbeddingError):
        get_embedding_provider(Settings(embedding_provider="gemini", gemini_api_key=""))


def test_factory_returns_fake() -> None:
    assert isinstance(get_embedding_provider(Settings(embedding_provider="fake")), FakeEmbeddingProvider)