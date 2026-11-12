"""
MỤC ĐÍCH: Unit test cho CohereEmbeddingProvider — KHÔNG gọi Cohere thật (không tốn quota, không cần mạng).

Client được thay bằng stub có hàm embed(...) trả vector giả, để kiểm tra logic của mình:
  - chunk dùng input_type="search_document", câu hỏi dùng "search_query";
  - luôn gửi output_dimension=EMBEDDING_DIM (1536) và embedding_types=["float"];
  - cắt batch 96 text/lần; vector được chuẩn hoá L2;
  - retry khi HTTP 429; lỗi khác -> EmbeddingError;
  - sai số lượng / sai số chiều -> EmbeddingError (gợi ý model embed-v4.0 / v5.0);
  - factory chọn đúng class và báo lỗi khi thiếu COHERE_API_KEY.
"""
import math
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.models import EMBEDDING_DIM
from app.rag.embeddings import cohere_embeddings as cohere_module
from app.rag.embeddings.cohere_embeddings import CohereEmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider


class _HttpError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _StubClient:
    """Giả lập cohere.ClientV2: ghi lại mọi lần gọi; trả vector [3, 4, 0, 0, ...] cho mỗi text."""

    def __init__(self, errors: list[Exception] | None = None, dim: int = EMBEDDING_DIM, drop: int = 0):
        self.calls: list[dict] = []
        self.errors = list(errors or [])
        self.dim = dim
        self.drop = drop

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        vec = [3.0, 4.0] + [0.0] * (self.dim - 2)
        n = len(kwargs["texts"]) - self.drop
        return SimpleNamespace(embeddings=SimpleNamespace(float_=[vec] * n))


def _provider(client: _StubClient, model: str = "embed-v5.0-pro") -> CohereEmbeddingProvider:
    return CohereEmbeddingProvider(api_key="unused", model=model, dimension=EMBEDDING_DIM, client=client)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cohere_module.time, "sleep", lambda _s: None)


def test_documents_use_search_document_and_dimension() -> None:
    client = _StubClient()
    vectors = _provider(client).embed_documents(["chunk a", "chunk b"])

    assert len(vectors) == 2 and len(vectors[0]) == EMBEDDING_DIM
    call = client.calls[0]
    assert call["model"] == "embed-v5.0-pro"
    assert call["input_type"] == "search_document"
    assert call["output_dimension"] == EMBEDDING_DIM
    assert call["embedding_types"] == ["float"]
    assert call["texts"] == ["chunk a", "chunk b"]


def test_query_uses_search_query() -> None:
    client = _StubClient()
    vector = _provider(client).embed_query("Bạn ấy biết Docker không?")

    assert len(vector) == EMBEDDING_DIM
    assert client.calls[0]["input_type"] == "search_query"
    assert client.calls[0]["texts"] == ["Bạn ấy biết Docker không?"]


def test_vectors_are_l2_normalized() -> None:
    vector = _provider(_StubClient()).embed_query("q")
    assert vector[:2] == pytest.approx([0.6, 0.8])               # (3, 4) / 5
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0)


def test_batches_of_96() -> None:
    client = _StubClient()
    vectors = _provider(client).embed_documents([f"t{i}" for i in range(200)])

    assert len(vectors) == 200
    assert [len(c["texts"]) for c in client.calls] == [96, 96, 8]
    assert client.calls[1]["texts"][0] == "t96"                   # giữ đúng thứ tự


def test_empty_list_makes_no_call() -> None:
    client = _StubClient()
    assert _provider(client).embed_documents([]) == []
    assert client.calls == []


def test_retries_on_429_then_succeeds() -> None:
    client = _StubClient(errors=[_HttpError(429), _HttpError(503)])
    assert len(_provider(client).embed_query("q")) == EMBEDDING_DIM
    assert len(client.calls) == 3


def test_gives_up_after_max_retries() -> None:
    client = _StubClient(errors=[_HttpError(429)] * 10)
    with pytest.raises(EmbeddingError):
        _provider(client).embed_query("q")
    assert len(client.calls) == CohereEmbeddingProvider.MAX_RETRIES


def test_non_retryable_error_raises_immediately() -> None:
    client = _StubClient(errors=[_HttpError(401)])               # sai key -> không retry
    with pytest.raises(EmbeddingError):
        _provider(client).embed_query("q")
    assert len(client.calls) == 1


def test_wrong_dimension_raises_with_hint() -> None:
    client = _StubClient(dim=1024)                                # vd. model v3 bỏ qua output_dimension
    with pytest.raises(EmbeddingError, match="embed-v5.0"):
        _provider(client, model="embed-multilingual-v3.0").embed_query("q")


def test_wrong_count_raises() -> None:
    with pytest.raises(EmbeddingError):
        _provider(_StubClient(drop=1)).embed_documents(["a", "b"])


def test_factory_builds_cohere() -> None:
    provider = get_embedding_provider(Settings(
        embedding_provider="cohere", cohere_embedding_model="embed-v4.0", cohere_api_key="test-key",
    ))
    assert isinstance(provider, CohereEmbeddingProvider)
    assert provider.model == "embed-v4.0"                        # lấy từ COHERE_EMBEDDING_MODEL
    assert provider.dimension == EMBEDDING_DIM


def test_factory_cohere_requires_key() -> None:
    with pytest.raises(EmbeddingError):
        get_embedding_provider(Settings(embedding_provider="cohere", cohere_api_key=None,
                                        cohere_embed_api_key=None))
