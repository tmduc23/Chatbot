"""
MỤC ĐÍCH: Kiểm tra cách chọn key Cohere khi có 2 key riêng (embed / rerank) và 1 key chung.

Quy tắc: key riêng được ưu tiên; để trống (hoặc chỉ có khoảng trắng) thì dùng COHERE_API_KEY.
Mọi test truyền ĐỦ 3 key (kể cả None) để không bị .env / biến môi trường của container lẫn vào.
"""
import pytest

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.reranking import get_reranker


def _settings(shared=None, embed=None, rerank=None, **extra) -> Settings:
    return Settings(cohere_api_key=shared, cohere_embed_api_key=embed, cohere_rerank_api_key=rerank, **extra)


def test_separate_keys_are_used_for_each_purpose() -> None:
    s = _settings(shared="shared", embed="key-embed", rerank="key-rerank")
    assert s.cohere_key("embed") == "key-embed"
    assert s.cohere_key("rerank") == "key-rerank"


def test_falls_back_to_shared_key() -> None:
    s = _settings(shared="shared", embed="key-embed")
    assert s.cohere_key("embed") == "key-embed"
    assert s.cohere_key("rerank") == "shared"


def test_blank_specific_key_falls_back() -> None:
    s = _settings(shared="shared", embed="   ")             # COHERE_EMBED_API_KEY= (để trống)
    assert s.cohere_key("embed") == "shared"


def test_no_key_at_all_returns_none() -> None:
    assert _settings().cohere_key("embed") is None
    assert _settings().cohere_key("rerank") is None


def test_unknown_purpose_is_a_bug() -> None:
    with pytest.raises(KeyError):
        _settings(shared="shared").cohere_key("chat")


def test_factories_receive_their_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    class _FakeClientV2:
        def __init__(self, api_key: str, **kwargs):
            seen.setdefault("first", api_key)
            seen["last"] = api_key

    import cohere
    monkeypatch.setattr(cohere, "ClientV2", _FakeClientV2)

    s = _settings(shared="shared", embed="key-embed", rerank="key-rerank",
                  embedding_provider="cohere", embedding_model="embed-v5.0-pro", reranker_provider="cohere")
    get_embedding_provider(s)
    assert seen["last"] == "key-embed"
    get_reranker(s)
    assert seen["last"] == "key-rerank"


def test_rerank_only_key_does_not_enable_embedding() -> None:
    s = _settings(rerank="key-rerank", embedding_provider="cohere")
    with pytest.raises(EmbeddingError, match="COHERE_EMBED_API_KEY"):
        get_embedding_provider(s)