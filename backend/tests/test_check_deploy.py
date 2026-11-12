"""
MỤC ĐÍCH: Test scripts/check_deploy.py (Phase 13) — không cần VPS, không cần Caddy.

  - Kiểm tra phía proxy (https, HSTS, tên server, cổng): dùng `http` giả trả về đúng thứ cần thử.
  - Kiểm tra phía backend: chạy THẬT trên create_app(...) qua TestClient, với cấu hình production
    -> chứng minh backend cấu hình đúng thì đạt, và cấu hình hở (dev) thì bị bắt.
"""
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.deps import get_embedder, get_llm_provider, get_reranker_provider
from app.core.config import Settings, get_settings
from app.core.config_check import security_warnings
from app.main import create_app
from app.models import EMBEDDING_DIM
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.rag.llm.fake import FakeLLM
from app.rag.reranking.fake import FakeReranker

_script = Path(__file__).resolve().parents[1] / "scripts" / "check_deploy.py"
_spec = importlib.util.spec_from_file_location("check_deploy", _script)
cd = importlib.util.module_from_spec(_spec)
sys.modules["check_deploy"] = cd          # dataclass cần module có trong sys.modules
_spec.loader.exec_module(cd)

BASE = "https://cv.example.com"


def _fake_http(routes: dict[tuple[str, str], "cd.Response"]):
    default = cd.Response(404, {}, "Not found")
    return lambda method, url, headers, body: routes.get((method, url), default)


# ---------------- phía proxy (Caddy) ----------------

def test_https_redirect() -> None:
    good = _fake_http({("GET", "http://cv.example.com/"): cd.Response(308, {"location": f"{BASE}/"}, "")})
    assert cd.check_https_redirect(good, BASE).ok
    plain_http = _fake_http({("GET", "http://cv.example.com/"): cd.Response(200, {}, "<html>")})
    assert not cd.check_https_redirect(plain_http, BASE).ok           # vẫn phục vụ http, không chuyển
    assert not cd.check_https_redirect(good, "http://cv.example.com").ok   # URL kiểm tra phải là https


def test_hsts_and_server_banner() -> None:
    caddy = _fake_http({("GET", f"{BASE}/health"): cd.Response(200, {"strict-transport-security": "max-age=31536000"}, "")})
    raw = _fake_http({("GET", f"{BASE}/health"): cd.Response(200, {"server": "uvicorn"}, "")})
    assert cd.check_hsts(caddy, BASE).ok and not cd.check_hsts(raw, BASE).ok
    assert cd.check_no_server_banner(caddy, BASE).ok and not cd.check_no_server_banner(raw, BASE).ok


def test_ports_closed() -> None:
    assert cd.check_ports_closed(lambda host, port: False, BASE).ok
    check = cd.check_ports_closed(lambda host, port: host == "cv.example.com" and port == 5432, BASE)
    assert not check.ok and "5432" in check.detail


def test_hidden_paths_detects_leaks() -> None:
    leaky = _fake_http({("GET", f"{BASE}/metrics"): cd.Response(200, {}, "# HELP http_requests_total ...")})
    check = cd.check_hidden_paths(leaky, BASE)
    assert not check.ok and "/metrics" in check.detail
    # frontend trả 200 cho trang "không tìm thấy" của nó: không phải rò rỉ
    spa = _fake_http({("GET", f"{BASE}/docs"): cd.Response(200, {}, "<html>404 | This page could not be found")})
    assert cd.check_hidden_paths(spa, BASE).ok


def test_chat_check_reads_answer() -> None:
    ok = _fake_http({("POST", f"{BASE}/api/v1/chat"): cd.Response(200, {}, '{"answer": "x", "sources": [{}]}')})
    assert cd.check_chat(ok, BASE, "cv", "q").ok
    limited = _fake_http({("POST", f"{BASE}/api/v1/chat"): cd.Response(429, {}, '{"detail": "slow down"}')})
    assert not cd.check_chat(limited, BASE, "cv", "q").ok


# ---------------- phía backend (chạy thật) ----------------

def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        app_env="production", storage_dir=str(tmp_path / "storage"), tokenizer_encoding="whitespace",
        embedding_provider="fake", reranker_provider="fake", llm_provider="fake",
        auth_enabled=True, jwt_secret=SecretStr("j" * 40), admin_api_key=None,
        rate_limit_enabled=False, answer_cache_enabled=False, background_ingestion=False,
        cors_origins="https://cv.example.com", expose_api_docs=None, metrics_token=SecretStr("m" * 32),
    )
    base.update(overrides)
    return Settings(**base)


def _testclient_http(settings: Settings) -> Iterator:
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbeddingProvider(EMBEDDING_DIM)
    app.dependency_overrides[get_reranker_provider] = lambda: FakeReranker()
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLM()
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as client:
        def send(method, url, headers, body):
            r = client.request(method, url.removeprefix(BASE), headers=headers, content=body)
            return cd.Response(r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.text)
        yield send


@pytest.fixture
def prod_http(tmp_path: Path) -> Iterator:
    yield from _testclient_http(_settings(tmp_path))


def test_production_backend_passes_all_backend_checks(prod_http) -> None:
    report = cd.run_checks(prod_http, BASE, behind_proxy=False)
    failed = [(c.name, c.detail) for c in report.checks if not c.ok]
    assert failed == []
    assert {c.name for c in report.checks} >= {"/health", "/ready (backend thấy database)",
                                                "API quản trị đòi đăng nhập", "CORS chặn trang web lạ"}


def test_dev_config_is_caught(tmp_path: Path) -> None:
    # Cấu hình dev lỡ đưa lên server: /docs mở, không đăng nhập, không ADMIN_API_KEY -> API quản trị mở toang
    for http in _testclient_http(_settings(tmp_path, app_env="development", auth_enabled=False)):
        names = {c.name for c in cd.run_checks(http, BASE, behind_proxy=False).checks if not c.ok}
    assert names == {"/docs, /openapi.json, /metrics bị ẩn", "API quản trị đòi đăng nhập"}


def test_cors_wildcard_is_caught(tmp_path: Path) -> None:
    for http in _testclient_http(_settings(tmp_path, cors_origins="*")):
        assert not cd.check_cors_rejects_other_sites(http, BASE).ok


def test_admin_probe_is_harmless(prod_http, tmp_path: Path) -> None:
    # DELETE 1 document id NGẪU NHIÊN: cấu hình đúng -> 401; cấu hình hở -> chỉ 404, không xoá được gì
    assert cd.check_admin_needs_login(prod_http, BASE).detail == "401"
    for http in _testclient_http(_settings(tmp_path, app_env="development", auth_enabled=False)):
        assert cd.check_admin_needs_login(http, BASE).detail == "404"


def test_production_warnings_for_public_site(tmp_path: Path) -> None:
    joined = " | ".join(security_warnings(_settings(tmp_path, cors_origins="http://cv.example.com",
                                                    expose_api_docs=True)))
    assert "http:// origin" in joined and "EXPOSE_API_DOCS=true" in joined
