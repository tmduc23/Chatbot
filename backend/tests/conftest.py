"""Cấu hình chung cho mọi test."""
import pytest

# docker compose nạp .env thành BIẾN MÔI TRƯỜNG của container. Settings() đọc biến môi trường
# kể cả khi test truyền _env_file=None -> test vô tình dùng key / model thật. Xoá trước mỗi test.
REAL_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENROUTER_ANSWER_API_KEY", "OPENROUTER_GROUNDING_API_KEY", "OPENROUTER_JUDGE_API_KEY",
    "GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "GROUNDING_CHECK", "GROUNDING_PROVIDER", "GROUNDING_MODEL",
    "JUDGE_PROVIDER", "JUDGE_MODEL", "NVIDIA_API_KEY"
)


@pytest.fixture(autouse=True)
def _no_real_llm_env(monkeypatch) -> None:
    for name in REAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
