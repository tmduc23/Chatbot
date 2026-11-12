"""
MỤC ĐÍCH (Phase 12a): tạo client SDK của Gemini / Cohere ở MỘT chỗ, luôn có TIMEOUT.

Không đặt timeout = nếu server bên kia treo (mạng chập chờn, provider quá tải), lời gọi có thể
chờ RẤT lâu (google-genai mặc định không giới hạn; Cohere 300s). Request /chat treo theo, giữ
1 luồng của server và 1 connection DB. Có timeout -> lỗi nhanh, retry của provider xử lý tiếp.

Thời gian chờ TỐI ĐA của 1 lời gọi = timeout x số lần thử + tổng thời gian chờ giữa các lần.
Vd. embedding Cohere: 5 lần x 60s + (4+8+16+32)s = 6 phút. Vì vậy worker (chạy nền) chịu được,
còn /chat thì nên để timeout nhỏ hơn (PROVIDER_TIMEOUT_SECONDS).
"""

from typing import Any


def make_gemini_client(api_key: str, timeout_s: float) -> Any:
    from google import genai  # import muộn: chỉ cần thư viện khi thật sự dùng Gemini
    from google.genai import types

    # HttpOptions.timeout tính bằng MILI giây
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


def make_cohere_client(api_key: str, timeout_s: float) -> Any:
    import cohere

    # max_retries=0: provider của mình đã tự retry 429/5xx -> tắt retry của SDK, tránh retry 2 tầng
    return cohere.ClientV2(api_key=api_key, timeout=timeout_s, max_retries=0)


def make_openai_client(api_key: str, base_url: str | None, timeout_s: float) -> Any:
    from openai import OpenAI

    # Dùng cho OpenRouter / Groq / NVIDIA NIM (đều là API kiểu OpenAI).
    # max_retries=0: OpenAICompatLLM đã tự xử lý lỗi -> không retry 2 tầng
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)