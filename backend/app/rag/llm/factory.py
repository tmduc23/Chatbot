"""
MỤC ĐÍCH: Chọn LLM theo cấu hình LLM_PROVIDER (gemini | openrouter | groq | fake).
"""
from app.core.config import Settings, get_settings
from app.rag.llm.base import LLMProvider
from app.rag.clients import make_gemini_client, make_openai_client


PROVIDERS = "gemini | openrouter | groq | fake"


def get_llm(settings: Settings | None = None, role: str = "answer") -> LLMProvider:
    """role (answer | grounding | judge) chỉ dùng để chọn key OpenRouter; model lấy từ settings.llm_model."""
    s = settings or get_settings()
    name = s.llm_provider.lower()
    if name == "fake":
        from app.rag.llm.fake import FakeLLM
        return FakeLLM()
    
    if name == "gemini":
        if s.gemini_api_key is None or not s.gemini_api_key.get_secret_value():
            raise ValueError("GEMINI_API_KEY is not set in .env")
        from app.rag.llm.gemini import GeminiLLM
        return GeminiLLM(api_key=s.gemini_api_key.get_secret_value(), model=s.llm_model,
                         temperature=s.llm_temperature, max_output_tokens=s.llm_max_output_tokens,
                         client=make_gemini_client(s.gemini_api_key.get_secret_value(), s.llm_timeout_seconds))
    if name == "openrouter":
        api_key = s.openrouter_key(role)
        if api_key is None:
            raise ValueError(f"OPENROUTER_API_KEY (or OPENROUTER_{role.upper()}_API_KEY) is not set in .env")
        from app.rag.llm.openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(api_key=api_key, model=s.llm_model,
                               base_url=s.openrouter_base_url, temperature=s.llm_temperature,
                               max_output_tokens=s.llm_max_output_tokens,
                               client=make_openai_client(api_key, s.openrouter_base_url, s.llm_timeout_seconds))

    if name == "groq":
        if s.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set in .env")
        api_key = s.groq_api_key.get_secret_value().strip()
        from app.rag.llm.openai_compat import OpenAICompatLLM   # Groq cũng theo chuẩn OpenAI: chỉ khác base_url + key
        return OpenAICompatLLM(api_key=api_key, model=s.llm_model,
                               base_url=s.groq_base_url, temperature=s.llm_temperature,
                               max_output_tokens=s.llm_max_output_tokens,
                               client=make_openai_client(api_key, s.groq_base_url, s.llm_timeout_seconds))

    if name == "nvidia":
        if s.nvidia_api_key is None:
            raise ValueError("NVIDIA_API_KEY is not set in .env")
        api_key = s.nvidia_api_key.get_secret_value().strip()
        from app.rag.llm.openai_compat import OpenAICompatLLM   # NIM cũng theo chuẩn OpenAI: chỉ khác base_url + key
        return OpenAICompatLLM(api_key=api_key, model=s.llm_model,
                               base_url=s.nvidia_base_url, temperature=s.llm_temperature,
                               max_output_tokens=s.llm_max_output_tokens,
                               client=make_openai_client(api_key, s.nvidia_base_url, s.llm_timeout_seconds))

    raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}' (use: {PROVIDERS})")

    