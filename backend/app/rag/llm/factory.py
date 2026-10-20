"""
MỤC ĐÍCH: Chọn LLM theo cấu hình LLM_PROVIDER (gemini | openrouter | fake).
"""
from app.core.config import Settings, get_settings
from app.rag.llm.base import LLMProvider

PROVIDERS = "gemini | openrouter | fake"


def get_llm(settings: Settings | None = None) -> LLMProvider:
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
                         temperature=s.llm_temperature, max_output_tokens=s.llm_max_output_tokens)
    if name == "openrouter":
        if s.openrouter_api_key is None or not s.openrouter_api_key.get_secret_value():
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        from app.rag.llm.openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(api_key=s.openrouter_api_key.get_secret_value(), model=s.llm_model,
                               base_url=s.openrouter_base_url, temperature=s.llm_temperature,
                               max_output_tokens=s.llm_max_output_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}' (use: {PROVIDERS})")