FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Tải sẵn bộ từ vựng của tokenizer lúc BUILD -> lúc chạy không cần mạng
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]