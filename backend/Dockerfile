# Chọn môi trường Python 3.11 trên linux sm
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE=1	Không tạo các file bytecode .pyc khi import module Python.
# PYTHONUNBUFFERED=1	Xuất log Python ngay lập tức, thuận tiện khi xem log Docker.
# PYTHONPATH=/app	Thêm /app vào đường dẫn tìm kiếm module Python.
# TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache	Chỉ định thư mục cache dữ liệu tokenizer của tiktoken.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

# /app: thư mục làm việc của backend trên dockers
WORKDIR /app                                
COPY requirements.txt .

# cài thư viện
RUN pip install --no-cache-dir -r requirements.txt
# Tải sẵn bộ từ vựng của tokenizer lúc BUILD -> lúc chạy không cần mạng
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]