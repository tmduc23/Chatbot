"""
MỤC ĐÍCH (Phase 12d): MONITORING — các con số Prometheus đọc từ GET /metrics mỗi 15 giây.

Log trả lời "chuyện gì đã xảy ra với request X". Metrics trả lời "hệ thống đang khoẻ không":
bao nhiêu request/giây, p95 latency, tỉ lệ từ chối trả lời, tỉ lệ trúng cache, LLM lỗi bao nhiêu lần...

3 loại metric:
  Counter   : chỉ tăng (tổng request, tổng lỗi). Prometheus tự tính "tăng bao nhiêu mỗi giây" bằng rate().
  Histogram : phân phối thời gian, chia theo "bucket" -> tính được p50/p95/p99 (spec mục 24: không chỉ average).
  Gauge     : lên xuống được (số document đang PENDING).

QUAN TRỌNG — "cardinality": mỗi tổ hợp nhãn (label) là 1 chuỗi số riêng trong Prometheus.
  route="/api/v1/documents/{document_id}"  -> 1 chuỗi cho mọi document           ✅
  path="/api/v1/documents/8f3a..."          -> 1 chuỗi MỖI document, phình vô hạn  ❌
Không bao giờ dùng id, câu hỏi, email làm nhãn.
"""
from prometheus_client import Counter, Gauge, Histogram

# Bucket theo giây. RAG chậm (LLM miễn phí 5-30s) nên cần bucket lớn hơn mặc định
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                         buckets=LATENCY_BUCKETS)

RAG_STAGE_LATENCY = Histogram("rag_stage_duration_seconds", "Thời gian từng bước RAG", ["stage"],
                              buckets=LATENCY_BUCKETS)
CHAT_ANSWERS = Counter("chat_answers_total", "Câu trả lời /chat", ["application", "outcome", "cached"])
LLM_TOKENS = Counter("llm_tokens_total", "Token LLM đã dùng", ["direction"])
UPSTREAM_ERRORS = Counter("upstream_errors_total", "Lỗi từ Gemini/Cohere/OpenRouter", ["error"])
RATE_LIMITED = Counter("rate_limited_total", "Request bị chặn bởi rate limit", ["scope"])
ANSWER_CACHE = Counter("answer_cache_requests_total", "Tra cứu cache câu trả lời", ["result"])
DOCUMENTS = Gauge("documents", "Số document theo trạng thái (đọc từ DB lúc Prometheus hỏi)", ["status"])
