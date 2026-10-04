from fastapi import FastAPI

# Middleware dùng để xử lý CORS

# CORS giúp backend kiểm soát: Frontend nào được phép gửi request tới backend
# Frontend chạy: http://localhost:3000
# Backend chạy:  http://localhost:8000
# Hai origin khác nhau nên cần CORS để cho phép Frontend gọi Backend.
from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router

# Function này có nhiệm vụ đọc các cấu hình từ .env và trả về một object của Settings
from app.core.config import get_settings

# Lấy toàn bộ cấu hình của ứng dụng
settings = get_settings()

# Tạo FastAPI application, đại diện cho Backend FastAPI
# Các API/router sau đó sẽ được gắn vào object trong app này
app = FastAPI(title="Multi-Application RAG", version="0.1.0")

# lấy danh sách các Frontend được phép gọi backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],        # cho phép tất cả các phương thức HTTP (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],        # cho phép  Header chứa các thông tin bổ sung của HTTP request.
)

# đăng ký tất cả các API trong health router vào FastAPI application
app.include_router(health_router)