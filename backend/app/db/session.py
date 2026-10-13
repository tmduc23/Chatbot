from collections.abc import Iterator
from functools import lru_cache
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import get_settings

# @lru_cache: tái sử dụng đối tượng đã tạo, tránh công việc khởi tạo lặp lại và dùng chung một đối tượng câu hình đã cache
# Engine đại diện cho SQLAIchemy Engine, quản lý việc kết nối tới database và connection pool

@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)

# SessionLocal là một factory tạo ra các session, mỗi session đại diện cho một kết nối tới database
# autoflush=False: không tự động flush dữ liệu vào database khi commit, 
# cần gọi session.flush() hoặc session.commit() để lưu dữ liệu
SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


# Tạo một session mới, dùng trong các API để thao tác với database
# yield db: trả về session cho API sử dụng, sau khi API xong sẽ tự động gọi db.close() để đóng session
def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()