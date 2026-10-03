from functools import lru_cache
from sqlalchemy import Engine, create_engine
from app.core.config import get_settings

# Engine đại diện cho SQLAIchemy Engine, quản lý việc kết nối tới database và connection pool

@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)