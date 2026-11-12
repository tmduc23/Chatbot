"""
Pydantic schema cho /api/v1/document.py
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import DocumentStatus



class DocumentOut(BaseModel):
    model_config =  ConfigDict(from_attributes=True)

    id: uuid.UUID
    application_id: str
    filename: str
    file_type: str
    status: DocumentStatus
    error_message: str | None
    attempts: int                   # đã xử lý mấy lần
    created_at: datetime

    # KHÔNG trả storage_path (đường dẫn trên server) và file_hash: client không cần, lộ cấu trúc server

class DocumentDetail(DocumentOut):
    chunk_count: int
    