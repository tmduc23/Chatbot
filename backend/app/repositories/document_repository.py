"""
MỤC ĐÍCH: Mọi truy vấn database liên quan đến "documents" và "chunks".

Quy tắc bảo mật: mọi hàm tìm kiếm đều nhận "application_id" và đưa nó
vào mệnh đề WHERE -> không bao giờ độc nhầm dữ liệu của application khác.


                 Document Repository
                         │
             ┌───────────┴────────────────────┐
             │                                │
        documents                           chunks
             │                                │
     ┌───────┴───────┐                ┌───────┴────────┐──────────────────────┐
     │               │                │                │                      │
   get_by_hash()   create()        add_chunks()   list_chunks()         list_chunks_without_embedding

Repository có nhiệm vụ:
    Tìm document đã tồn tại
    Tạo document mới
    Thêm các chunk vào database
    Lấy các chunk của một document
    List các chunk chưa có embedding
"""

import uuid
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus


# Tìm document đã tồn tại
def get_by_hash(db: Session, application_id: str, file_hash: str) -> Document | None:
    return db.scalars(
        select(Document).where(
            Document.application_id == application_id,
            Document.file_hash == file_hash
        )
    ).first()

# Tạo document
def create(
        db: Session, *, application_id: str, filename:str, file_type: str,
        storage_path: str, file_hash: str, status: DocumentStatus
) -> Document:
    doc = Document(
        application_id = application_id, filename=filename, file_type=file_type,
        storage_path=storage_path, file_hash=file_hash, status=status,
    )
    db.add(doc)
    db.flush()      # để có doc.id ngay
    return doc

# Thêm nhiều chunk
def add_chunks(db: Session, chunks: list[Chunk]) -> None:
    db.add_all(chunks)
    db.flush()


# Lấy các chunk của một document trong một application cụ thể, db.scalars lấy trực tiếp Document
def list_chunks(db: Session, application_id: str, document_id: uuid.UUID) -> list[Chunk]:
    return list(db.scalars(
        select(Chunk)
        .where(Chunk.application_id == application_id, Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    ))

# Các chunk chưa có embedding - dùng embed để bổ sung

def list_chunks_without_embedding(db: Session, application_id: str) -> list[Chunk]:
    return list(db.scalars(
        select(Chunk)
        .where(Chunk.application_id == application_id, Chunk.embedding.is_(None))
    ))