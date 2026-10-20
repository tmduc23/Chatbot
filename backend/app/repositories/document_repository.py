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
    - **`delete_document` trả `bool`**: 
    Phase 9 dùng giá trị này để trả 404 khi `False`. 
    Câu `DELETE ... WHERE application_id = ? AND id = ?` với id của app khác đơn giản là xoá **0 dòng**, không có lỗi nào lộ ra.
    - Chunks của document tự xoá theo nhờ `ON DELETE CASCADE` trên khoá ngoại 2 cột mới.
"""

import uuid
from sqlalchemy import select, delete, func
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus

# ---Phase 8: đọc. xóa document - luôn theo CẶP (application_id, document_id) --- 
# Chỉ biết document_id là CHƯA ĐỦ: id đến từ bên ngoài (URL của API ở phase 9), người gọi
# có thể đoán/chép id của application_id khác. Lọc thêm application_id thì id "lạc" không thấy gì


# Tìm document cùng hash để chống trùng 
def get_by_hash(db: Session, application_id: str, file_hash: str) -> Document | None:
    return db.scalars(
        select(Document).where(
            Document.application_id == application_id,
            Document.file_hash == file_hash
        )
    ).first()

# Tạo document rồi flush để có doc.id, chưa commit
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


# Lấy các chunk của một document trong một application cụ thể, sắp xếp theo chunk_index, db.scalars lấy trực tiếp Document
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

def list_documents(db: Session, application_id: str) -> list[Document]:
    return list(db.scalars(
        select(Document)
        .where(Document.application_id == application_id)
        .order_by(Document.created_at, Document.filename)
    ))

def get_document(db: Session, application_id: str, document_id: uuid.UUID) -> Document | None:
    return db.scalars(
        select(Document).where(Document.application_id == application_id, Document.id == document_id)
    ).first()

# Phase 9: số chunk của 1 document (hiện ở GET /documents/{id})
def count_chunks(db: Session, application_id: str, document_id: uuid.UUID) -> int:
    return db.scalar(
        select(func.count(Chunk.id))
        .where(Chunk.application_id == application_id, Chunk.document_id == document_id)
    ) or 0

def delete_document(db: Session, application_id: str, document_id: uuid.UUID) -> bool:
    """Xoá document (chunks xoá theo nhờ ON DELETE CASCADE). True nếu có dòng bị xoá.
    Không commit — service quyết định. Không xoá file gốc trong storage (việc của service)."""
    result = db.execute(
        delete(Document).where(Document.application_id == application_id, Document.id == document_id)
    )
    return result.rowcount > 0