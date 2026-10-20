"""
Route /api/v1/documents (Phase 9). Toàn bộ là API quản trị -> cần X-API-Key (require_admin).
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.api.deps import get_document_service, require_admin
from app.schemas.document import DocumentDetail, DocumentOut
from app.services.document_service import DocumentService

# áp cho mọi route trong file
router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_admin)])


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    application_id: str = Form(...),
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    # multipart/form-data: application là 1 ô form, file là 1 ô file (cần thư viện python-multipart)
    return service.ingest_upload(application_id, file.file, file.filename or "upload")

@router.get("", response_model=list[DocumentOut])
def list_documents(
    application_id: str = Query(...), 
    service: DocumentService = Depends(get_document_service)):
    return service.list_documents(application_id)

# model_validate: dữ liệu bên ngoài (dict hoặc object) vào schema
# model_dump: chuyển sang dạng dict gán cho từng trường cụ thể
@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: uuid.UUID,
    application_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
):
    doc = service.get_document(application_id, document_id)
    return DocumentDetail(
        **DocumentOut.model_validate(doc).model_dump(),
        chunk_count=service.count_chunks(application_id, document_id)      
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: uuid.UUID,
    application_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
):
    service.delete_document(application_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)