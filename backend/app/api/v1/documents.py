"""
Route /api/v1/documents (Phase 9). Toàn bộ là API quản trị -> cần X-API-Key (require_admin).
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.api.deps import RateLimit, get_access, get_document_service
from app.core.config import Settings, get_settings
from app.schemas.document import DocumentDetail, DocumentOut
from app.services.access import AccessControl
from app.services.document_service import DocumentService

# Phase 12b: cần vai trò editor trên ĐÚNG application_id của request (hoặc superadmin / X-API-Key).
# AUTH_ENABLED=false: như Phase 9, chỉ cần X-API-Key.
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED,
             responses={202: {"model": DocumentOut, "description": "Đã nhận, worker đang xử lý (Phase 12a)"}},
             dependencies=[Depends(RateLimit("upload"))])   # Phase 12c
def upload_document(
    response: Response,
    application_id: str = Form(...),
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
    access: AccessControl = Depends(get_access),
):
    access.require_editor(application_id)   # kiểm quyền TRƯỚC khi đọc/lưu file
    # multipart/form-data: application là 1 ô form, file là 1 ô file (cần thư viện python-multipart)
    background = settings.background_ingestion
    doc = service.ingest_upload(application_id, file.file, file.filename or "upload", background=background)
    if background:
        # 202 Accepted: "đã nhận, CHƯA xong". Client hỏi lại GET /documents/{id} tới khi READY/FAILED
        response.status_code = status.HTTP_202_ACCEPTED
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    application_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
    access: AccessControl = Depends(get_access),
):
    access.require_editor(application_id)
    return service.list_documents(application_id)


# model_validate: dữ liệu bên ngoài (dict hoặc object) vào schema
# model_dump: chuyển sang dạng dict gán cho từng trường cụ thể
@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: uuid.UUID,
    application_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
    access: AccessControl = Depends(get_access),
):
    access.require_editor(application_id)
    doc = service.get_document(application_id, document_id)
    return DocumentDetail(
        **DocumentOut.model_validate(doc).model_dump(),
        chunk_count=service.count_chunks(application_id, document_id),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: uuid.UUID,
    application_id: str = Query(...),
    service: DocumentService = Depends(get_document_service),
    access: AccessControl = Depends(get_access),
):
    access.require_editor(application_id)
    service.delete_document(application_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)