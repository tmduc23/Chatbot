"""
Route /api/v1/applications (Phase 9). Route MỎNG: nhậ request -> gọi service -> trả schema.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import get_access, get_application_service
from app.schemas.application import ApplicationCreate, ApplicationOut
from app.services.access import AccessControl

from app.services.application_service import ApplicationService


router = APIRouter(prefix="/applications", tags=["applications"])


# Phase 12b: mỗi route gọi access.require_...() TRƯỚC khi làm gì khác
@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
def create_application(
    body: ApplicationCreate,
    service: ApplicationService = Depends(get_application_service),
    access: AccessControl = Depends(get_access),
):
    access.require_superadmin()
    return service.create(body.id, body.name, body.description, is_public=body.is_public)

def create_application(body: ApplicationCreate, service: ApplicationService = Depends(get_application_service)):
    return service.create(body.id, body.name, body.description)


@router.get("", response_model=list[ApplicationOut])
def list_application(
    service: ApplicationService = Depends(get_application_service),
    access: AccessControl = Depends(get_access),
):
    # khách: chỉ thấy app công khai; user: + app mình là thành viên; admin: tất cả
    return access.filter_visible(service.list())

@router.get("/{application_id}", response_model=ApplicationOut)
def get_application(
    application_id: str,
    service: ApplicationService = Depends(get_application_service),
    access: AccessControl = Depends(get_access),
):
    access.require_chat(application_id)
    return service.get(application_id)
