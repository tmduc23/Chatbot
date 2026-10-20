"""
Route /api/v1/applications (Phase 9). Route MỎNG: nhậ request -> gọi service -> trả schema.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import get_application_service, require_admin
from app.schemas.application import ApplicationCreate, ApplicationOut
from app.services.application_service import ApplicationService


router = APIRouter(prefix="/applications", tags=["applications"])


# Đặt ở POST cho chỉ POST cần admin
@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_admin)])
def create_application(body: ApplicationCreate, service: ApplicationService = Depends(get_application_service)):
    return service.create(body.id, body.name, body.description)


@router.get("", response_model=list[ApplicationOut])
def list_application(service: ApplicationService = Depends(get_application_service)):
    return service.list()

@router.get("/{application_id}", response_model=ApplicationOut)
def get_application(application_id: str, service: ApplicationService = Depends(get_application_service)):
    return service.get(application_id)