"""
MỤC ĐÍCH: Nghiệp vụ quản lý application (Phase 9: POST/GET /api/v1/applications).

Mỏng, nhưng vẫn tách khỏi route: route không tự commit, không tự bắt IntegrityError
"""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationAlreadyExistsError
from app.models import Application
from app.repositories import application_repository as app_repo


class ApplicationService:
    def __init__(self, db: Session):
        self.db = db

    def create(self, application_id: str, name: str, description: str | None = None) -> Application:
        if app_repo.get(self.db, application_id) is not None:
            raise ApplicationAlreadyExistsError(f"Application: '{application_id}' already exists")
        try:
            app = app_repo.create(self.db, application_id, name, description)
            self.db.commit()
        except IntegrityError:
            # 2 request tạo cùng id gần như đồng thời: cả 2 qua bước kiểm tra trên,
            # khóa chính chặn request thứ 2 -> vẫn trả 409 thay vì 500
            self.db.rollback()
            raise ApplicationAlreadyExistsError(f"Application '{application_id}' already exists") from None
        return app

    def list(self) -> list[Application]:
        return app_repo.list_all(self.db)


    # require: gom việc lấy "application, không có thì báo lỗi"
    def get(self, application_id: str) -> Application:
        return app_repo.require(self.db, application_id)