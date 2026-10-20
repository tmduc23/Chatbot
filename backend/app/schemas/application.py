"""
Pydantic schema cho /api/v1/applications 
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

"""
SQLAlchemy là dạng lưu trữ, Pydantic là dạng trả cho client, còn from_attributes là cây cầu nối 2 dạng.
"""

#slug: chữ thường, số, "-" hoặc "_", 2-50 ký tự, bắt đầu bằng chữ/số. Vd: "cv", "projects", "blog-2026"
APPLICATION_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,49}$"


class ApplicationCreate(BaseModel):
    id: str = Field(pattern=APPLICATION_ID_PATTERN, examples=["cv"])
    name: str = Field(min_length=1, max_length=200, examples=["CV của Đức"])
    description: str | None = Field(default=None, max_length=2000)


class ApplicationOut(BaseModel):
    # from_attributes: cho phép tạo schema thẳng từ object SQLAlchemy (ApplicationOut.model_validate(app))
    # chuyển từ tìm app["id"] (dict) -> app.id (thuộc tính)

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime