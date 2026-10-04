import uuid
from sqlalchemy import String, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base, CreatedAtMixin


class User(CreatedAtMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    # hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)