"""
MỤC ĐÍCH (Phase 12b): truy vấn bảng users và application_members.

Email luôn lưu và tìm bằng chữ thường: "Duc@Gmail.com" và "duc@gmail.com" là 1 người.
Repository KHÔNG commit (service / script commit).
"""
import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import ApplicationMember, User


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalars(select(User).where(User.email == normalize_email(email))).first()


def create(db: Session, email: str, password_hash: str | None, name: str | None = None,
           is_superadmin: bool = False) -> User:
    user = User(email=normalize_email(email), name=name, password_hash=password_hash,
                is_active=True, is_superadmin=is_superadmin)
    db.add(user)
    db.flush()
    return user


def roles_of(db: Session, user_id: uuid.UUID) -> dict[str, str]:
    """{application_id: role} của 1 user. Mỗi request đọc lại -> thu hồi quyền có hiệu lực ngay."""
    rows = db.execute(
        select(ApplicationMember.application_id, ApplicationMember.role).where(ApplicationMember.user_id == user_id)
    )
    return {app_id: role for app_id, role in rows}


def set_role(db: Session, application_id: str, user_id: uuid.UUID, role: str) -> ApplicationMember:
    member = db.get(ApplicationMember, (application_id, user_id))
    if member is None:
        member = ApplicationMember(application_id=application_id, user_id=user_id, role=role)
        db.add(member)
    else:
        member.role = role
    db.flush()
    return member


def remove_role(db: Session, application_id: str, user_id: uuid.UUID) -> bool:
    result = db.execute(delete(ApplicationMember).where(
        ApplicationMember.application_id == application_id, ApplicationMember.user_id == user_id))
    return result.rowcount == 1


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.email)))
