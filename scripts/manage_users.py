"""
MỤC ĐÍCH (Phase 12b): quản lý tài khoản và quyền — KHÔNG có trang đăng ký, chỉ bạn tạo user bằng script này.

    python scripts/manage_users.py create-user duc@example.com --name "Duc" --superadmin
    python scripts/manage_users.py create-user hr@example.com
    python scripts/manage_users.py grant hr@example.com cv viewer       # viewer | editor
    python scripts/manage_users.py revoke hr@example.com cv
    python scripts/manage_users.py set-password hr@example.com
    python scripts/manage_users.py deactivate hr@example.com           # khoá ngay, token cũ hết tác dụng
    python scripts/manage_users.py activate hr@example.com
    python scripts/manage_users.py set-public cv on                    # khách không đăng nhập chat được
    python scripts/manage_users.py list

Mật khẩu: script HỎI (gõ không hiện chữ). Không nhận mật khẩu qua tham số dòng lệnh, vì lệnh
nằm lại trong ~/.bash_history và `ps`. Chạy không tương tác (CI) thì đặt biến môi trường NEW_PASSWORD.
"""
import argparse
import getpass
import os
import sys

from sqlalchemy import select

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.db.session import SessionLocal
from app.models import ROLES, Application
from app.repositories import user_repository as user_repo
from app.services.auth_service import AuthService


def _read_password() -> str:
    if os.environ.get("NEW_PASSWORD"):
        return os.environ["NEW_PASSWORD"]
    first = getpass.getpass("Mật khẩu mới (>= 10 ký tự): ")
    if first != getpass.getpass("Nhập lại: "):
        sys.exit("Hai lần nhập không khớp.")
    return first


def _list(auth: AuthService) -> None:
    db = auth.db
    print(f"{'email':<35} {'active':<7} {'superadmin':<11} roles")
    for u in user_repo.list_users(db):
        roles = ", ".join(f"{a}:{r}" for a, r in sorted(user_repo.roles_of(db, u.id).items())) or "-"
        print(f"{u.email:<35} {str(u.is_active):<7} {str(u.is_superadmin):<11} {roles}")
    print()
    for a in db.scalars(select(Application).order_by(Application.id)):
        print(f"application {a.id:<20} {'CÔNG KHAI' if a.is_public else 'riêng tư'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quản lý user và quyền (Phase 12b)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create-user")
    p.add_argument("email")
    p.add_argument("--name")
    p.add_argument("--superadmin", action="store_true")
    sub.add_parser("set-password").add_argument("email")
    sub.add_parser("activate").add_argument("email")
    sub.add_parser("deactivate").add_argument("email")
    p = sub.add_parser("grant")
    p.add_argument("email")
    p.add_argument("application_id")
    p.add_argument("role", choices=ROLES)
    p = sub.add_parser("revoke")
    p.add_argument("email")
    p.add_argument("application_id")
    p = sub.add_parser("set-public")
    p.add_argument("application_id")
    p.add_argument("value", choices=["on", "off"])
    sub.add_parser("list")
    args = parser.parse_args()

    with SessionLocal() as db:
        auth = AuthService(db, get_settings())
        try:
            if args.cmd == "create-user":
                user = auth.create_user(args.email, _read_password(), args.name, args.superadmin)
                print(f"Đã tạo {user.email}{' (superadmin)' if user.is_superadmin else ''}.")
            elif args.cmd == "set-password":
                auth.set_password(args.email, _read_password())
                print("Đã đổi mật khẩu.")
            elif args.cmd in ("activate", "deactivate"):
                auth.set_active(args.email, args.cmd == "activate")
                print(f"Đã {'mở' if args.cmd == 'activate' else 'khoá'} {args.email}.")
            elif args.cmd == "grant":
                auth.grant(args.email, args.application_id, args.role)
                print(f"{args.email} là {args.role} của '{args.application_id}'.")
            elif args.cmd == "revoke":
                removed = auth.revoke(args.email, args.application_id)
                print("Đã thu hồi quyền." if removed else "User không có quyền gì trên application này.")
            elif args.cmd == "set-public":
                auth.set_public(args.application_id, args.value == "on")
                print(f"'{args.application_id}' giờ là {'CÔNG KHAI' if args.value == 'on' else 'riêng tư'}.")
            else:
                _list(auth)
        except AppError as exc:
            sys.exit(f"Lỗi: {exc}")


if __name__ == "__main__":
    main()
