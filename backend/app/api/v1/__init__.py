"""Gom mọi router của API phiên bản 1. main.py gắn vào với prefix /api/v1."""
from fastapi import APIRouter

from app.api.v1 import applications, auth, chat, documents

api_v1_router = APIRouter()
api_v1_router.include_router(auth.router)
api_v1_router.include_router(applications.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(chat.router)