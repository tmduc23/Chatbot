from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from app.db.session import get_engine

router = APIRouter(tags=["health"])

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}             # process còn hay không


@router.get("/ready")
def ready() -> dict[str, str]:
    try: 
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))

    except Exception:
        raise HTTPException(status_code=503, detail="Database not reachable")

    return {"status": "ready"}       # database kết nối được