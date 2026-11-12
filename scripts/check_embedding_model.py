"""
MỤC ĐÍCH: Kiểm tra mỗi document đang có vector của model nào.

Với mỗi document: lấy chunk đầu tiên, embed lại bằng model ĐANG đặt trong .env,
rồi so cosine với vector đang lưu trong DB.
  - cosine ~ 1.0  -> đúng model hiện tại   (OK)
  - cosine thấp   -> model khác             (KHAC)
  - không có vector                         (NULL)
Tốn quota: 1 text mỗi document (gửi theo batch).

    python scripts/check_embedding_model.py --app qasper
    python scripts/check_embedding_model.py --all
"""
import argparse
import math

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.rag.embeddings.factory import get_embedding_provider

SAME_MODEL = 0.95   # cùng model + cùng text thì cosine gần như bằng 1


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--app")
    g.add_argument("--all", action="store_true")
    args = p.parse_args()

    s = get_settings()
    provider = get_embedding_provider(s)
    model = s.cohere_embedding_model if s.embedding_provider == "cohere" else s.embedding_model
    print(f"Model hiện tại: {s.embedding_provider} / {model}\n")

    with SessionLocal() as db:
        q = (select(Document.application_id, Document.filename, Chunk.text, Chunk.embedding)
             .join(Chunk, Chunk.document_id == Document.id)
             .where(Chunk.chunk_index == 0)
             .order_by(Document.application_id, Document.filename))
        if args.app:
            q = q.where(Document.application_id == args.app)
        rows = db.execute(q).all()

    have = [r for r in rows if r.embedding is not None]
    fresh = provider.embed_documents([r.text for r in have]) if have else []
    score = {id(r): cosine(list(r.embedding), v) for r, v in zip(have, fresh, strict=True)}

    counts: dict[str, int] = {"OK": 0, "KHAC": 0, "NULL": 0}
    for r in rows:
        if r.embedding is None:
            tag, c = "NULL", float("nan")
        else:
            c = score[id(r)]
            tag = "OK" if c >= SAME_MODEL else "KHAC"
        counts[tag] += 1
        print(f"{tag:5} {c:6.3f}  {r.application_id:10} {r.filename}")

    print(f"\nOK={counts['OK']}  KHAC={counts['KHAC']}  NULL={counts['NULL']}")
    if counts["KHAC"] or counts["NULL"]:
        print("-> Chạy: python scripts/reembed_all.py --all   (embed lại tất cả bằng model hiện tại)")


if __name__ == "__main__":
    main()