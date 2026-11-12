"""
MỤC ĐÍCH: Test cách chunk thứ 2 (CHUNK_STRATEGY=section): theo đoạn văn + mục, có dòng "Tiêu đề > Mục".

Dùng WhitespaceTokencounter (1 từ = 1 token) để tự đếm được
Test cuối nạp file thật vào database (giống test_document_service.py).
"""

import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import delete

from app.core.config import Settings
from app.db.session import SessionLocal
from app.evaluation.report import config_snapshot
from app.ingestion.chunker import chunk_pages, chunk_sections
from app.ingestion.loaders import PageText
from app.ingestion.pipeline import parse_document
from app.ingestion.tokenizer import WhitespaceTokenCounter
from app.models import EMBEDDING_DIM, Application
from app.rag.embeddings.fake import FakeEmbeddingProvider
from app.repositories import document_repository as doc_repo
from app.services.document_service import DocumentService

WS = WhitespaceTokenCounter()
PAPER = (
    "Glacier Melt\n\n"
    "## Method\n\n"
    "We model melt with a degree-day model.\n\n"
    "Tiny.\n\n"
    "## Results\n\n"
    "Melt doubled in thirty years."
)

def _texts(text: str, **kw) -> list[str]:
    params = {"chunk_size": 30, "counter": WS, "min_tokens": 3}
    params.update(kw)
    return [c.text for c in chunk_sections([PageText(text, None)], **params)]


def test_each_chunk_starts_with_title_and_section() -> None:
    assert _texts(PAPER) == [
        "Glacier Melt > Method\nWe model melt with a degree-day model.",
        "Glacier Melt > Method\nTiny.",                    # Đoạn ngắn cuối mục: KHÔNG gộp sang mục Results
        "Glacier Melt > Results\nMelt doubled in thirty years."
    ]

def test_short_paragraphs_are_merged_insize_one_section() -> None:
    text = "T\n\n## A\n\nOne.\n\nTwo.\n\nThree words here."
    assert _texts(text, min_tokens=4) == ["T > A\nOne.\n\nTwo.\n\nThree words here."]


def test_long_paragraph_is_split_at_sentences_and_fits_chunk_size() -> None:
    long = " ".join(f"Sentence {i} is here." for i in range(20))
    chunks = chunk_sections([PageText(f"T\n\n## A\n\n{long}", None)], chunk_size=20, counter=WS)
    assert len(chunks) > 1
    assert all(c.text.startswith("T > A\n") for c in chunks)
    assert all(c.token_count <= 20 for c in chunks)
    body = " ".join(c.text.removeprefix("T > A\n") for c in chunks)
    assert body.split() == long.split()
    assert all(c.text.endswith(".") for c in chunks)


def test_document_without_headings_has_no_context_line() -> None:
    cv = "Nguyen Van A\n\nPython, FastAPI, Docker.\n\nBuilt a RAG chatbot."
    assert _texts(cv, min_tokens=1) == ["Nguyen Van A", "Python, FastAPI, Docker.", "Built a RAG chatbot."]


def test_header_can_be_turned_off() -> None:
    assert _texts(PAPER, header=False)[0] == "We model melt with a degree-day model."


def test_pages_are_kept_and_sections_continues_on_next_page() -> None:
    pages = [PageText("T \n\n## Intro\n\nFirst page text here.", 1), PageText("Second page text here.", 2)]
    chunks = chunk_sections(pages, chunk_size=30, counter=WS, min_tokens=50)
    assert [(c.page, c.text) for c in chunks] == [
        (1, "T > Intro\nFirst page text here."),            # không gộp qua 2 trang dù ngắn hơn min_tokens
        (2, "T > Intro\nSecond page text here."),           # mục "Intro" vẫn tiếp tục ở trang 2
    ]
    assert [c.chunk_index for c in chunks] == [0, 1]

def test_parse_document_default_is_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "paper.txt"
    f.write_text(PAPER, encoding="utf-8")
    old = chunk_pages([PageText(PAPER, None)], 30, 5, WS)
    assert parse_document(f, "txt", 30, 5, WS) == old                      # không truyền strategy = như cũ
    section = parse_document(f, "txt", 30, 5, WS, strategy="section", min_tokens=3)
    assert section[0].text.startswith("Glacier Melt > Method\n")


def test_settings_validate_strategy_and_snapshot_records_it() -> None:
    assert Settings(_env_file=None, chunk_strategy=" Section ").chunk_strategy == "section"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_strategy="paragraph")
    snap = config_snapshot(Settings(_env_file=None, chunk_strategy="section"))
    assert snap["chunk_strategy"] == "section" and snap["chunk_section_header"] is True


def test_ingest_with_section_strategy_stores_context_line(tmp_path: Path) -> None:
    db = SessionLocal()
    app_id = f"t_{uuid.uuid4().hex[:8]}"
    db.add(Application(id=app_id, name="test"))
    db.commit()
    try:
        settings = Settings(_env_file=None, chunk_size=60, chunk_strategy="section", chunk_min_tokens=3,
                            tokenizer_encoding="whitespace")
        service = DocumentService(db, settings=settings, storage_dir=tmp_path / "storage",
                                  embedder=FakeEmbeddingProvider(EMBEDDING_DIM))
        f = tmp_path / "glacier.txt"
        f.write_text(PAPER, encoding="utf-8")
        doc = service.ingest_file(app_id, f)
        chunks = doc_repo.list_chunks(db, app_id, doc.id)
        assert [c.text.split("\n")[0] for c in chunks] == [
            "Glacier Melt > Method", "Glacier Melt > Method", "Glacier Melt > Results"]
    finally:
        db.rollback()
        db.execute(delete(Application).where(Application.id == app_id))
        db.commit()
        db.close()