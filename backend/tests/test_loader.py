"""MỤC ĐÍCH: Test loader TXT và DOCX bằng file tạo tạm thời (PDF kiểm tra thủ công với CV thật)."""

from pathlib import Path

import pytest
from docx import Document as DocxDocument

from app.core.exceptions import UnsupportedFileTypeError
from app.ingestion.loaders import load_document

# Load file .txt
def test_load_txt(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("Xin chào, tôi là kỹ sư AI.", encoding="utf-8")
    pages = load_document(f, "txt")
    assert len(pages) == 1
    assert pages[0].text == "Xin chào, tôi là kỹ sư AI."
    assert pages[0].page is None

# Load file .docx và không làm mất nội dung trong bảng
def test_load_docx_includes_tables(tmp_path: Path) -> None:
    f = tmp_path / "cv.docx"
    doc = DocxDocument()
    doc.add_paragraph("Skills")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Python"
    table.rows[0].cells[1].text = "FastAPI"
    doc.save(str(f))

    text = load_document(f, "docx")[0].text
    assert "Skills" in text
    assert "Python | FastAPI" in text   # nội dung trong bảng không bị mất

# Từ chối định dạng không hỗ trợ 
def test_unsupported_type_raises(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_document(tmp_path / "image.png", "png")