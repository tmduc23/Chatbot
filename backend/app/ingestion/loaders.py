"""
MỤC ĐÍCH: Bước LOADER + PARSER của Indexing Pipeline

Đọc file gốc (PDF/ DOCX/ TXT/ MD) và trả về danh sách PageText:
mỗi phần tử là text thô của 1 trang kèm số trang

Vì sao giữ số trang? -> Để Phase 7 trích dẫn được "cv.pdf - trang 2"
DOCX/TXT không có khái niệm trang nên page = None

Muốn hỗ trợ định dạng mới (HTML, webpage ...): viết 1 hàm load_xxx
rồi đăng ký vào LOADERS - không phải sửa chỗ nào khác
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError

@dataclass(frozen=True)
class PageText:
    text: str
    page: int | None    # đánh số từ 1; None nếu định dạng không có trang


def load_pdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return [
        PageText(text=page.extract_text() or "", page=i)
        for i, page in enumerate(reader.pages, start=1)
    ]

def load_docx(path: Path) -> list[PageText]:
    from docx import Document as DocxDocument
    from docx.table import Table

    doc = DocxDocument(str(path))
    lines: list[str] = []
    # iter_inner_content(): duyệt paragraph và bảng theo đúng thứ tự trong file.
    # CV hay dùng bảng để chia cột - nếu chỉ đọc doc.paragraphs sẽ mất nội dung
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells: list[str] = []
                for cell in row.cells:
                    t = cell.text.strip()
                    # ô gộp (merged cell) bị lặp lại -> bỏ trùng liên tiếp
                    if t and (not cells or cells[-1] !=t):
                        cells.append(t)
                if cells:
                    lines.append(" | ".join(cells))
        else:
            lines.append(block.text)
    return [PageText(text="\n".join(lines), page=None)]

def load_text(path: Path) -> list[PageText]:
    # utf-8-sig: tự bỏ ký tự BOM mà Notepad trên Window hay thêm vào đầu file
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return [PageText(text=text, page=None)]



LOADERS: dict[str, Callable[[Path], list[PageText]]] = {
    "pdf": load_pdf,
    "docx": load_docx,
    "txt": load_text,
    "md": load_text
}

SUPPORTED_FILE_TYPES = frozenset(LOADERS)

def load_document(path: Path, file_type:str) -> list[PageText]:
    loader = LOADERS.get(file_type)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{file_type}'. Supported: {sorted(SUPPORTED_FILE_TYPES)}"
        )

    return loader(path)