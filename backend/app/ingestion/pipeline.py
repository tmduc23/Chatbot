"""
MỤC ĐÍCH: Ghép các bước Loader -> Cleaner -> Chunker thành 1 hàm duy nhất.

Hàm này là "thuần xử lý": chỉ nhận file, trả về list chunk.
Không đụng database, không biết application/document là gì.
=> Dễ test, dễ chạy thử với 1 file bất kỳ, dễ thay từng bước.

`counter` (bộ đếm token) được truyền vào thay vì tạo bên trong -> test có thể
dùng bộ đếm đơn giản, production dùng tiktoken.
"""
from pathlib import Path

from app.ingestion.chunker import TextChunk, chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import PageText, load_document
from app.ingestion.tokenizer import TokenCounter


def parse_document(
    path: Path, file_type: str, chunk_size: int, chunk_overlap: int, counter: TokenCounter
) -> list[TextChunk]:
    raw_pages = load_document(path, file_type)                                  # 1. Loader + Parser
    pages = [PageText(text=clean_text(p.text), page=p.page) for p in raw_pages] # 2. Cleaner
    pages = [p for p in pages if p.text]                                        #  bỏ trang trống
    return chunk_pages(pages, chunk_size, chunk_overlap, counter)               # 3. Chunker