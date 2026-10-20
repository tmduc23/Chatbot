"""
MỤC ĐÍCH: Biến chunk sau rerank thành đoạn CONTEXT gửi cho LLM (spec mục 11)

Nhiệm vụ:
    1. giữ thứ tự của reranker (chunk liên quan nhất đứng đầu);
    2. loại trùng: cùng chunk_id, hoặc cùng nội dung (bỏ khác biệt khoảng trắng/hoa thường);
    3. giới hạn tổng dộ dài (CONTEXT_MAX_CHARS) -> không nhồi quá nhiều vào LLM;
    4. đánh số nguồn [Source 1], [Source 2] ... và GIỮ metadata để trích dẫn (Phase 7)

Format (theo spec):
    [Source 1]
    Document: cv.pdf
    Page: 2

    Built a RAG chatbot with FastAPI...

Hết chỗ thì DỪNG (không nhảy xuống lấy chunk ngắn hơn ở đưới): chunk càng về sau càng kém liên quan.
Riêng chunk đầu tiên nếu dài hơn cả giới hạn thì bị cắt bớt, để context không bao giờ rỗng
"""

import uuid
from dataclasses import dataclass

from app.rag.reranking.base import RerankedChunk

SEPARATOR = "\n\n"

@dataclass(frozen=True)
class Source:
    number: int             # số trong [Source n]
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    score: float             # điểm reranker 

    def to_dict(self) -> dict:
            """Format trả về cho frontend (spec mục 13). UUID -> str để chuyển được sang JSON."""
            return {
                "number": self.number,
                "document_id": str(self.document_id),
                "filename": self.filename,
                "page": self.page,
                "chunk_id": str(self.chunk_id),
            }

@dataclass(frozen=True)
class BuiltContext:
    text: str                # đoạn văn gửi cho LLM
    sources: list[Source]    # danh sách kèm nguồn số thứ tự để trích dẫn
    dropped: int             # số chunk bị bỏ (trùng hoặc hết chỗ)

def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _header(number: int, filename: str, page: int | None) -> str:
    lines = [f"[Source {number}]", f"Document: {filename}"]
    if page is not None:
        lines.append(f"Page: {page}")
    return "\n".join(lines) + "\n\n"

# Context chunk và prompt gửi cho LLM
def build_context(chunks: list[RerankedChunk], max_chars: int = 6000) -> BuiltContext:
    blocks: list[str] = []                  # các đoạn [Source n] .. text đã chọn
    sources: list[Source] = []              # source tương ứng
    seen_ids: set[uuid.UUID] = set()        
    seen_texts: set[str] = set()
    used = 0                                # số ký tự đã dùng
    dropped = 0                             # số chunk bị bỏ

    for i, r in enumerate(chunks):
        c = r.chunk                         # trùng thì bỏ qua, xét tiếp chunk tiếp theo 
        key = _normalize(c.text)
        if c.chunk_id in seen_ids or key in seen_texts:
            dropped += 1
            continue

        number = len(sources) + 1           # số thứ tự lấy theo sô nguồn đã nhận, không theo i
        header = _header(number, c.filename, c.page)
        body = c.text.strip()
        extra = len(SEPARATOR) if blocks else 0     # các block nối nhau bằng \n\n
        if used + extra + len(header) + len(body) > max_chars:
            if blocks:                              # đã có context -> dừng, bỏ phần còn lại
                dropped += len(chunks) - i
                break                               # outloop, xuống return 
            body = body[: max(0, max_chars - len(header))]      # chunk đầu tiên quá dài -> cắt

        blocks.append(header + body)        # nhận chunk
        used += extra + len(header) + len(body)
        seen_ids.add(c.chunk_id)
        seen_texts.add(key)
        sources.append(Source(number, c.chunk_id, c.document_id, c.filename, c.page, c.chunk_index, r.score))

    return BuiltContext(SEPARATOR.join(blocks), sources, dropped)       # nối các block bằng \n\n