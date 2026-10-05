"""
MỤC ĐÍCH: Bước CHUNKER của Indexing Pipeline - cắt text dài thành các đoạn nhỏ,
đo độ dài bằng TOKEN

Vì sao phải chunk?
  - Embedding của cả một tài liệu dài là "trung bình" của quá nhiều ý -> tìm kiếm kém.
  - LLM chỉ nên nhận vài đoạn liên quan nhất, không phải cả file.

1. TÁCH thành các "đơn vị" (unit), mỗi unit <= chunk_size token, ưu tiên ranh giới tự nhiên:
       đoạn văn ("\\n\\n") > dòng ("\\n") > câu (". ") > từ (" ")
     Chỉ tách nhỏ hơn khi cần: đoạn văn vừa chunk_size thì giữ nguyên cả đoạn.
     Trường hợp hiếm (1 "từ" dài hơn chunk_size, vd. URL rất dài) -> cắt theo ký tự.

2. GỘP các unit liên tiếp thành chunk, tới khi sắp vượt chunk_size.
     Chunk tiếp theo bắt đầu bằng vài unit CUỐI của chunk trước (tổng <= chunk_overlap token)
     -> giữ ngữ cảnh khi một ý nằm vắt qua 2 chunk.

Vì sao KHÔNG cắt thẳng theo token (encode -> cắt mảng token -> decode)?
  Tokenizer tách tiếng Việt thành các mảnh BYTE. Cắt giữa một chữ có dấu sẽ ra ký tự
  hỏng "�". Ở đây chỉ dùng tokenizer để ĐẾM, còn điểm cắt luôn nằm giữa 2 ký tự hợp lệ.

Hệ quả cần biết: overlap là "TỐI ĐA chunk_overlap token, gồm các unit trọn vẹn".
Nếu unit cuối dài hơn chunk_overlap (vd. cả một đoạn văn) thì 2 chunk không chồng nhau —
chấp nhận được vì khi đó điểm cắt đã nằm đúng ranh giới đoạn văn.

Mỗi trang được chunk riêng để mỗi chunk biết chính xác nó thuộc trang nào.
"""


import re
from dataclasses import dataclass
from app.ingestion.loaders import PageText
from app.ingestion.tokenizer import TokenCounter

SEPARATORS = ("\n\n", "\n", ". ", " ")
# Ranh giới CÂU/DÒNG dùng khi lấy overlap:
#   (?<=[.!?…])\s+  = khoảng trắng đứng NGAY SAU dấu kết thúc câu . ! ? …
#   \n+             = một hoặc nhiều dấu xuống dòng
# "3.6" hay "v1.2" KHÔNG bị coi là hết câu vì sau dấu chấm không có khoảng trắng.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")

@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int        # thứ tự trong toàn bộ tài liệu, bắt đầu từ 0
    page: int | None
    token_count: int

# Tác giữ lại dấu phân cách
def _split_keep_separator(text: str, sep: str) -> list[str]:
    """'a. b. c' tách theo '. ' -> ['a. ', 'b. ', 'c'] (giữ dấu phân cách để ghép lại không mất chữ)."""
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p]

# HÀM 2b: _split_sentences — tách thành câu/dòng, GIỮ khoảng trắng phía sau mỗi câu
def _split_sentences(text: str) -> list[str]:
    """
    Tách text thành các câu / dòng trọn vẹn. Ghép lại ("".join) được đúng text gốc.

    _split_sentences("Tôi dùng Python. GPA 3.6 trên 4!\\nDòng mới")
    ['Tôi dùng Python. ', 'GPA 3.6 trên 4!\\n', 'Dòng mới']
    """
    pieces: list[str] = []
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        pieces.append(text[start:m.end()])   # câu + khoảng trắng/xuống dòng ngay sau nó
        start = m.end()
    if start < len(text):
        pieces.append(text[start:])          # phần còn lại (câu cuối không có dấu kết thúc)
    return [p for p in pieces if p.strip()]

# HÀM 2c: _overlap_tail — lấy các câu CUỐI của một chunk làm overlap
def _overlap_tail(text: str, budget: int, counter: TokenCounter) -> str:
    """
    Lấy NHIỀU câu cuối nhất có thể của `text` mà tổng vẫn <= budget token.
    Không bao giờ cắt nửa câu: câu cuối dài hơn budget -> trả về "" (không overlap).

    Ví dụ (1 từ = 1 token): 3 câu dài 2, 2, 2 token.
    from app.ingestion.tokenizer import WhitespaceTokenCounter
    ws = WhitespaceTokenCounter()
    _overlap_tail("A1 A2. B1 B2. C1 C2.", budget=4, counter=ws)
    'B1 B2. C1 C2.'
     _overlap_tail("A1 A2. B1 B2. C1 C2.", budget=3, counter=ws)
    'C1 C2.'

    Câu cuối 5 token > budget 3 -> không lấy gì (thay vì cắt nửa câu):
    _overlap_tail("Ngắn. Câu cuối này dài năm từ.", budget=3, counter=ws)
    ''
    """
    if budget <= 0:
        return ""
    tail = ""
    for sentence in reversed(_split_sentences(text)):   # đi ngược từ câu cuối lên
        candidate = sentence + tail
        if counter.count(candidate) > budget:
            break                                      # thêm câu này sẽ vượt budget -> dừng
        tail = candidate
    return tail
def _hard_split(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Cắt theo ký tự, mỗi mảnh dài nhất có thể mà vẫn <+ max_tokens (tìm kiếm nhị phân)."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        lo, hi = start + 1,  len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if counter.count(text[start:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(text[start:lo])
        start = lo
    return pieces

def _split_into_units(
    text: str, max_tokens: int, counter: TokenCounter, separators: tuple[str, ...] = SEPARATORS
) -> list[str]:
    if counter.count(text) <= max_tokens:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            units: list[str] = []
            for piece in _split_keep_separator(text, sep):
                units.extend(_split_into_units(piece, max_tokens, counter, separators[i + 1:]))
            return units
    return _hard_split(text, max_tokens, counter)

def _merge_units(units: list[str], chunk_size: int, chunk_overlap: int, counter: TokenCounter) -> list[str]:
    chunks: list[str] = []
    window: list[tuple[str, int]] = []   # các phần của chunk đang gom, kèm số token của từng phần
    total = 0                            # tổng token trong window (cộng dồn, không phải đếm lại)

    for unit in units:
        n = counter.count(unit)          # đếm token của unit này đúng 1 lần

        if window and total + n > chunk_size:
            # (a) Thêm unit này sẽ vượt giới hạn -> đóng window hiện tại thành 1 chunk.
            chunk_text = "".join(u for u, _ in window)
            chunks.append(chunk_text)

            # (b) Overlap = các CÂU cuối của chunk vừa đóng, tổng không vượt `budget`:
            #     - <= chunk_overlap          -> overlap không quá lớn
            #     - <= chunk_size - n         -> overlap + unit mới vẫn vừa 1 chunk
            budget = min(chunk_overlap, chunk_size - n)
            tail = _overlap_tail(chunk_text, budget, counter)
            window = [(tail, counter.count(tail))] if tail else []
            total = window[0][1] if window else 0

        # (c) Thêm unit mới vào window (chunk mới, hoặc chunk đang gom dở).
        window.append((unit, n))
        total += n

    if window:                           # còn nội dung trong window -> chunk cuối cùng
        chunks.append("".join(u for u, _ in window))

    # strip(): bỏ khoảng trắng / xuống dòng thừa ở 2 đầu (sinh ra do giữ dấu phân cách ở HÀM 1).
    # Bỏ chunk rỗng (vd. chỉ toàn khoảng trắng).
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, chunk_size: int, chunk_overlap: int, counter: TokenCounter) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not text.strip():
        return []
    units = _split_into_units(text, chunk_size, counter)
    return _merge_units(units, chunk_size, chunk_overlap, counter)


def chunk_pages(
    pages: list[PageText], chunk_size: int, chunk_overlap: int, counter: TokenCounter
) -> list[TextChunk]:
    result: list[TextChunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap, counter):
            result.append(TextChunk(text=piece, chunk_index=len(result), page=page.page,
                                    token_count=counter.count(piece)))
    return result