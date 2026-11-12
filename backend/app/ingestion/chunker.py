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

    # finditer() sẽ tìm tất cả các vị trí mà regex khớp 
    # m là 1 match object
    # Lấy từ start đến match hiện tại
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

# Mục đích nhận 1 đoạn text, nếu quá dài thì tìm cách chia nhỏ các unit, ưu tiên chia ở ranh giới tự nhiên 
"""
Logic (đệ quy):
      (1) Text đã vừa?  -> trả về nguyên text, KHÔNG tách gì thêm.
      (2) Chưa vừa: lấy dấu phân cách TO NHẤT có xuất hiện trong text, tách theo nó.
          Mỗi mảnh vừa tách lại được xử lý ĐỆ QUY, nhưng chỉ với các dấu NHỎ HƠN
          (separators[i + 1:]) -> không bao giờ tách lại theo dấu đã dùng.
      (3) Hết dấu phân cách mà vẫn quá dài -> _hard_split.
"""
def _split_into_units(
    text: str, max_tokens: int, counter: TokenCounter, separators: tuple[str, ...] = SEPARATORS
) -> list[str]:
    if counter.count(text) <= max_tokens:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            units: list[str] = []
            for piece in _split_keep_separator(text, sep):
                units.extend(_split_into_units(piece, max_tokens, counter, separators[i + 1:]))      # Đệ quy quá dài thì lại tách tiếp
            return units
    return _hard_split(text, max_tokens, counter)       # tách cuối cùng là tách theo ký tự

"""
Gộp các unit liên tiếp thành chunk <= chunk_size token, có overlap giữa 2 chunk liền nhau.

    Ý tưởng "CỬA SỔ TRƯỢT" (sliding window):
      - `window` chứa các unit của chunk đang gom; `total` = tổng token trong window.
      - Lần lượt thêm từng unit vào window.
      - Nếu thêm unit mới sẽ VƯỢT chunk_size:
          (a) "Đóng" window hiện tại thành 1 chunk.
          (b) Bỏ bớt unit ở ĐẦU window, chỉ giữ phần ĐUÔI <= chunk_overlap token
              -> phần đuôi này thành phần mở đầu (overlap) của chunk tiếp theo.
          (c) Thêm unit mới vào và tiếp tục.
"""

def _merge_units(units: list[str], chunk_size: int, chunk_overlap: int, counter: TokenCounter) -> list[str]:
    chunks: list[str] = []
    window: list[tuple[str, int]] = []   # (text_chung, số token)
    total = 0                            # tổng token trong window (cộng dồn, không phải đếm lại)


    for unit in units:
        n = counter.count(unit)          # đếm token của unit này đúng 1 lần

        if window and total + n > chunk_size:           # kiểm tra window có nội dung hay chưa và thêm unit mới sẽ vượt token
            # (a) Thêm unit này sẽ vượt giới hạn -> đóng window hiện tại thành 1 chunk.
            chunk_text = "".join(u for u, _ in window)      # 1 chunk
            chunks.append(chunk_text)

            # (b) Overlap = các CÂU cuối của chunk vừa đóng, tổng không vượt `budget`:
            #     - <= chunk_overlap          -> overlap không quá lớn
            #     - <= chunk_size - n         -> overlap + unit mới vẫn vừa 1 chunk
            budget = min(chunk_overlap, chunk_size - n)
            tail = _overlap_tail(chunk_text, budget, counter)       # xét ngược chunktext, lấy tail mới 
            window = [(tail, counter.count(tail))] if tail else []  # window mới gồm thành phần overlap
            total = window[0][1] if window else 0                   # tổng số token trong window mới, số token overlap

        # (c) Thêm unit mới vào window (chunk mới, hoặc chunk đang gom dở).
        window.append((unit, n))        # thêm unit mới và số token của unit
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
    if not text.strip():        # text rỗng -> không chunk
        return []
    units = _split_into_units(text, chunk_size, counter)    # chia unit
    return _merge_units(units, chunk_size, chunk_overlap, counter)  # hợp thành các chunk


def chunk_pages(
    pages: list[PageText], chunk_size: int, chunk_overlap: int, counter: TokenCounter
) -> list[TextChunk]:
    result: list[TextChunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap, counter):
            result.append(TextChunk(text=piece, chunk_index=len(result), page=page.page,
                                    token_count=counter.count(piece)))
    return result



# chunk_sections (CHUNK_STRATEGY=section): theo ĐOẠN VĂN + MỤC
# Dòng tiêu đề mục kiểu Markdown: "## Results", "### 4.2. Ablation"
_HEADING = re.compile(r"^#{1,6}\s+(.+)$")

def _document_title(pages: list[PageText]) -> str | None:
    """
    Tiêu đề tài liệu = khối chữ ĐẦU TÊN, nhưng chỉ khi tài liệu có dòng "##mục"
    (file .txt của QASPER/PeerQA: dòng 1 là tiêu đề paper, sau đó là các mục).
    Không có mục nào (CV, PDF thường) -> None: không đoán bừa tiêu đề.

    >>> _document_title([PageText("My Paper\\n\\n## Intro\\n\\nText.", None)])
    'My Paper'
    >>> _document_title([PageText("Just a paragraph.\\n\\nAnother one.", None)]) is None
    True
    """
    text = "\n\n".join(p.text for p in pages)
    if not any(_HEADING.match(line) for line in text.splitlines()):
        return None
    first = text.strip().split("\n\n", 1)[0].strip()
    if "\n" in first or _HEADING.match(first) or len(first) > 300:
        return None
    
    return first


def chunk_sections(
    pages: list[PageText], chunk_size: int, counter: TokenCounter,
    min_tokens: int = 40, header: bool = True,
) -> list[TextChunk]:
    """
    Chunk theo CẤU TRÚC thay vì theo số token. Dùng cho paper khoa học (CHUNK_STRATEGY=section)

    Khác chunk_pages 3 chỗ:
        1. Mỗi ĐOẠN VĂN là 1 chunk. Đoạn văn ngắn hơn min_tokens thì gộp với đoạn sau, nhưng
        KHÔNG BAO GIỜ gộp qua ranh giới MỤC ("##..."). Đoạn dài hơn chunk_size thì tách ở ranh 
        giới câu bằng split_text (không overlap).
        2. Đầu mỗi chunk có 1 dòng ngữ cảnh "Tiêu đề -> Mục" (header=True), để chunk tự nói nó thuộc
        paper nào, phần nào. Dòng này tính vào chunk_size.
        3. Không overlap: ranh giới đoạn văn là ranh giới ý, không cần lặp lại câu.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    title = _document_title(pages)
    result: list[TextChunk] = []
    section = ""                        # tên section hiện tại
    pending: list[str] = []             # các paragraph đang chờ gộp (cùng mục, cùng trang)

    # Tạo dòng ngữ cảnh đầu chunk
    def prefix() -> str:
        if not header or not (title or section):
            return ""
        line = " > ".join(x for x in (title, section) if x) + "\n"
        # dòng ngữ cảnh dài quá nửa chunk (tiêu đề rất dài) -> bỏ, giữ chỗ cho nội dung
        return line if counter.count(line) <= chunk_size // 2 else ""

    # Đóng gói thành TextChunk và đưa vào result
    def emit(body: str, page: int | None) -> None:
        text = prefix() + body
        result.append(TextChunk(text=text, chunk_index=len(result), page=page, token_count=counter.count(text)))


    # xuất các đoạn trong 1 pending thành 1 chunk
    def flush(page: int | None) -> None:
        if pending:
            emit("\n\n".join(pending), page)
            pending.clear()

    for page in pages:
        for block in (b.strip() for b in page.text.split("\n\n")):
            if not block or block == title:
                continue
            first, separator, rest = block.partition("\n")
            heading = _HEADING.match(first)
            if heading:
                flush(page.page)            # "## Tên mục" -> đóng mục cũ, sang mục mới
                section = heading.group(1).strip()      # đổi tên sang thư mục mới
                block = rest.strip()        # trường hợp đoạn dính ngay dưới dòng tiêu đề
                if not block: 
                    continue
            budget = chunk_size - counter.count(prefix())         # chỗ còn lại cho nội dung

            if counter.count(block) > budget:                   # đoạn quá dài -> tách ở ranh giới câu
                flush(page.page)                                # flush các paragraph trong pending 
                for piece in split_text(block, budget, 0, counter):     # split_text(): chia block hiện tại thành các phần nhỏ hơn hoặc bằng budget
                    emit(piece, page.page)                              # xuất từng thành phần thành chunk riêng
                continue
            if pending and counter.count("\n\n".join([*pending, block])) > budget:      # block mới + pending > budget
                flush(page.page)                                                        # xuất paragraph cũ thành chunk
            pending.append(block)                                                       # sau đó thực hiện append
            if counter.count("\n\n".join(pending)) >= min_tokens:       
                flush(page.page)
        flush(page.page)            # không gộp đoạn qua 2 trang -> số trang luôn đúng
    return result
    