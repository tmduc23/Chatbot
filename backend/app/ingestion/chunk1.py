"""
MỤC ĐÍCH: Bước CHUNKER của Indexing Pipeline — cắt text dài thành các đoạn nhỏ (chunk),
đo độ dài bằng TOKEN.

======================================================================================
1. VÌ SAO PHẢI CHUNK?
======================================================================================
  - Embedding của cả một tài liệu dài là "trung bình" của quá nhiều ý -> tìm kiếm kém.
    Chunk nhỏ, mỗi chunk một ý -> vector "sắc nét" -> tìm đúng đoạn cần tìm.
  - LLM chỉ nên nhận vài đoạn liên quan nhất, không phải cả file (tốn token, dễ lạc đề).

======================================================================================
2. TỔNG QUAN THUẬT TOÁN (2 bước)
======================================================================================

  text ─► BƯỚC 1: _split_into_units ─► [unit, unit, ...] ─► BƯỚC 2: _merge_units ─► [chunk, chunk, ...]
          (tách cho đủ NHỎ)            mỗi unit <= chunk_size  (gộp lại cho đủ LỚN)

  BƯỚC 1 — TÁCH: chia text thành các "đơn vị" (unit), mỗi unit <= chunk_size token.
    Ưu tiên ranh giới tự nhiên theo thứ tự:  đoạn văn "\\n\\n"  >  dòng "\\n"  >  câu ". "  >  từ " "
    CHỈ tách nhỏ hơn khi cần: một đoạn văn đã vừa chunk_size thì giữ nguyên cả đoạn.
    Trường hợp hiếm: một "từ" dài hơn chunk_size (vd. URL rất dài) -> cắt cứng theo ký tự.

  BƯỚC 2 — GỘP: ghép các unit liên tiếp vào một chunk cho tới khi sắp vượt chunk_size.
    Chunk tiếp theo bắt đầu bằng vài unit CUỐI của chunk trước (tổng <= chunk_overlap token)
    -> giữ ngữ cảnh khi một ý nằm vắt qua 2 chunk.

  Vì sao phải 2 bước?
    - Chỉ tách, không gộp: CV có nhiều dòng ngắn ("Python", "Docker"...) -> hàng trăm chunk
      1-2 token, vô nghĩa khi tìm kiếm.
    - Chỉ gộp, không tách: một đoạn văn dài hơn chunk_size sẽ không bao giờ vừa.

======================================================================================
3. VÌ SAO KHÔNG CẮT THẲNG THEO TOKEN?
======================================================================================
  Cách "ngây thơ": encode text -> mảng token -> cắt mỗi 256 token -> decode.
  Vấn đề: tokenizer tách chữ tiếng Việt có dấu thành các mảnh BYTE; chữ "ệ" có thể
  nằm ở 2 token khác nhau. Cắt giữa 2 token đó -> decode ra ký tự hỏng "�".
  Ở đây tokenizer chỉ dùng để ĐẾM; điểm cắt luôn nằm giữa 2 ký tự hợp lệ.

======================================================================================
4. HỆ QUẢ CẦN BIẾT VỀ OVERLAP
======================================================================================
  Overlap = "TỐI ĐA chunk_overlap token, gồm các unit TRỌN VẸN" (không cắt nửa câu).
  Nếu unit cuối dài hơn chunk_overlap (vd. cả một đoạn văn 100 token, overlap = 40)
  thì 2 chunk KHÔNG chồng nhau. Chấp nhận được: khi đó điểm cắt đã nằm đúng ranh giới
  đoạn văn / câu, nên ít khi làm mất ngữ cảnh.

======================================================================================
5. CÁCH CHẠY CÁC VÍ DỤ TRONG FILE NÀY
======================================================================================
  Các dòng bắt đầu bằng ">>>" trong docstring là ví dụ CHẠY ĐƯỢC (doctest).
  Kiểm tra mọi ví dụ đều đúng:
      docker compose exec backend pytest --doctest-modules app/ingestion/chunker.py -v
  Các ví dụ dùng WhitespaceTokenCounter: 1 TỪ = 1 TOKEN, để bạn tự đếm tay được.
"""
from dataclasses import dataclass

from app.ingestion.loaders import PageText
from app.ingestion.tokenizer import TokenCounter

# Thứ tự ưu tiên khi tách: thử dấu phân cách "to" trước, không được mới dùng dấu "nhỏ" hơn.
#   "\n\n" = ranh giới ĐOẠN VĂN   (cleaner đã gộp nhiều dòng trống thành đúng 1 dòng trống)
#   "\n"   = ranh giới DÒNG        (CV hay viết mỗi ý một dòng)
#   ". "   = ranh giới CÂU
#   " "    = ranh giới TỪ          (phương án cuối trước khi phải cắt cứng theo ký tự)
SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class TextChunk:
    """
    Kết quả cuối cùng của chunker cho MỘT chunk.

    frozen=True: không sửa được sau khi tạo -> tránh vô tình thay đổi dữ liệu giữa các bước.
    """

    text: str
    chunk_index: int   # thứ tự trong TOÀN BỘ tài liệu (không reset theo trang), bắt đầu từ 0
    page: int | None   # số trang (PDF); None với DOCX/TXT
    token_count: int   # số token THỰC TẾ của chunk (lưu vào metadata để theo dõi)


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 1: _split_keep_separator — tách chuỗi nhưng GIỮ LẠI dấu phân cách
# ──────────────────────────────────────────────────────────────────────────────────────
def _split_keep_separator(text: str, sep: str) -> list[str]:
    """
    Tách `text` theo `sep`, gắn `sep` vào CUỐI mảnh bên trái.

    Vì sao phải giữ dấu phân cách?
      str.split() của Python VỨT BỎ dấu phân cách. Tách "a. b" theo ". " rồi ghép lại
      sẽ được "ab" -> mất dấu chấm và khoảng trắng. Giữ lại thì "".join(mảnh) == text gốc.

    >>> _split_keep_separator("a. b. c", ". ")
    ['a. ', 'b. ', 'c']

    Ghép lại không mất ký tự nào:
    >>> "".join(_split_keep_separator("a. b. c", ". ")) == "a. b. c"
    True

    Text kết thúc bằng dấu phân cách -> mảnh rỗng ở cuối bị bỏ:
    >>> _split_keep_separator("Dòng 1\\nDòng 2\\n", "\\n")
    ['Dòng 1\\n', 'Dòng 2\\n']
    """
    parts = text.split(sep)                               # "a. b. c" -> ["a", "b", "c"]
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]  # gắn lại sep, trừ mảnh cuối -> ["a. ", "b. ", "c"]
    return [p for p in pieces if p]                       # bỏ chuỗi rỗng (khi text kết thúc bằng sep)


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 2: _hard_split — phương án cuối: cắt cứng theo KÝ TỰ
# ──────────────────────────────────────────────────────────────────────────────────────
def _hard_split(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """
    Cắt `text` thành các mảnh, mỗi mảnh DÀI NHẤT có thể mà vẫn <= max_tokens.

    Chỉ dùng khi text không còn dấu phân cách nào mà vẫn quá dài
    (vd. một URL 600 ký tự liền, chuỗi base64...). Hiếm gặp trong CV.

    Vì sao cắt theo KÝ TỰ chứ không theo token? -> Mục 3 đầu file: không làm vỡ chữ có dấu.

    Cách tìm điểm cắt: TÌM KIẾM NHỊ PHÂN (binary search) trên vị trí ký tự.
      Thử cắt ở giữa: vừa -> thử dài hơn; quá dài -> thử ngắn hơn.
      Text 1.000 ký tự chỉ cần ~10 lần đếm token thay vì 1.000 lần.

    Ví dụ với bộ đếm "1 ký tự = 1 token":
    >>> class CharCounter:
    ...     def count(self, text): return len(text)
    >>> _hard_split("abcdefghij", 4, CharCounter())
    ['abcd', 'efgh', 'ij']
    """
    pieces: list[str] = []
    start = 0                                    # vị trí bắt đầu của mảnh hiện tại
    while start < len(text):
        # Tìm `end` lớn nhất sao cho text[start:end] <= max_tokens.
        # lo = start + 1: mỗi mảnh có ÍT NHẤT 1 ký tự -> vòng lặp luôn tiến lên, không bao giờ treo.
        lo, hi = start + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2             # +1: làm tròn LÊN, tránh lặp vô hạn khi hi = lo + 1
            if counter.count(text[start:mid]) <= max_tokens:
                lo = mid                         # vừa -> điểm cắt có thể còn xa hơn
            else:
                hi = mid - 1                     # quá dài -> điểm cắt phải gần hơn
        pieces.append(text[start:lo])            # lo = điểm cắt xa nhất còn vừa
        start = lo                               # mảnh tiếp theo bắt đầu ngay sau đó
    return pieces


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 3: _split_into_units — BƯỚC 1: tách text thành các unit đủ nhỏ (ĐỆ QUY)
# ──────────────────────────────────────────────────────────────────────────────────────
def _split_into_units(
    text: str, max_tokens: int, counter: TokenCounter, separators: tuple[str, ...] = SEPARATORS
) -> list[str]:
    """
    Trả về danh sách unit, mỗi unit <= max_tokens, cắt ở ranh giới "to" nhất có thể.

    Logic (đệ quy):
      (1) Text đã vừa?  -> trả về nguyên text, KHÔNG tách gì thêm.
      (2) Chưa vừa: lấy dấu phân cách TO NHẤT có xuất hiện trong text, tách theo nó.
          Mỗi mảnh vừa tách lại được xử lý ĐỆ QUY, nhưng chỉ với các dấu NHỎ HƠN
          (separators[i + 1:]) -> không bao giờ tách lại theo dấu đã dùng.
      (3) Hết dấu phân cách mà vẫn quá dài -> _hard_split.

    Ví dụ (WhitespaceTokenCounter: 1 từ = 1 token):
    >>> from app.ingestion.tokenizer import WhitespaceTokenCounter
    >>> ws = WhitespaceTokenCounter()

    Text 3 từ, giới hạn 5 -> đã vừa, giữ nguyên:
    >>> _split_into_units("Python FastAPI Docker", 5, ws)
    ['Python FastAPI Docker']

    2 dòng (3 + 4 từ = 7), giới hạn 4 -> tách theo "\\n", mỗi dòng đã vừa -> dừng:
    >>> _split_into_units("one two three\\nfour five six seven", 4, ws)
    ['one two three\\n', 'four five six seven']

    Giới hạn 2 -> tách theo "\\n" vẫn chưa vừa -> mỗi dòng tách tiếp theo " " (từ):
    >>> _split_into_units("one two three\\nfour five six seven", 2, ws)
    ['one ', 'two ', 'three\\n', 'four ', 'five ', 'six ', 'seven']
    """
    if counter.count(text) <= max_tokens:        # (1) đã vừa -> dừng đệ quy
        return [text]
    for i, sep in enumerate(separators):         # (2) thử từ dấu TO nhất
        if sep in text:
            units: list[str] = []
            for piece in _split_keep_separator(text, sep):
                # Đệ quy với các dấu NHỎ HƠN dấu vừa dùng.
                # Mảnh nào đã vừa thì lời gọi đệ quy trả về ngay ở bước (1).
                units.extend(_split_into_units(piece, max_tokens, counter, separators[i + 1:]))
            return units
    return _hard_split(text, max_tokens, counter)  # (3) hết cách -> cắt cứng


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 4: _merge_units — BƯỚC 2: gộp các unit thành chunk, kèm overlap
# ──────────────────────────────────────────────────────────────────────────────────────
def _merge_units(units: list[str], chunk_size: int, chunk_overlap: int, counter: TokenCounter) -> list[str]:
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

    Ví dụ: 4 câu, mỗi câu 3 từ (= 3 token), chunk_size=7, chunk_overlap=3.
    >>> from app.ingestion.tokenizer import WhitespaceTokenCounter
    >>> ws = WhitespaceTokenCounter()
    >>> units = ["A1 A2 A3. ", "B1 B2 B3. ", "C1 C2 C3. ", "D1 D2 D3."]
    >>> _merge_units(units, chunk_size=7, chunk_overlap=3, counter=ws)
    ['A1 A2 A3. B1 B2 B3.', 'B1 B2 B3. C1 C2 C3.', 'C1 C2 C3. D1 D2 D3.']

    Diễn biến từng bước:
      thêm A -> window=[A]    total=3
      thêm B -> window=[A,B]  total=6   (6 <= 7, còn chỗ)
      gặp C  -> 6+3=9 > 7 => (a) ĐÓNG chunk "A B"
                              (b) bỏ A (vì 6 > 3) -> window=[B] total=3 (3 <= 3 và 3+3 <= 7: dừng)
                              (c) thêm C -> window=[B,C] total=6
      gặp D  -> 6+3=9 > 7 => ĐÓNG "B C"; bỏ B, giữ [C]; thêm D -> [C,D]
      hết unit -> ĐÓNG chunk cuối "C D"
    => B và C mỗi câu xuất hiện ở 2 chunk liền nhau: đó là OVERLAP.

    Cùng dữ liệu, chunk_overlap=0 -> các chunk không chồng nhau:
    >>> _merge_units(units, chunk_size=7, chunk_overlap=0, counter=ws)
    ['A1 A2 A3. B1 B2 B3.', 'C1 C2 C3. D1 D2 D3.']
    """
    chunks: list[str] = []
    window: list[tuple[str, int]] = []   # các unit của chunk đang gom, kèm số token của từng unit
    total = 0                            # tổng token trong window (cộng dồn, không phải đếm lại)

    for unit in units:
        n = counter.count(unit)          # đếm token của unit này đúng 1 lần

        if window and total + n > chunk_size:
            # (a) Thêm unit này sẽ vượt giới hạn -> đóng window hiện tại thành 1 chunk.
            chunks.append("".join(u for u, _ in window))

            # (b) Bỏ unit ở ĐẦU window cho tới khi thoả CẢ HAI điều kiện:
            #     - phần còn lại (đuôi) <= chunk_overlap   -> overlap không quá lớn
            #     - đuôi + unit mới     <= chunk_size      -> chunk mới không vượt giới hạn
            # window.pop(0) lấy ra phần tử đầu tiên (unit, số_token); [1] là số token.
            while window and (total > chunk_overlap or total + n > chunk_size):
                total -= window.pop(0)[1]

        # (c) Thêm unit mới vào window (chunk mới, hoặc chunk đang gom dở).
        window.append((unit, n))
        total += n

    if window:                           # còn unit trong window -> chunk cuối cùng
        chunks.append("".join(u for u, _ in window))

    # strip(): bỏ khoảng trắng / xuống dòng thừa ở 2 đầu (sinh ra do giữ dấu phân cách ở HÀM 1).
    # Bỏ chunk rỗng (vd. chỉ toàn khoảng trắng).
    return [c.strip() for c in chunks if c.strip()]


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 5: split_text — HÀM CÔNG KHAI: kiểm tra tham số + chạy BƯỚC 1 rồi BƯỚC 2
# ──────────────────────────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int, chunk_overlap: int, counter: TokenCounter) -> list[str]:
    """
    Cắt MỘT đoạn text thành các chunk (chuỗi), mỗi chunk <= chunk_size token.

    Ví dụ: đoạn "kinh nghiệm" (4 từ) + đoạn "dự án" gồm 2 câu (6 + 4 = 10 từ). Tổng 14 từ.
    >>> from app.ingestion.tokenizer import WhitespaceTokenCounter
    >>> ws = WhitespaceTokenCounter()
    >>> text = ("Kinh nghiệm: Python, FastAPI.\\n\\n"
    ...         "Dự án: chatbot RAG với pgvector. Triển khai bằng Docker.")

    chunk_size=12: mỗi ĐOẠN VĂN đã vừa (4 <= 12, 10 <= 12) -> không tách thành câu.
    2 đoạn cộng lại 14 > 12 -> mỗi đoạn một chunk:
    >>> for chunk in split_text(text, chunk_size=12, chunk_overlap=6, counter=ws):
    ...     print(repr(chunk))
    'Kinh nghiệm: Python, FastAPI.'
    'Dự án: chatbot RAG với pgvector. Triển khai bằng Docker.'

    chunk_size=8: đoạn "dự án" (10 > 8) bị tách thành 2 câu (6 và 4) -> 3 chunk:
    >>> for chunk in split_text(text, chunk_size=8, chunk_overlap=4, counter=ws):
    ...     print(repr(chunk))
    'Kinh nghiệm: Python, FastAPI.'
    'Dự án: chatbot RAG với pgvector.'
    'Triển khai bằng Docker.'

    Ở cả 2 ví dụ không có overlap. Lý do: unit cuối của mỗi chunk không thể giữ lại,
    vì nó quá dài so với chunk_overlap, hoặc giữ lại thì chunk mới sẽ vượt chunk_size
    (xem mục 4 đầu file). Overlap xuất hiện khi các unit ngắn, như ví dụ ở HÀM 4.

    Tham số sai -> báo lỗi NGAY, trước khi làm bất cứ việc gì:
    >>> split_text("abc", chunk_size=100, chunk_overlap=100, counter=ws)
    Traceback (most recent call last):
    ...
    ValueError: chunk_overlap must be >= 0 and < chunk_size
    """
    # Kiểm tra tham số ĐẦU TIÊN, trước mọi lệnh return khác.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    # Đọc là: 0 <= overlap VÀ overlap < chunk_size (KHÔNG có dấu = ở vế sau).
    # overlap == chunk_size bị cấm: chunk sau lặp lại toàn bộ chunk trước -> không tiến lên được.
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not text.strip():                         # text rỗng / toàn khoảng trắng -> không có chunk nào
        return []
    units = _split_into_units(text, chunk_size, counter)               # BƯỚC 1: tách
    return _merge_units(units, chunk_size, chunk_overlap, counter)     # BƯỚC 2: gộp


# ──────────────────────────────────────────────────────────────────────────────────────
# HÀM 6: chunk_pages — HÀM CÔNG KHAI: chunk nhiều trang, gắn số trang + thứ tự + số token
# ──────────────────────────────────────────────────────────────────────────────────────
def chunk_pages(
    pages: list[PageText], chunk_size: int, chunk_overlap: int, counter: TokenCounter
) -> list[TextChunk]:
    """
    Chunk từng trang RIÊNG BIỆT rồi nối kết quả lại. Đây là hàm pipeline.py gọi.

    Vì sao chunk từng trang riêng?
      Để mỗi chunk thuộc đúng MỘT trang -> Phase 7 trích dẫn được "cv.pdf — trang 2".
      Nếu nối cả file rồi mới chunk, một chunk có thể nằm vắt qua 2 trang.
      Đánh đổi: không có overlap giữa chunk cuối trang 1 và chunk đầu trang 2.

    chunk_index đánh số LIÊN TỤC qua các trang (0, 1, 2, ...), không reset về 0 ở mỗi trang.

    >>> from app.ingestion.tokenizer import WhitespaceTokenCounter
    >>> ws = WhitespaceTokenCounter()
    >>> pages = [PageText("Một hai ba.", 1), PageText("Bốn năm.", 2)]
    >>> for c in chunk_pages(pages, chunk_size=10, chunk_overlap=2, counter=ws):
    ...     print(c.chunk_index, c.page, c.token_count, repr(c.text))
    0 1 3 'Một hai ba.'
    1 2 2 'Bốn năm.'
    """
    result: list[TextChunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap, counter):
            result.append(TextChunk(
                text=piece,
                chunk_index=len(result),           # = số chunk đã có -> tự tăng liên tục qua các trang
                page=page.page,
                token_count=counter.count(piece),  # đếm lại trên chunk CUỐI CÙNG (sau strip) cho chính xác
            ))
    return result