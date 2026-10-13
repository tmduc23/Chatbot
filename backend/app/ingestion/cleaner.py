"""
MỤC ĐÍCH: Bước CLEANER của Indexing Pipeline

Text trichs từ PDF/DOCX thường "bẩn": nhiều khoảng trắng thừa, dòng trông liên tiếp, ký tự lạ. Làm sạch giúp:
  - chunk chứa nhiều nội dung thaapjt hơn (không phí chỗ cho khoảng trắng);
  - embedding và keyword search chính xác hơn

Đặc biệt quan trọng với tiếng việt: chữ "é" có thể được lưu theo 2 cách
(1 kysy tự, hoặc "e" + dấu sắc rời). Hai cách nhìn giống nhệt nhau nhưng máy coi là
khác nhau -> keyword search "kinh nghiệm" có thể không khớp
Chuẩn hóa Unicode NFC đưa tất cả về cùng một dạng
"""

import re
import unicodedata

# re.complie để biên dịch 1 chuỗi Regex thành 1 Regex Pattern Object sau đó dùng .sub(), .search()
_SPACES = re.compile(r"[ \t\u00a0\u200b]+")   # space, tab, non-breaking space, zero-width space
_SPACES_AROUND_NEWLINE = re.compile(r" *\n *")  # xử lý khoảng trắng xung quanh xuống dòng
_MANY_NEWLINES = re.compile(r"\n{3,}") 

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)       # Chuẩn hóa Unicode
    text = text.replace("\x00", "")                 # Xóa ký tự NULL
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # Chuẩn hóa newline
    text = _SPACES.sub(" ", text)                   # Gom khoảng trắng liên tiếp
    text = _SPACES_AROUND_NEWLINE.sub("\n", text)  # Xóa space quanh newline
    text = _MANY_NEWLINES.sub("\n\n", text)        # Giữ tối đa 1 dòng trống
    return text.strip()                             # Xóa whitespace đầu và cuối