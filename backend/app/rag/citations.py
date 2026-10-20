"""
MỤC ĐÍCH: Đọc và KIỂM TRA trich dẫn [Source n] trong câu trả lời của LLM (Phase 7, spec mục 13)

LLM không phải lúc nào cũng trích đúng format. Các kiểu hay gặp:
    [Source 1]                          chuẩn
    [Source 1][Source 3]                chuẩn, nhiều nguồn
    [Source 1, 3]                       gộp trong 1 ngoặc
    [Sources 1, 3] [source 2]           số nhiều, chữ thường
    [Source 7]                          BỊA: context chỉ có 3 nguồn

clean_citations():
    1. Đưa mọi kiểu về dạng chuẩn [Source 1][Source 3]      -> frontend chỉ cần hiểu 1 format
    2. XÓA số không tồn tại trong context                   -> không hiển thị nguồn "ma" cho người dùng;
    3. Trả về danh sách số hợp lệ (theo thứ tự xuất hiện) và số bị loại (để log, đếm trong eval).
"""

import re
from dataclasses import dataclass

# [Source 1] | [Sources 1, 3] | [source 1, Source 3] — không phân biệt hoa thường
_CITATION = re.compile(r"\[\s*sources?\s+(\d+(?:\s*,\s*(?:sources?\s+)?\d+)*)\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")        


@dataclass(frozen=True)
class CitationCheck:
    text: str           # Câu trả lời đã chuẩn hóa, đã xóa số không hợp lệ
    cited: list[int]    # số hợp lệ, theo thứ tự xuất hiện lần đầu, không trùng
    invalid: list[int]  # số LLM trích nhưng không có trong context

# Đưa về thứ tự xuất hiện
def parse_citations(text: str) -> list[int]:
    """'A [Source 2]. B [Source 1, 2].' -> [2, 1] (thứ tự xuất hiện lần đầu, không trùng)"""
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        for n in _NUMBER.findall(match.group(1)):           # tìm tất cả số trong text
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


"""
VÀO:  "Duc học HUST [sources 2, 7]. Dùng Docker [Source 7] và Python [Source 1, 3]. GPA 3.35 [Source 2]."
RA:   text    = "Duc học HUST [Source 2]. Dùng Docker và Python [Source 1][Source 3]. GPA 3.35 [Source 2]."
      cited   = [2, 1, 3]
      invalid = [7]
"""
def clean_citations(text: str, valid: set[int]) -> CitationCheck:
    cited: list[int] = []
    invalid: list[int] = []

    def replace(match: re.Match) -> str:
        kept = []
        for n in (int(x) for x in _NUMBER.findall(match.group(1))):
            if n in valid:
                kept.append(n)
                if n not in cited:
                    cited.append(n)
            elif n not in invalid:
                invalid.append(n)
        return "".join(f"[Source {n}]" for n in kept)

    cleaned = _CITATION.sub(replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)   # "Docker [Source 7]." -> "Docker ." -> "Docker.", xóa khoảng trắng trước dấu câu
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()    # tìm có từ 2 khoảng trắng hoặc tab liên tiếp, thay bằng 1 dấu cách
    return CitationCheck(cleaned, cited, invalid)   

