"""
MỤC ĐÍCH: Đếm số TOKEN của một đoạn text — thước đo dùng cho chunker.

Vì sao đếm token thay vì ký tự?
  - Model (embedding, LLM) đọc và tính giới hạn/chi phí theo TOKEN, không theo ký tự.
  - Cùng 500 ký tự: tiếng Anh ~110 token, tiếng Việt có dấu có thể ~150-200 token.
    Chia theo ký tự -> chunk tiếng Việt "nặng" hơn chunk tiếng Anh mà không hay biết.

Tokenizer nào?
  Tokenizer của Gemini không công khai để chạy offline. Dùng `tiktoken` (o200k_base)
  làm thước đo XẤP XỈ: chạy local, rất nhanh, hỗ trợ tốt tiếng Việt.
  Sai số vài % không quan trọng vì chunk (~256 token) nhỏ hơn rất nhiều so với
  giới hạn đầu vào của gemini-embedding-001 (2048 token).

Chunker chỉ phụ thuộc vào interface TokenCounter -> có thể đổi tokenizer
(hoặc dùng WhitespaceTokenCounter trong test) mà không sửa chunker.
"""
import re
from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    def __init__(self, encoding_name: str = "o200k_base"):
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def count(self, text: str) -> int:
        # disallowed_special=(): coi chuỗi như "<|endoftext|>" trong tài liệu là text thường
        return len(self._encoding.encode(text, disallowed_special=()))


class WhitespaceTokenCounter:
    """Đếm số từ (tách theo khoảng trắng). Kết quả dễ đoán -> dùng cho unit test."""

    name = "whitespace"
    _WORD = re.compile(r"\S+")

    def count(self, text: str) -> int:
        return len(self._WORD.findall(text))


@lru_cache
def get_token_counter(encoding_name: str) -> TokenCounter:
    if encoding_name == "whitespace":
        return WhitespaceTokenCounter()
    return TiktokenCounter(encoding_name)