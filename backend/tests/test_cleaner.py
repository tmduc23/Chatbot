"""
MỤC ĐÍCH: Unit test cho cleaner - không cần database
"""
from app.ingestion.cleaner import clean_text

# Chữ ệ viết dạng tổ hợp rời được đưa về NFC.
def test_unicode_is_normalized_to_nfc() -> None:
    decomposed =  "Kinh nghie\u0323\u0302m"   # "ệ" viết bằng e + dấu nặng + dấu mũ rời
    assert clean_text(decomposed) == "Kinh nghiệm"

# Gộp khoảng trắng, tab và nhiều dòng trống thành một dòng trống.
def test_collapses_spaces_and_blank_lines() -> None:
    raw = "  Python   \t  FastAPI \n\n\n\n  Docker  "
    assert clean_text(raw) == "Python FastAPI\n\nDocker"

# Ký tự \x00 bị xoá.
def test_removes_null_characters() -> None:
    assert clean_text("abc\x00def") == "abcdef"