"""
MỤC ĐÍCH: Test đọc và kiểm tra trích dẫn [Source n] (Phase 7, spec: unit test "citation parsing").
Hàm thuần, không cần DB, không gọi LLM.
"""

import pytest

from app.rag.citations import parse_citations, clean_citations

# `parametrize` ở test đầu chạy 5 trường hợp, nên file này có **9 test** (5 + 4).

@pytest.mark.parametrize(("text", "expected"), [
    ("Docker [Source 1].", [1]),
    ("A [Source 2]. B [Source 1][Source 2].", [2, 1]),                # thứ tự xuất hiện, không trùng
    ("A [Source 1, 3]. B [Sources 2, Source 4].", [1, 3, 2, 4]),     # gộp vào 1 ngoặc
    ("a [source 2] b [ SOURCE 5]", [2, 5]),                          # chữ thường /  hoa, có khoảng trắng
    ("Source 1, [Source], (Source 2)", []),                           # không đúng định dạng [Source n] -> bỏ qua 
])
def test_parse_citations(text: str, expected: list[int]) -> None:
    assert parse_citations(text) == expected

# Xóa citation không tồn tại
def test_clean_removes_citations_that_do_not_exist() -> None:
    check = clean_citations("Dùng Docker [Source 1][Source 7].", valid={1, 2, 3})
    assert check.text == "Dùng Docker [Source 1]."
    assert check.cited == [1]
    assert check.invalid == [7]

# Test clean invalid citation và không để lại marker
def test_clean_with_only_invalid_citations_leaves_no_marker() -> None:
    check = clean_citations("Dùng Docker [Source 9].", valid={1, 2})
    assert check.text == "Dùng Docker."              # không còn khoảng trắng thừa trước dấu chấm
    assert check.cited == [] and check.invalid == [9]

# Test normlize và bỏ invalid source
def test_clean_normalizes_combined_citations() -> None:
    check = clean_citations("Docker và Kubernetes [Sources 1, 4, 2].", valid={1, 2, 3})
    assert check.text == "Docker và Kubernetes [Source 1][Source 2]."
    assert check.cited == [1, 2] and check.invalid == [4]

def test_clean_keeps_text_without_citations() -> None:
    check = clean_citations("Có kinh nghiệm Docker.", valid={1})
    assert check.text == "Có kinh nghiệm Docker."
    assert check.cited == [] and check.invalid == []



def test_fullwidth_brackets_are_normalized() -> None:
    # gpt-oss (Groq) hay viết 【Source 1】 thay vì [Source 1]
    check = clean_citations("Force keeps it moving 【Source 1】 【Source 2】 .", valid={1, 2})
    assert check.cited == [1, 2]
    assert check.text == "Force keeps it moving [Source 1] [Source 2]."