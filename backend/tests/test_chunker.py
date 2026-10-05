"""
MỤC ĐÍCH: Unit test cho chunker theo TOKEN.

Phần lớn test dùng WhitespaceTokenCounter (1 từ = 1 token) để kết quả dễ đoán, dễ tính tay.
Test cuối dùng tiktoken thật với tiếng Việt để chắc chắn không vỡ dấu.
"""

import pytest

from app.ingestion.chunker import chunk_pages, split_text
from app.ingestion.loaders import PageText
from app.ingestion.tokenizer import TiktokenCounter, WhitespaceTokenCounter

WS = WhitespaceTokenCounter()
# 40 câu, mỗi câu đúng 7 từ = 7 token (theo WhitespaceTokenCounter)
LONG_TEXT = " ".join(f"Sentence number {i} talks about RAG systems." for i in range(40))

class CharCounter:
    """1 ký tự = 1 token — để test trường hợp phải cắt cứng theo ký tự."""

    def count(self, text: str) -> int:
        return len(text)

# Nếu text < chunk_size thì là 1 chunk
def test_short_text_is_one_chunk() -> None:
    assert split_text("Hello world", chunk_size=50, chunk_overlap=10, counter=WS) == ["Hello world"]

def test_empty_text_gives_no_chunks() -> None:
    assert split_text("   \n\n ", chunk_size=50, chunk_overlap=10, counter=WS) == []

def test_chunks_respect_token_limit() -> None:
    chunks = split_text(LONG_TEXT, chunk_size=30, chunk_overlap=8, counter=WS)
    assert len(chunks) > 1
    assert all(WS.count(c) <= 30 for c in chunks)

def test_no_word_is_lost() -> None:
    chunks = split_text(LONG_TEXT, chunk_size=30, chunk_overlap=8, counter=WS)
    joined = " ".join(chunks)
    for word in set(LONG_TEXT.split()):
        assert word in joined

def test_cuts_at_sentence_boundaries() -> None:
    chunks = split_text(LONG_TEXT, chunk_size=30, chunk_overlap=8, counter=WS)
    assert all(c.endswith("systems.") for c in chunks)       # không chunk nào kết thúc giữa câu

def test_consecutive_chunks_overlap_by_whole_sentence() -> None:
    chunks = split_text(LONG_TEXT, chunk_size=30, chunk_overlap=8, counter=WS)
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence = nxt.split(". ")[0]
        assert first_sentence in prev                        # câu đầu của chunk sau = câu cuối chunk trước

def test_no_overlap_when_zero() -> None:
    chunks = split_text(LONG_TEXT, chunk_size=30, chunk_overlap=0, counter=WS)
    assert sum(WS.count(c) for c in chunks) == WS.count(LONG_TEXT)   # tổng token không bị lặp

def test_overlap_takes_last_sentence_inside_paragraph() -> None:
    # Đoạn 1 (9 từ) vừa chunk_size nên là 1 unit nguyên khối;
    # overlap vẫn lấy được CÂU CUỐI của đoạn (3 từ <= 4), không cắt nửa câu.
    para1 = "First point here. Second point here. Third point here.\n\n"
    para2 = " ".join(["word"] * 8)
    chunks = split_text(para1 + para2, chunk_size=12, chunk_overlap=4, counter=WS)
    assert len(chunks) == 2
    assert chunks[1].startswith("Third point here.")


def test_no_overlap_rather_than_cutting_a_sentence() -> None:
    # Câu cuối của chunk 1 dài 6 từ > chunk_overlap 4 -> không overlap, chứ không cắt nửa câu
    text = "one two three four five six.\n\n" + " ".join(["word"] * 8)
    chunks = split_text(text, chunk_size=10, chunk_overlap=4, counter=WS)
    assert chunks == ["one two three four five six.", " ".join(["word"] * 8)]

def test_prefers_paragraph_boundary() -> None:
    text = " ".join(["alpha"] * 20) + "\n\n" + " ".join(["beta"] * 20)
    chunks = split_text(text, chunk_size=30, chunk_overlap=0, counter=WS)
    assert chunks == [" ".join(["alpha"] * 20), " ".join(["beta"] * 20)]

def test_hard_split_very_long_word() -> None:
    chunks = split_text("x" * 250, chunk_size=100, chunk_overlap=0, counter=CharCounter())
    assert [len(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_config_raises(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=size, chunk_overlap=overlap, counter=WS)

def test_chunk_pages_keeps_page_index_and_token_count() -> None:
    pages = [PageText("Page one text.", 1), PageText("Page two has more text.", 2)]
    chunks = chunk_pages(pages, chunk_size=50, chunk_overlap=10, counter=WS)
    assert [(c.chunk_index, c.page, c.token_count) for c in chunks] == [(0, 1, 3), (1, 2, 5)]

def test_real_tokenizer_vietnamese_no_broken_characters() -> None:
    pytest.importorskip("tiktoken")
    counter = TiktokenCounter("o200k_base")
    text = " ".join(
        f"Dự án {i}: xây dựng hệ thống truy xuất thông tin với PostgreSQL và pgvector." for i in range(30)
    )
    chunks = chunk_pages([PageText(text, 1)], chunk_size=64, chunk_overlap=16, counter=counter)
    assert len(chunks) > 1
    assert all(c.token_count <= 64 for c in chunks)
    assert all(c.token_count == counter.count(c.text) for c in chunks)
    assert not any("\ufffd" in c.text for c in chunks)        # không có ký tự hỏng "�"