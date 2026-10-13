"""
MỤC ĐÍCH: Test context builder — chạy offline, không DB, không API.

Kiểm tra: format [Source n] theo spec, giữ metadata, loại trùng, giới hạn độ dài,
chunk đầu quá dài bị cắt (context không bao giờ rỗng), danh sách rỗng.
"""

import uuid

from app.rag.context_builder import build_context
from app.rag.reranking.base import RerankedChunk
from app.rag.retrieval.base import RetrievedChunk

# Tạo nhanh 1 RerankedChunk
# uuid5 tạo id cố định
def _r(name: str, text: str, page: int | None = None, score: float = 0.9, cid: uuid.UUID | None = None) -> RerankedChunk:
    cid = cid or uuid.uuid5(uuid.NAMESPACE_DNS, name + text)
    chunk = RetrievedChunk(chunk_id=cid, document_id=uuid.uuid5(uuid.NAMESPACE_DNS, name), filename=name,
                           page=page, chunk_index=0, text=text, score=0.01) 
    return RerankedChunk(chunk, score, 1)       # 1 là retrieval_rank



# Context chunk và prompt gửi cho LLM: build context

# test định dạng [source n], document, page và số thứ tự
def test_format_and_numbering() -> None:
    ctx = build_context([_r("cv.pdf", "Built a RAG chatbot.", page=2), _r("notes.md", "Knows Docker.")])
    assert ctx.text == ("[Source 1]\nDocument: cv.pdf\nPage: 2\n\nBuilt a RAG chatbot.\n\n"
                        "[Source 2]\nDocument: notes.md\n\nKnows Docker.")       # page None -> không có dòng Page
    assert [s.number for s in ctx.sources] == [1, 2]
    assert ctx.sources[0].filename == "cv.pdf" and ctx.sources[0].page == 2
    assert ctx.dropped == 0

# Source.score là điểm reranker, không phỉa điểm retrieval
# điểm reranker được giữ nguyên trong source, tránh score=c.chunk.score (điểm retrieval 0.01) thay vì score=c.score
def test_keeps_reranker_score_in_sources() -> None:
    ctx = build_context([_r("cv.pdf", "A", score=0.7)])
    assert ctx.sources[0].score == 0.7


# bỏ trùng id và trùng nội dung
def test_removes_duplicates_by_id_and_by_text() -> None:
    same_id = uuid.uuid4()
    chunks = [
        _r("cv.pdf", "Knows Docker.", cid=same_id),
        _r("cv.pdf", "Other text", cid=same_id),          # trùng chunk_id
        _r("cv2.pdf", "  knows   DOCKER. "),              # trùng nội dung (khác khoảng trắng, hoa thường), chuẩn hóa và bỏ
        _r("cv.pdf", "Python"),
    ]
    ctx = build_context(chunks)
    assert [s.number for s in ctx.sources] == [1, 2]
    assert "Python" in ctx.text and ctx.dropped == 2        # nhận thành source 2

# drop khi context đủ, tránh lấy 1 và 3, bỏ 2 do context 
def test_stops_when_context_is_full() -> None:
    chunks = [_r("a", "x" * 100), _r("b", "y" * 100), _r("c", "z")]
    ctx = build_context(chunks, max_chars=200)
    assert len(ctx.text) <= 200
    assert [s.filename for s in ctx.sources] == ["a"]     # "b" không vừa -> dừng, KHÔNG nhảy xuống lấy "c"
    assert ctx.dropped == 2

# cắt khi quá độ dài
def test_first_chunk_is_truncated_instead_of_empty_context() -> None:
    ctx = build_context([_r("a", "x" * 5000)], max_chars=300)
    assert len(ctx.text) <= 300
    assert len(ctx.sources) == 1

# không có chunk thì context rỗng, không lỗi
def test_empty_input() -> None:
    ctx = build_context([])
    assert ctx.text == "" and ctx.sources == [] and ctx.dropped == 0