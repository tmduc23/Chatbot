"""
MỤC ĐÍCH: Interface chung cho Reranker + kiểu kết quả + reranker "không làm gì" (baseline)
Vì sao cần rerank sau hybrid search?
  Retriever (bi-encoder / keyword) chấm câu hỏi và chunk ĐỘC LẬP với nhau -> nhanh nhưng thô.
  Reranker đọc câu hỏi và chunk CÙNG LÚC -> hiểu chunk có thật sự trả lời câu hỏi không.
  Chậm hơn nhiều, nên chỉ chạy trên ~20 ứng viên tốt nhất, không phải cả database.

Khác spec một chút (có chủ đích): spec gợi ý rerank(query, documents: list[str]).
Ở đây nhận list[RetrievedChunk] để GIỮ metadata(filename, page, chunk_id) cho citation.
Ở phase 7 - nếu chỉ truyền string thì sẽ mất dấu chunk nào là chunk nào.

Gemini và Cohere khi lỗi đều gọi `PassthroughReranker().rerank(...)`, nên tự động có `judged=False`, không phải sửa 2 file đó.
Field mới có giá trị mặc định nên mọi chỗ đang viết `RerankedChunk(c, s, i)` vẫn chạy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.rag.retrieval.base import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk       # chunk được retrieve thông qua RRF
    score: float                # điểm của reranker
    retrieval_rank: int         # hạng TRƯỚC khi rerank (bắt đầu từ 1) -  để so sánh trước/sau
    # True =  score do reranker chấm (so được với RERANKER_MIN_SCORE), thang 0-1, so được với ngưỡng
    # False = không có ai chấm (PassthroughReranker, hoặc reranker lỗi rồi fail-open):
    #         score chỉ là điểm retrieval (RRF ~0.016-0.033) -> KHÔNG được lọc theo ngưỡng
    #         điểm retrieval, khác thang đo, bỏ qua lọc và cho cả 5 chunk đi tiếp.
    judged: bool = True


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int, domain: str | None = None
    ) -> list[RerankedChunk]:
        """domain mô tả bộ tài liệu của application (application.description), None = mặc định"""


def rank_by_scores(chunks: list[RerankedChunk], score: list[float], top_k: int) -> list[RerankedChunk]:
    """Sắp xếp theo điểm giảm dần; hòa điểm thì giữ thứ tự retrieval ban đầu"""
    items = [RerankedChunk(c, s, i) for i, (c, s) in enumerate(zip(chunks, score, strict=True), start=1)]
    items.sort(key=lambda r: (-r.score, r.retrieval_rank))
    return items[:top_k]

class PassthroughReranker(Reranker):
    """
    Không rerank: giữ nguyên thứ tự retrieval, chỉ cắt top-k
    Dùng làm baseline để đo reranker có giúp gì không, và làm phương án dự phòng khi API bị lỗi.
    judged=False có nghĩa là không được reranked bằng LLM, auto cho qua luôn
    """

    def rerank(
            self, query: str, chunks: list[RetrievedChunk], top_k: int, domain: str | None = None 
    ) -> list[RerankedChunk]:
        return [RerankedChunk(c, c.score, i, judged=False) for i, c in enumerate(chunks[:top_k], start=1)]