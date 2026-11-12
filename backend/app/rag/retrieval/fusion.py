"""
MỤC ĐÍCH: Gộp nhiều danh sách kết quả thành MỘT bằng Reciprocal Rank Fusion (RRF)

Vấn đề: điểm vector (cosine 0.0-1.0) và điểm keyword (ts_rank) có thang đo
khác nhau hoàn toàn-> không thể cộng trực tiếp

RRF chỉ dùng THỨ HẠNG, bỏ qua điểm gốc: 
    RRF(chunk) = Σ  1 / (k + rank_i(chunk))       (rank bắt đầu từ 1) 
Ví dụ với k = 60: 
    chunk A: hạng 1 ở vector, hạng 3 ở keyword -> 1/61 + 1/63 = 0.0323 
    chunk B: hạng 2 ở vector, không có ở keyword -> 1/62     = 0.0161 
  => A lên đầu: được CẢ HAI phương pháp đồng ý thì đáng tin hơn. 
k (mặc định 60, theo bài báo gốc của Cormack et al., 2009) làm "phẳng" chênh lệch 
giữa các hạng đầu, để một kết quả hạng 1 ở một nhánh không lấn át tất cả.

ranked_lists = {
    "vector": [
        chunk_A,  # rank 1
        chunk_B,  # rank 2
        chunk_C,  # rank 3
    ],
    "keyword": [
        chunk_B,  # rank 1
        chunk_D,  # rank 2
        chunk_A,  # rank 3
    ],
}

ranks[A] = {
    "vector": 1,
    "keyword": 3
}

Kết quả RRF:
RetrievedChunk(
    chunk_id=A,
    document_id=D1,
    filename="CV.pdf",
    page=2,
    chunk_index=5,
    text="He has experience with Docker...",
    score=0.03226,
    ranks={
        "vector": 1,
        "keyword": 3
    }
)
"""

import uuid
from dataclasses import replace

from app.rag.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]], k: int = 60,  weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    if k <= 0:
        raise ValueError("RRF k must be > 0")

    weights = weights or {}
    if any(w < 0 for w in weights.values()):
        raise ValueError("RRF weights must be >= 0")

    scores: dict[uuid.UUID, float] = {}
    first_seen: dict[uuid.UUID, RetrievedChunk] = {}
    ranks: dict[uuid.UUID, dict[str, int]] = {}                             # chunk_id, {source-> rank}

    # source : vector hoặc keyword
    for source, results in ranked_lists.items():
        for rank, chunk in enumerate(results, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weights.get(source, 1.0) / (k + rank)           # nếu có thì lấy điểm hiện tại, nếu chưa có thì lấy 0.0
            first_seen.setdefault(cid, chunk)                               # setdefault do 1 chunk xuất hiện ở nhiều retriver, lưu object chunk
            ranks.setdefault(cid, {})[source] = rank                        # lưu rank của 2 retriver

    # Sắp xếp theo điểm RRF giảm dần, hòa điểm thì ưu tiên chunk có hạng tốt nhất cao hơn
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values())))

    # replace là lấy RetrievedChunk ban đầu trong first_seen, tạo ra một bản sao mới của nó, nhưng update score và ranks
    return [replace(first_seen[cid], score=scores[cid], ranks=ranks[cid]) for cid in ordered]   