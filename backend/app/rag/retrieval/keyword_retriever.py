"""
MỤC ĐÍCH: Tìm chunk theo TỪ KHÓA CHÍNH XÁC bằng PostgreSQL full-text search.

Vì sao cần khi đã có vector search?
    Vector search giỏi "nghĩa gần nhau" nhưng hay trượt với:
        - tên riêng, tên công nghệ hiếm: "pgvector", "Langraph", "Kafka"
        - mã số, phiên bản, viết tắt: "AWS SAA-C03", "GPA 3.6", "K8s"
    
        Keyword search bắt chính xác những từ này

    Phương pháp:
        1. build_or_query(): tách câu hỏi thành từ, bỏ stop-word ("what", "is", "của", "là", ...),
        nối bằng OR: "docker | compose | kubernetes"

        2. SQL: WHERE application_id = ? AND search_vector @@ to_tsquery(...)
                ORDER BY ts_rank_cd (...) DESC
            ts_rank_cd: chunk khớp với NHIỀU từ hơn, và các từ đứng GẦN nhau hơn -> điểm cao hơn

Giới hạn đã biết (config "simple"):
    - không stemming: "deploy" không khớp "deployment"
    - không bỏ dấu: "kinh nghiem" không khớp "kinh nghiệm"

    Đây là lý do dùng HYBRID: vector search bù cho những chỗ này

"Python_3.11!"
       ↓
    Regex
       ↓
["Python", "3", "11"]
"""

import logging
import re
import time
import unicodedata

from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session

from app.models import TS_CONFIG, Chunk, Document, DocumentStatus
from app.rag.retrieval.base import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)   # chữ + số, không lấy dấu gạch dưới 
MAX_TERMS = 32

STOPWORDS = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "she", "that", "the", "their", "them", "they", "this", "to", "was", "were", 
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "about", "any", 
    "tell", "describe", 
    # Tiếng Việt (tách theo âm tiết) 
    "anh", "ấy", "bạn", "các", "cho", "chị", "có", "của", "đã", "đang", "được", "gì", "hãy", 
    "không", "là", "làm", "mà", "những", "nào", "này", "người", "như", "thế", "trong", "và", 
    "về", "với", "em", "ở", "sẽ", "thì", "nó",
})


def extract_terms(query: str) -> list[str]:
    # Tách câu hỏi thành danh sách từ khóa (đã viết thường, bỏ stop-word, bỏ trùng, bỏ từ dưới 2 ký tự)
    query = unicodedata.normalize("NFC", query.lower())
    terms: list[str] = []
    for word in _WORD.findall(query):
        if len (word) < 2 or word in STOPWORDS or word in terms:
            continue
        terms.append(word)

    return terms[:MAX_TERMS]

# Dùng OR giúp Recall cao, tránh bỏ lỡ tài liệu
def build_or_query(query: str) -> str | None:
    """What is his Docker experience? -> 'docker | experience'. None nếu không còn từ nào"""
    terms = extract_terms(query)
    return " | ".join(terms) if terms else None

class KeywordRetriever(Retriever):
    def __init__(self, db: Session):
        self.db = db

    def retrieve(self, query: str, application_id: str, top_k: int) -> list[RetrievedChunk]:
        or_query = build_or_query(query)
        if or_query is None:
            return []


        t0 = time.perf_counter()
        # TS_CONFIG là hằng số trong code (không phải input người dùng) -> an toàn khi dùng literal.
        # Câu hỏi của người dùng vẫn đi qua bound parameter như bình thường.
        # 'simple': reconfig -> cấu hình FTS
        # docker | experience: nội dung cần tìm
        # literal_column giúp đánh dấu TS_CONFIG: simple là 1 quy tắc
        ts_query = func.to_tsquery(literal_column(f"'{TS_CONFIG}'::regconfig"), or_query)

        # Chấm điểm mỗi Chunk phù hợp với query đến mức nào, đặt tên cho kết quả
        rank = func.ts_rank_cd(Chunk.search_vector, ts_query).label("rank")

        # SQL query được SQLAlchemy xây dựng
        stmt = (
            select(Chunk, Document.filename, rank)              # select là trả lời lấy cái gì, lấy toàn bộ Chunk, filename document và rank
            .join(Document, Document.id == Chunk.document_id)   # trả lời nối dữ liệu từ đâu để lấy
            .where(                                             # where là phần đặt ddieuf kiện
                Chunk.application_id == application_id,
                Document.status == DocumentStatus.READY,
                Chunk.search_vector.op("@@")(ts_query)          # @@ dùng để kiểm tra và lọc casc chunk thỏa mãn điều kiện tìm kiếm toàn văn
            )
            .order_by(rank.desc(), Chunk.chunk_index)           # desc: descending
            .limit(top_k)
        )
        rows = self.db.execute(stmt).all()                      # thực thi và .all() lấy tất cả các dòng kết quả và lưu vào biến rows
        logger.info(
            "keyword_search app=%s query=%r results=%d search_ms=%.0f",
            application_id, or_query, len(rows), (time.perf_counter() - t0) * 1000,
        )
        return[
            RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page=chunk.page, chunk_index=chunk.chunk_index, text=chunk.text,
                score=float(r)
            )
            for chunk, filename, r in rows
        ]