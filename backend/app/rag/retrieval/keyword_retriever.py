"""
MỤC ĐÍCH: Tìm chunk theo TỪ KHOÁ CHÍNH XÁC bằng PostgreSQL full-text search.

Vì sao cần khi đã có vector search?
  Vector search giỏi "nghĩa gần nhau" nhưng hay trượt với:
    - tên riêng, tên công nghệ hiếm: "pgvector", "LangGraph", "Kafka"
    - mã số, phiên bản, viết tắt: "AWS SAA-C03", "GPA 3.6", "K8s"
  Keyword search bắt chính xác những thứ này.

Cách làm:
  1. build_or_query(): tách câu hỏi thành từ, bỏ stop-word ("what", "is", "của", "là"...),
     nối bằng OR:  "docker | compose | kubernetes"
     (Nếu dùng AND, câu hỏi tự nhiên dài sẽ gần như không bao giờ khớp.)
  2. SQL:  WHERE application_id = ? AND search_vector @@ to_tsquery(...)
           ORDER BY ts_rank_cd(...) DESC
     ts_rank_cd: chunk khớp NHIỀU từ hơn, và các từ đứng GẦN nhau hơn -> điểm cao hơn.

Hai cách xếp hạng (KEYWORD_RANKING):
  - ts_rank (mặc định): ts_rank_cd của Postgres. KHÔNG có IDF -> từ phổ biến ("model" có trong
    61% chunk QASPER) được tính ngang từ hiếm ("BookTest").
  - bm25: BM25 tính bằng SQL, ngay trên cột search_vector có sẵn (không extension, không migration):
        score = Σ_t  idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
        idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
    N, avgdl, df tính trong phạm vi 1 application (app khác không ảnh hưởng điểm).
    tf = số vị trí của từ trong chunk; dl = số từ khác nhau của chunk (length(tsvector)).
  - pg_textsearch: BM25 của extension pg_textsearch, chạy trên index ix_chunks_bm25 (migration 0006).
    Cùng công thức BM25 nhưng index tính N / avgdl / df trên TOÀN BẢNG (mọi application gộp lại);
    điều kiện application_id lọc SAU khi xếp hạng. Cần Postgres 17/18 có extension.

Giới hạn đã biết (config 'simple'):
  - không stemming: "deploy" không khớp "deployment";
  - không bỏ dấu: "kinh nghiem" không khớp "kinh nghiệm".
  Đây là lý do dùng HYBRID: vector search bù cho những chỗ này.
"""
import logging
import re
import time
import unicodedata
import hashlib
import uuid
from collections.abc import Sequence

from sqlalchemy import func, literal_column, select, text
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
    """Tách câu hỏi thành danh sách từ khoá (đã viết thường, bỏ stop-word, bỏ trùng)."""
    query = unicodedata.normalize("NFC", query.lower())
    terms: list[str] = []
    for word in _WORD.findall(query):
        if len(word) < 2 or word in STOPWORDS or word in terms:
            continue
        terms.append(word)
    return terms[:MAX_TERMS]


# Câu hỏi QASPER trong dataset có dạng:  In the paper "<tên paper>": <câu hỏi thật>
# Tên paper toàn từ hiếm (IDF cao) mà lại lặp khắp paper -> BM25 đẩy chunk "nói về chủ đề paper"
# lên trên chunk chứa đáp án. KEYWORD_STRIP_PAPER_PREFIX=true: nhánh keyword chỉ tìm theo câu hỏi thật
# (nhánh vector vẫn nhận câu đầy đủ). Nhận cả nháy kép lẫn nháy đơn.
_PAPER_PREFIX = re.compile(r"""^\s*In the paper\s+(["'])(?P<title>.+?)\1\s*:\s*""", re.IGNORECASE)

def strip_paper_prefix(query: str) -> str:
    """'In the paper "X": How long ...?' -> 'How long ...?'. Không có tiền tố -> giữ nguyên."""
    return _PAPER_PREFIX.sub("", query, count=1)

def build_or_query(query: str) -> str | None:
    """'What is his Docker experience?' -> 'docker | experience'. None nếu không còn từ nào."""
    terms = extract_terms(query)
    return " | ".join(terms) if terms else None


KEYWORD_RANKINGS = ("ts_rank", "bm25", "pg_textsearch")

# TS_CONFIG là hằng số trong code -> an toàn khi ghép vào SQL. Mọi giá trị từ người dùng đi qua bound parameter.
# q: các từ của câu hỏi sau khi Postgres chuẩn hoá (cùng cách với cột search_vector).
# Ép kiểu term::tsquery để KHÔNG chuẩn hoá lại một từ đã chuẩn hoá.
_BM25_SQL = f"""
WITH q AS (
    SELECT DISTINCT lexeme AS term FROM unnest(to_tsvector('{TS_CONFIG}'::regconfig, :qtext))
),
app_chunks AS (
    SELECT c.id, c.search_vector FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.application_id = :app AND d.status = 'READY' {{docs}}
),
stats AS (
    SELECT count(*)::float AS n, greatest(avg(length(search_vector)), 1)::float AS avgdl FROM app_chunks
),
df AS (
    SELECT q.term, count(a.id)::float AS df
    FROM q JOIN app_chunks a ON a.search_vector @@ quote_literal(q.term)::tsquery
    GROUP BY q.term
),
tf AS (
    SELECT a.id, u.lexeme AS term, coalesce(array_length(u.positions, 1), 1)::float AS tf,
           length(a.search_vector)::float AS dl
    FROM app_chunks a
    CROSS JOIN LATERAL unnest(a.search_vector) AS u
    JOIN q ON q.term = u.lexeme
)
SELECT tf.id,
       sum(ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
           * tf.tf * (:k1 + 1) / (tf.tf + :k1 * (1 - :b + :b * tf.dl / stats.avgdl))) AS score
FROM tf JOIN df ON df.term = tf.term CROSS JOIN stats
GROUP BY tf.id
ORDER BY score DESC, tf.id
LIMIT :top_k
"""

# pg_textsearch: toán tử <@> trả điểm BM25 ÂM (càng nhỏ càng khớp) -> ORDER BY tăng dần, đổi dấu khi trả về.
# Chunk không chứa từ nào được điểm 0 -> bỏ ở Python (giống ts_rank / bm25: chỉ trả chunk có khớp).
PGTS_INDEX = "ix_chunks_bm25"          # GIỮ: tên index BM25 chung (migration 0006)

_PGTS_SQL_TEMPLATE = """
SELECT c.id, -(c.text <@> to_bm25query(:qtext, '{index}')) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.application_id = {app} AND d.status = 'READY' {docs}
ORDER BY c.text <@> to_bm25query(:qtext, '{index}')
LIMIT :top_k
"""
# Chế độ paper: thêm điều kiện này vào WHERE; danh sách id đi qua bound parameter :docs.
DOCS_FILTER = "AND c.document_id = ANY(CAST(:docs AS uuid[]))"


def pgts_sql(document_ids: Sequence[uuid.UUID] | None = None) -> str:
    """Câu SQL pg_textsearch dùng index CHUNG; có document_ids thì lọc thêm theo tài liệu."""
    return _PGTS_SQL_TEMPLATE.format(index=PGTS_INDEX, app=":app", docs=DOCS_FILTER if document_ids else "")


_PGTS_SQL = pgts_sql()

def app_index_name(application_id: str) -> str:
    """'qasper-test' -> 'ix_chunks_bm25_qasper_test_1a2b3c4d' (tên hợp lệ, <= 63 ký tự, không trùng giữa 2 app)."""
    slug = re.sub(r"[^a-z0-9]+", "_", application_id.lower()).strip("_")[:24]
    digest = hashlib.sha1(application_id.encode("utf-8")).hexdigest()[:8]
    return f"{PGTS_INDEX}_{slug}_{digest}"

def sql_literal(value: str) -> str:
    """Chuỗi -> literal SQL an toàn: nháy đơn bên trong được nhân đôi ('o''brien')."""
    return "'" + value.replace("'", "''") + "'"


def app_index_sql(application_id: str, document_ids: Sequence[uuid.UUID] | None = None) -> str:
    """Câu SQL tìm trong index RIÊNG của 1 app (application_id ghi thẳng dạng literal)."""
    return _PGTS_SQL_TEMPLATE.format(index=app_index_name(application_id), app=sql_literal(application_id),
                                     docs=DOCS_FILTER if document_ids else "")

class KeywordRetriever(Retriever):
    def __init__(self, db: Session, ranking: str = "ts_rank", k1: float = 1.2, b: float = 0.75, strip_prefix: bool = False):
        if ranking not in KEYWORD_RANKINGS:
            raise ValueError(f"Unknown KEYWORD_RANKING '{ranking}' (use: {' | '.join(KEYWORD_RANKINGS)})")
        self.db = db
        self.ranking = ranking
        self.k1 = k1
        self.b = b
        self._own_index: dict[str, bool] = {}   # app -> có index riêng không (hỏi DB 1 lần)
        self.strip_prefix = strip_prefix

    def retrieve(self, query: str, application_id: str, top_k: int,
                 document_ids: Sequence[uuid.UUID] | None = None) -> list[RetrievedChunk]:
        docs = list(document_ids) if document_ids else None         # [] cũng coi như "không lọc"
        scope = {"docs": docs} if docs else {}
        if self.strip_prefix:
            query = strip_paper_prefix(query)
        or_query = build_or_query(query)
        
        if or_query is None:
            return []
        if self.ranking == "bm25":
            return self._retrieve_bm25(" ".join(extract_terms(query)), application_id, top_k, **scope)
        if self.ranking == "pg_textsearch":
            return self._retrieve_pg_textsearch(" ".join(extract_terms(query)), application_id, top_k, **scope)

        t0 = time.perf_counter()
        # TS_CONFIG là hằng số trong code (không phải input người dùng) -> an toàn khi dùng literal.
        # Câu hỏi của người dùng vẫn đi qua bound parameter như bình thường.
        ts_query = func.to_tsquery(literal_column(f"'{TS_CONFIG}'::regconfig"), or_query)
        rank = func.ts_rank_cd(Chunk.search_vector, ts_query).label("rank")
        stmt = (
            select(Chunk, Document.filename, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.application_id == application_id,
                Document.status == DocumentStatus.READY,
                Chunk.search_vector.op("@@")(ts_query),
            )
            .order_by(rank.desc(), Chunk.chunk_index)
            .limit(top_k)
        )

        if docs:
            stmt = stmt.where(Chunk.document_id.in_(docs))
        
        rows = self.db.execute(stmt).all()
        logger.info(
            "keyword_search app=%s query=%r results=%d search_ms=%.0f",
            application_id, or_query, len(rows), (time.perf_counter() - t0) * 1000,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page=chunk.page, chunk_index=chunk.chunk_index, text=chunk.text,
                score=float(r),
            )
            for chunk, filename, r in rows
        ]

    def _retrieve_bm25(self, qtext: str, application_id: str, top_k: int,
                       docs: list[uuid.UUID] | None = None) -> list[RetrievedChunk]:
        t0 = time.perf_counter()

        # N / avgdl / df tính trên các chunk QUA BỘ LỌC: chế độ paper -> thống kê của chính paper đó
        params = {"qtext": qtext, "app": application_id, "k1": self.k1, "b": self.b, "top_k": top_k}

        if docs:
            params["docs"] = docs

        scored = self.db.execute(text(_BM25_SQL.format(docs=DOCS_FILTER if docs else "")), params).all()
        return self._load_chunks(scored, "bm25", application_id, qtext, t0)

    def has_own_index(self, application_id: str) -> bool:
        if application_id not in self._own_index:
            found = self.db.execute(
                text("SELECT 1 FROM pg_indexes WHERE tablename = 'chunks' AND indexname = :name"),
                {"name": app_index_name(application_id)},
            ).first()
            self._own_index[application_id] = found is not None
        return self._own_index[application_id]

    def _retrieve_pg_textsearch(self, qtext: str, application_id: str, top_k: int,
                                docs: list[uuid.UUID] | None = None) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        # Có index riêng -> PHẢI dùng nó: khi index riêng tồn tại mà vẫn gọi tên index chung,
        # Postgres chọn index riêng rồi báo lỗi "query specifies index ... but planner chose ...".
        # Chế độ paper: điểm BM25 (IDF, avgdl) vẫn tính trên CẢ index, chỉ lọc kết quả theo tài liệu.
        if self.has_own_index(application_id):
            sql, params, label = app_index_sql(application_id, docs), {"qtext": qtext, "top_k": top_k}, "pg_textsearch/app"
        else:
            sql, params, label = pgts_sql(docs), {"qtext": qtext, "app": application_id, "top_k": top_k}, "pg_textsearch"
        if docs:
            params["docs"] = docs
        scored = self.db.execute(text(sql), params).all()
        scored = [(chunk_id, score) for chunk_id, score in scored if score > 0]
        return self._load_chunks(scored, label, application_id, qtext, t0)

    def _load_chunks(self, scored: list, label: str, application_id: str, qtext: str,
                     t0: float) -> list[RetrievedChunk]:
        """(chunk_id, score) đã xếp hạng -> RetrievedChunk, GIỮ NGUYÊN thứ tự điểm."""
        if not scored:
            return []
        rows = self.db.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_([chunk_id for chunk_id, _ in scored]))
        ).all()
        by_id = {chunk.id: (chunk, filename) for chunk, filename in rows}
        logger.info(
            "keyword_search(%s) app=%s terms=%r results=%d search_ms=%.0f",
            label, application_id, qtext, len(scored), (time.perf_counter() - t0) * 1000,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                page=chunk.page, chunk_index=chunk.chunk_index, text=chunk.text,
                score=float(score),
            )
            for chunk_id, score in scored
            for chunk, filename in [by_id[chunk_id]]
        ]
