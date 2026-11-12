"""
MỤC ĐÍCH:  Các công thức đánh giá retrieval - hàm thuần, không DB, không API 

Với MỘT câu hỏi, đầu vào là:
    ranks       = hạng (1, 2, 3, ...) của các chunk ĐÚNG trong danh sách kết quả, vd. [2, 5]
    n_relevant  = tổng  số chunk đúng có trong DB cho câu đó (dù có tìm thấy hay không)

    Hit@k       = 1 nếu có ít nhất 1 chunk trong top k, ngược lại 0
    Recall@k    = (số chunk đúng trong top k) / n_relevant
    Precision   = (số chunk đúng trong top k) / k
    RR          = 1 / hạng của chunk đúng ĐẦU TIÊN (không có -> 0)
    nDCG@k      = DCG@k / DCG lý tưởng; DCG@k = tổng 1/log2(hạng + 1) của các chunk đúng trong top k
                  (chỉ số chính của bảng xếp hạng MTEB/BEIR)

Điểm của cả bộ = trung bình các câu (MRR = mean reciprocal rank).    

Chấm câu trả lời theo kiểu SQuAD (so TỪ với đáp án mẫu):
    token_f1     = F1 giữa các từ của câu trả lời và đáp án mẫu (script chấm chính thức của SQuAD)
    token_recall = bao nhiêu phần từ của đáp án mẫu có trong câu trả lời (không phạt trả lời dài)

Khoảng tin cậy (bootstrap_ci): mẫu nhỏ (40 câu) thì 1 câu đổi là số nhảy 2.5%, nên mọi chỉ số
được báo kèm khoảng 95%: "trên một bộ câu hỏi khác cùng loại, điểm có thể rơi vào đâu".
""" 

import math
from collections.abc import Sequence
import unicodedata
import re
from collections import Counter
from collections.abc import Sequence
import random

def hit_at_k(ranks: Sequence[int], k: int) -> float:
    return 1 if any(r <= k for r in ranks) else 0



def recall_at_k(ranks: Sequence[int], n_relevant: int, k: int) -> float:
    if n_relevant <= 0:
        raise ValueError("recall cần ít nhất 1 chunk đúng (câu có đáp án mà DB không có chunk nào khớp?)")
    return min(1.0, sum(1 for r in ranks if r <=k) / n_relevant)


def precision_at_k(ranks: Sequence[int], k: int) -> float:
    # Chia cho k chứ không chia cho số kết quả: trả ít hơn k kết quả vẫn bị tính là "ô trống"
    return sum(1 for r in ranks if r <= k) / k


def reciprocal_rank(ranks: Sequence[int], k: int | None = None) -> float:
    found = [r for r in ranks if k is None or r <= k]
    return 1.0 / min(found) if found else 0

def ndcg_at_k(ranks: Sequence[int], n_relevant: int, k: int) -> float:
    """Như Recall@k nhưng chunk đúng càng xếp CAO càng được nhiều điểm: hạng 1 = 1.0, hạng 2 = 0.63, hạng 3 = 0.5...

    Chia cho DCG "lý tưởng" (mọi chunk đúng đứng đầu danh sách) -> điểm luôn trong khoảng 0..1.
    """
    if n_relevant <= 0:
        raise ValueError("nDCG cần ít nhất 1 chunk đúng")
    dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= k)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / ideal

# Trung bình thời gian 1 câu
def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """ Percentile kiểu "nearest_rank": p95 = giá trị mà 95% số lần đo nhỏ hơn hoặc bằng."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]

# Tính trung bình các case
def latency_summary(value_ms: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean_ms": mean(value_ms),
        "p50_ms": percentile(value_ms, 50),
        "p95_ms": percentile(value_ms, 95),
        "max_ms": max(value_ms) if value_ms else None
    } 

def rate(flags: Sequence[bool]) -> float | None:
    """
    Tỉ lệ True. None khi không có câu nào (không chia cho 0, không báo 0% sai)
    """
    return sum(flags) / len(flags) if flags else None


_CITATION = re.compile(r"\[\s*sources?\s*[\d,\s]+\]", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def answer_tokens(text: str) -> list[str]:
    """Tách từ như script chấm SQuAD: chữ thường, bỏ dấu câu, bỏ a/an/the. Thêm: bỏ [Source n].

    Bỏ dấu câu bằng nhóm Unicode "P" (mọi loại dấu câu) chứ không xoá ký tự ngoài ASCII,
    nên chữ có dấu tiếng Việt ("Bách") vẫn giữ nguyên.
    """
    text = _CITATION.sub(" ", unicodedata.normalize("NFC", text)).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return _ARTICLES.sub(" ", text).split()


def token_f1(answer: str, reference: str) -> float:
    pred, gold = answer_tokens(answer), answer_tokens(reference)
    if not pred or not gold:
        return float(pred == gold)          # như SQuAD: cả 2 rỗng = 1, chỉ 1 bên rỗng = 0
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def token_recall(answer: str, reference: str) -> float:
    pred, gold = answer_tokens(answer), answer_tokens(reference)
    if not gold:
        return float(not pred)
    return sum((Counter(pred) & Counter(gold)).values()) / len(gold)


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, n_resamples: int = 2000,
                 seed: int = 0) -> list[float] | None:
    """Khoảng tin cậy của TRUNG BÌNH bằng percentile bootstrap. None khi có ít hơn 2 giá trị.

    Lặp 2000 lần: bốc ngẫu nhiên CÓ HOÀN LẠI đúng n câu từ chính n câu đã chấm, tính trung bình.
    Xếp 2000 trung bình đó, bỏ 2.5% thấp nhất và 2.5% cao nhất -> phần còn lại là khoảng 95%.
    seed cố định: chạy lại trên cùng kết quả ra đúng cùng khoảng. Trả list (không phải tuple) để lưu JSON.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    cut = int((1 - confidence) / 2 * n_resamples)
    return [means[cut], means[n_resamples - 1 - cut]]

