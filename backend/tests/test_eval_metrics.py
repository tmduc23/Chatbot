"""MỤC ĐÍCH: Unit test cho công thức đánh giá (Phase 11) — hàm thuần, không cần DB."""
import pytest

from app.evaluation.metrics import (
    answer_tokens, bootstrap_ci, hit_at_k, latency_summary, mean, ndcg_at_k, percentile, precision_at_k, rate, recall_at_k,
    reciprocal_rank, token_f1, token_recall,
)


def test_hit_at_k() -> None:
    assert hit_at_k([3], 3) == 1.0
    assert hit_at_k([3], 2) == 0.0
    assert hit_at_k([], 5) == 0.0


def test_recall_counts_only_top_k() -> None:
    # 2 chunk đúng trong DB, tìm được ở hạng 1 và 4
    assert recall_at_k([1, 4], n_relevant=2, k=3) == 0.5
    assert recall_at_k([1, 4], n_relevant=2, k=5) == 1.0


def test_recall_needs_relevant_chunks() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], n_relevant=0, k=5)


def test_precision_divides_by_k() -> None:
    assert precision_at_k([1, 2], 5) == pytest.approx(0.4)
    assert precision_at_k([], 3) == 0.0


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank([4, 2]) == 0.5           # hạng tốt nhất là 2
    assert reciprocal_rank([4], k=3) == 0.0          # ngoài top 3 -> 0
    assert reciprocal_rank([]) == 0.0


def test_ndcg_rewards_higher_rank() -> None:
    assert ndcg_at_k([1], n_relevant=1, k=10) == 1.0
    assert ndcg_at_k([2], n_relevant=1, k=10) == pytest.approx(0.631, abs=1e-3)   # 1/log2(3)
    assert ndcg_at_k([3], n_relevant=1, k=10) == pytest.approx(0.5)               # 1/log2(4)
    assert ndcg_at_k([11], n_relevant=1, k=10) == 0.0                             # ngoài top 10
    assert ndcg_at_k([], n_relevant=1, k=10) == 0.0


def test_ndcg_with_several_relevant_chunks() -> None:
    # 2 chunk đúng ở hạng 1 và 3; lý tưởng là hạng 1 và 2
    expected = (1 + 1 / 2) / (1 + 1 / 1.5849625)
    assert ndcg_at_k([1, 3], n_relevant=2, k=10) == pytest.approx(expected)
    assert ndcg_at_k([1, 2], n_relevant=2, k=10) == pytest.approx(1.0)
    assert ndcg_at_k([1], n_relevant=3, k=1) == 1.0          # top 1 chỉ chứa được 1 chunk -> lý tưởng = 1 chunk
    with pytest.raises(ValueError):
        ndcg_at_k([], n_relevant=0, k=5)


def test_mean_and_rate_of_nothing_is_none() -> None:
    assert mean([]) is None
    assert rate([]) is None                          # không có câu nào: "không biết", không phải 0%
    assert rate([True, False, True, True]) == 0.75


def test_percentile_nearest_rank() -> None:
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None


def test_percentile_does_not_need_sorted_input() -> None:
    assert percentile([100, 10, 50], 50) == 50


def test_latency_summary() -> None:
    s = latency_summary([100.0, 300.0, 200.0])
    assert s == {"mean_ms": 200.0, "p50_ms": 200.0, "p95_ms": 300.0, "max_ms": 300.0}
    assert latency_summary([])["p95_ms"] is None


# ---------------- chấm câu trả lời kiểu SQuAD ----------------

def test_answer_tokens_drop_citations_and_punctuation_keep_vietnamese() -> None:
    assert answer_tokens("Đức học Bách khoa [Source 1].") == ["đức", "học", "bách", "khoa"]
    assert answer_tokens("The Eiffel Tower, in Paris!") == ["eiffel", "tower", "in", "paris"]   # bỏ a/an/the


def test_token_f1_like_official_squad() -> None:
    assert token_f1("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert token_f1("Paris, France", "France") == pytest.approx(2 / 3)   # precision 1/2, recall 1
    assert token_f1("Berlin", "Paris") == 0.0
    assert token_f1("", "Paris") == 0.0


def test_token_recall_does_not_punish_long_answers() -> None:
    answer = "Rollo became the first ruler of Normandy in 911 [Source 2]."
    assert token_recall(answer, "911") == 1.0
    assert token_f1(answer, "911") < 0.25          # F1 bị kéo xuống vì câu trả lời dài hơn đáp án mẫu


# ---------------- khoảng tin cậy ----------------

def test_bootstrap_ci_contains_mean_and_is_reproducible() -> None:
    values = [1.0] * 32 + [0.0] * 8                  # 40 câu, đúng 80%
    lo, hi = bootstrap_ci(values)
    assert 0.6 < lo < 0.8 < hi < 0.95                # công thức ước lượng: 0.8 ± 0.12
    assert bootstrap_ci(values) == [lo, hi]          # seed cố định -> chạy lại ra y hệt


def test_bootstrap_ci_narrows_with_more_questions() -> None:
    small = bootstrap_ci([1.0] * 32 + [0.0] * 8)
    large = bootstrap_ci([1.0] * 320 + [0.0] * 80)   # cùng 80% nhưng gấp 10 lần số câu
    assert (large[1] - large[0]) < (small[1] - small[0]) / 2


def test_bootstrap_ci_edge_cases() -> None:
    assert bootstrap_ci([]) is None and bootstrap_ci([1.0]) is None   # ít hơn 2 câu: không có khoảng
    assert bootstrap_ci([True, True, True]) == [1.0, 1.0]             # mọi câu như nhau: khoảng rỗng
