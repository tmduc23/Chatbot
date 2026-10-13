"""
MỤC ĐÍCH: Uni test cho việc tách câu hỏi thành từ khóa ( không cần database)
"""

from app.rag.retrieval.keyword_retriever import build_or_query, extract_terms


def test_removes_english_stopwords_and_punctuation() -> None:
    assert extract_terms("What is his experience with Docker?") == ["experience", "docker"]


def test_removes_vietnamese_stopwords_keeps_diacritics() -> None:
    assert extract_terms("Kinh nghiệm của anh ấy là gì?") == ["kinh", "nghiệm"]

def test_lowercases_and_deduplicates() -> None:
    assert extract_terms("Docker docker DOCKER Compose") == ["docker", "compose"]

def test_splits_on_underscore_and_symbols() -> None:
    assert extract_terms("snake_case, C#/F# & k8s") == ["snake", "case", "k8s"]

def test_builds_or_query() -> None:
    assert build_or_query("Docker and Kubernetes") == "docker | kubernetes"

def test_only_stopwords_gives_none() -> None:
    assert build_or_query("what is it?") is None