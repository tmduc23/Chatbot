"""Câu trả lời phải cùng ngôn ngữ với câu hỏi: prompt ghi rõ ngôn ngữ (answer-v3)."""
from app.rag.prompts.answer import build_answer_prompt, no_answer_message, question_language


def test_question_language() -> None:
    assert question_language("Bạn ấy học trường nào?") == "Vietnamese"
    assert question_language("Which university did Duc attend?") == "English"
    assert question_language("Duc dùng Docker không?") == "Vietnamese"   # trộn: có 1 chữ có dấu là tiếng Việt


def test_answer_prompt_ends_with_language_of_question() -> None:
    en = build_answer_prompt("What are three examples of measurement?", "[Source 1] Thời gian, bộ nhớ...")
    vi = build_answer_prompt("Bạn ấy dùng công cụ gì?", "[Source 1] Docker, FastAPI")
    assert en.endswith("Answer in English.")      # sources tiếng Việt cũng không đổi ngôn ngữ trả lời
    assert vi.endswith("Answer in Vietnamese.")
    assert "Question:\nBạn ấy dùng công cụ gì?" in vi


def test_no_answer_message_uses_same_detection() -> None:
    assert no_answer_message("Bạn ấy bao nhiêu tuổi?").startswith("Xin lỗi")
    assert no_answer_message("How old is he?").startswith("Sorry")