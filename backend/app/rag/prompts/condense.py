"""
MỤC ĐÍCH: Prompt viết lại câu hỏi NỐI TIẾP thành câu hỏi ĐỘC LẬP (standalone question) - Phase 9

Vì sao cần: retrieval chỉ nhìn thấy câu hỏi hiện tại, không thấy lịch sử chat.
    User: "Bạn ấy từng làm ở công ty nào"   -> retrieval tốt
    User: "Ở đó bạn ấy dùng công nghệ gì"   -> "ở đó" là ở đâu ? retrieval tìm sai
Viết lại: "Ở công ty X, bạn ấy dùng công nghệ gì" -> retrieval tìm đúng chunk
"""


CONDENSE_PROMPT_VERSION = "condense-v1"

CONDENSE_SYSTEM_INSTRUCTION = (
    "You rewrite follow-up questions. Given a conversation and follow-up question, "
    "rewrite the follow-up question in to ONE standalone question that can be understood "
    "without the conversation. Replace pronouns and references (he, she, it, there, that project, "
    "ở đó, dự án đó, bạn ấy, ...) with that they refer to. "
    "If the follow-up question is already standable, return it unchanged. "
    "Keep the same language as the follow-up question. "
    "Do NOT answer the question. Return ONLY the rewritten question, nothing else."
)


MAX_TURN_CHARS = 500            # cắt bớt câu trả lời dài trong lịch sử: chỉ cần đủ để hiểu "nó/ở đó" là gì


def build_condense_prompt(history: list[tuple[str, str]], question: str) -> str:
    """history: [(role, content), ...] theo thứ tự cũ -> mới, role = "user" | assistant"."""
    lines = []
    for role, content in history:
        text = content.strip()
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "..."
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return (
        "Conversation:\n" + "\n".join(lines) +
        f"\n\nFollow-up question: {question}\n\nStandalone question:"
    )