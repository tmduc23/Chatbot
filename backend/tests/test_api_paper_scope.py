"""
MỤC ĐÍCH: Test API /chat với document_ids (chế độ paper). Dùng lại client/fixture của test_api.py.

  - document_ids hợp lệ -> mọi nguồn trả về thuộc đúng document đó;
  - document_ids không có trong app (kể cả id của app khác) -> 404, không gọi LLM;
  - danh sách rỗng -> 422 (null mới là "cả app").
"""
import uuid

from fastapi.testclient import TestClient

from tests.test_api import (  # noqa: F401  (fixture pytest phải được import vào module mới dùng được)
    API, _new_app, _upload, app_ids, client, cv_app, db, llm, settings,
)


def _doc_id(client: TestClient, app_id: str, filename: str) -> str:
    docs = client.get(f"{API}/documents", params={"application_id": app_id}).json()
    return next(d["id"] for d in docs if d["filename"] == filename)


def test_chat_with_document_ids_only_cites_that_document(client: TestClient, cv_app: str) -> None:
    job = _doc_id(client, cv_app, "job.txt")
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "What did Duc use?",
                                          "document_ids": [job]})
    assert r.status_code == 200, r.text
    sources = r.json()["sources"]
    assert sources and {s["document_id"] for s in sources} == {job}


def test_chat_with_unknown_or_foreign_document_is_404(client: TestClient, cv_app: str, app_ids, llm) -> None:
    other = _new_app(client, app_ids)
    assert _upload(client, other, "x.txt", b"Duc used Docker.").status_code == 201
    foreign = _doc_id(client, other, "x.txt")
    for doc in (str(uuid.uuid4()), foreign):
        r = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Docker?", "document_ids": [doc]})
        assert r.status_code == 404, r.text
    assert llm.answer_prompts == []   # dừng trước khi tìm / gọi LLM


def test_chat_with_empty_document_ids_is_422(client: TestClient, cv_app: str) -> None:
    r = client.post(f"{API}/chat", json={"application_id": cv_app, "query": "Docker?", "document_ids": []})
    assert r.status_code == 422
