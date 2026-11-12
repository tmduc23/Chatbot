// Chạy: npm test   (Node có sẵn trình chạy test: node --test, không cần cài thêm thư viện)
import assert from "node:assert/strict";
import { test } from "node:test";
import { cleanExcerpt, sourceLabel, splitAnswer } from "./citations.ts";


test("câu không có trích dẫn -> 1 đoạn chữ", () => {
  assert.deepEqual(splitAnswer("Xin chào"), [{ kind: "text", text: "Xin chào" }]);
});

test("tách chữ và [Source n] đúng thứ tự", () => {
  assert.deepEqual(splitAnswer("Bạn ấy dùng Docker [Source 1]."), [
    { kind: "text", text: "Bạn ấy dùng Docker " },
    { kind: "cite", number: 1 },
    { kind: "text", text: "." },
  ]);
});

test("nhiều trích dẫn liền nhau, ở đầu và cuối câu", () => {
  assert.deepEqual(splitAnswer("[Source 2]A[Source 1][Source 3]"), [
    { kind: "cite", number: 2 },
    { kind: "text", text: "A" },
    { kind: "cite", number: 1 },
    { kind: "cite", number: 3 },
  ]);
});

test("số nhiều chữ số", () => {
  assert.deepEqual(splitAnswer("x [Source 12]")[1], { kind: "cite", number: 12 });
});

test("gọi 2 lần cho cùng kết quả (regex /g không bị 'nhớ' vị trí cũ)", () => {
  const answer = "A [Source 1]";
  assert.deepEqual(splitAnswer(answer), splitAnswer(answer));
});

test("dạng chưa chuẩn hoá không bị coi là trích dẫn", () => {
  // backend đã chuẩn hoá về [Source n]; frontend không đoán các kiểu khác
  assert.deepEqual(splitAnswer("[source 1]"), [{ kind: "text", text: "[source 1]" }]);
});

test("nhãn nguồn có và không có số trang", () => {
  assert.equal(sourceLabel("CV.pdf", 2), "CV.pdf · trang 2");
  assert.equal(sourceLabel("notes.md", null), "notes.md");
});


// ---- Phase 10b: đoạn trích ----

test("đoạn trích: bỏ khoảng trắng thừa, giữ xuống dòng", () => {
  assert.equal(cleanExcerpt("  Kỹ năng:   Python,\t Docker  \r\n- FastAPI  "), "Kỹ năng: Python, Docker\n- FastAPI");
});

test("đoạn trích: nhiều dòng trống -> 1 dòng trống", () => {
  assert.equal(cleanExcerpt("Học vấn\n\n\n\nHUST"), "Học vấn\n\nHUST");
});

test("đoạn trích: tin nhắn cũ không có text -> chuỗi rỗng", () => {
  assert.equal(cleanExcerpt(null), "");
  assert.equal(cleanExcerpt(undefined), "");
});