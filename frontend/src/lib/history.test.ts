import assert from "node:assert/strict";
import { test } from "node:test";
import { makeTitle, parseSessions, relativeTime, removeSession, upsertSession, type SavedSession } from "./history.ts";

const s = (id: string, title = id, updatedAt = "2026-10-20T10:00:00Z"): SavedSession => ({ id, title, updatedAt });

test("history: phiên mới được đưa lên đầu", () => {
  const list = upsertSession([s("a"), s("b")], s("c"));
  assert.deepEqual(list.map((x) => x.id), ["c", "a", "b"]);
});

test("history: hỏi tiếp phiên cũ -> lên đầu, giữ tiêu đề cũ, cập nhật thời gian", () => {
  const list = upsertSession([s("a", "Câu đầu"), s("b")], s("b", "Câu thứ hai", "2026-10-20T11:00:00Z"));
  assert.deepEqual(list.map((x) => x.id), ["b", "a"]);
  assert.equal(list[0].title, "b");
  assert.equal(list[0].updatedAt, "2026-10-20T11:00:00Z");
});

test("history: giới hạn số phiên, bỏ phiên cũ nhất", () => {
  const list = upsertSession([s("a"), s("b"), s("c")], s("d"), 3);
  assert.deepEqual(list.map((x) => x.id), ["d", "a", "b"]);
});

test("history: xoá 1 phiên khỏi danh sách", () => {
  assert.deepEqual(removeSession([s("a"), s("b")], "a").map((x) => x.id), ["b"]);
});

test("history: dữ liệu localStorage hỏng -> danh sách rỗng, phần tử sai kiểu bị bỏ", () => {
  assert.deepEqual(parseSessions(null), []);
  assert.deepEqual(parseSessions("{not json"), []);
  assert.deepEqual(parseSessions('{"id":"a"}'), []);
  assert.deepEqual(parseSessions(JSON.stringify([s("a"), { id: 1 }, null])), [s("a")]);
});

test("history: tiêu đề 1 dòng, cắt câu quá dài", () => {
  assert.equal(makeTitle("  Bạn ấy\nhọc   ở đâu? "), "Bạn ấy học ở đâu?");
  const long = makeTitle("x".repeat(200));
  assert.equal(long.length, 80);
  assert.ok(long.endsWith("…"));
});

test("history: thời gian tương đối", () => {
  const now = new Date("2026-10-20T12:00:00Z");
  assert.equal(relativeTime("2026-10-20T11:59:40Z", now), "vừa xong");
  assert.equal(relativeTime("2026-10-20T11:55:00Z", now), "5 phút trước");
  assert.equal(relativeTime("2026-10-20T09:00:00Z", now), "3 giờ trước");
  assert.equal(relativeTime("2026-10-19T08:00:00Z", now), "hôm qua");
});