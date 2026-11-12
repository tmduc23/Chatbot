import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_UPLOAD_MB, checkUpload, isProcessing, uploadNotice } from "./upload.ts";

test("file hợp lệ -> null", () => {
  assert.equal(checkUpload("CV.pdf", 1000), null);
  assert.equal(checkUpload("ghi-chu.MD", 10), null); // không phân biệt hoa thường
});

test("sai đuôi hoặc không có đuôi", () => {
  assert.match(checkUpload("anh.png", 10)!, /chỉ nhận/);
  assert.match(checkUpload("README", 10)!, /chỉ nhận/);
});

test("file rỗng và file quá lớn", () => {
  assert.match(checkUpload("a.txt", 0)!, /rỗng/);
  assert.match(checkUpload("a.pdf", MAX_UPLOAD_MB * 1024 * 1024 + 1)!, /lớn hơn/);
  assert.equal(checkUpload("a.pdf", MAX_UPLOAD_MB * 1024 * 1024), null); // đúng giới hạn vẫn được
});

// Phase 12a
test("PENDING / PROCESSING là đang xử lý, READY / FAILED thì không", () => {
  assert.equal(isProcessing({ status: "PENDING" }), true);
  assert.equal(isProcessing({ status: "PROCESSING" }), true);
  assert.equal(isProcessing({ status: "READY" }), false);
  assert.equal(isProcessing({ status: "FAILED" }), false);
});

test("thông báo sau upload: 202 PENDING không bị hiện thành lỗi", () => {
  assert.equal(uploadNotice({ status: "PENDING", filename: "cv.pdf", error_message: null }).kind, "ok");
  assert.match(uploadNotice({ status: "PENDING", filename: "cv.pdf", error_message: null }).text, /đang xử lý/);
  assert.equal(uploadNotice({ status: "READY", filename: "cv.pdf", error_message: null }).kind, "ok");
  const failed = uploadNotice({ status: "FAILED", filename: "scan.pdf", error_message: "No text extracted" });
  assert.equal(failed.kind, "error");
  assert.match(failed.text, /No text extracted/);
});
