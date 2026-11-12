import assert from "node:assert/strict";
import { test } from "node:test";
import { parseAccent, parseThemePref, resolveTheme } from "./theme.ts";

test("theme: giá trị đã lưu hợp lệ được giữ, giá trị lạ -> theo máy", () => {
  assert.equal(parseThemePref("dark"), "dark");
  assert.equal(parseThemePref("light"), "light");
  assert.equal(parseThemePref(null), "system");
  assert.equal(parseThemePref("purple"), "system");
});

test("theme: 'system' đi theo hệ điều hành, chọn tay thì bỏ qua hệ điều hành", () => {
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
  assert.equal(resolveTheme("light", true), "light");
  assert.equal(resolveTheme("dark", false), "dark");
});

test("accent: chỉ nhận màu có trong danh sách, còn lại -> blue", () => {
  assert.equal(parseAccent("teal"), "teal");
  assert.equal(parseAccent("neon"), "blue");
  assert.equal(parseAccent(null), "blue");
});