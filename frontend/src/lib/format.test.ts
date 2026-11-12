import assert from "node:assert/strict";
import { test } from "node:test";
import { splitBold } from "./format.ts";

test("bold: tách **chữ đậm** khỏi chữ thường", () => {
  assert.deepEqual(splitBold("Dice **82.62%** trên BUSI"), [
    { text: "Dice ", bold: false },
    { text: "82.62%", bold: true },
    { text: " trên BUSI", bold: false },
  ]);
});

test("bold: không có ** hoặc ** lẻ thì giữ nguyên", () => {
  assert.deepEqual(splitBold("a * b"), [{ text: "a * b", bold: false }]);
  assert.deepEqual(splitBold("a **b"), [{ text: "a **b", bold: false }]);
});