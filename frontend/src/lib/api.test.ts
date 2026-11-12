import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { API_URL, ApiError, api, authHeaders, friendlyError, parseDetail } from "./api.ts";
import { parseCredentials } from "./storage.ts";
import type { Credentials } from "./types.ts";

const KEY: Credentials = { kind: "key", key: "k" };   // Phase 12b: hàm quản trị nhận Credentials thay cho chuỗi key

// Thay fetch thật bằng hàm giả: ghi lại request và trả response định sẵn -> không cần backend
const realFetch = globalThis.fetch;
type Call = { url: string; init: RequestInit };
let calls: Call[] = [];

function mockFetch(status: number, body: unknown) {
  calls = [];
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

test("parseDetail: detail là chuỗi", () => {
  assert.equal(parseDetail({ detail: "Application 'x' not found" }, "?"), "Application 'x' not found");
});

test("parseDetail: lỗi validate 422 của FastAPI (detail là mảng)", () => {
  const body = { detail: [{ loc: ["body", "query"], msg: "String should have at least 1 character" }] };
  assert.equal(parseDetail(body, "?"), "String should have at least 1 character");
});

test("parseDetail: body lạ -> dùng fallback", () => {
  assert.equal(parseDetail(null, "Bad Gateway"), "Bad Gateway");
  assert.equal(parseDetail({ error: "x" }, "Bad Gateway"), "Bad Gateway");
});

test("chat gửi đúng URL, method và JSON body", async () => {
  mockFetch(200, { session_id: "s1", answer: "ok" });
  const res = await api.chat("cv", "Bạn ấy học ở đâu?", null);

  assert.equal(res.session_id, "s1");
  assert.equal(calls[0].url, `${API_URL}/api/v1/chat`);
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].init.body)), {
    application_id: "cv", query: "Bạn ấy học ở đâu?", session_id: null,
  });
});

test("API quản trị gửi header X-API-Key; API công khai thì không", async () => {
  mockFetch(200, []);
  await api.listDocuments({ kind: "key", key: "s3cret" }, "cv");
  assert.equal((calls[0].init.headers as Record<string, string>)["X-API-Key"], "s3cret");

  mockFetch(200, []);
  await api.listApplications();
  assert.equal((calls[0].init.headers as Record<string, string>)["X-API-Key"], undefined);
});

test("upload gửi FormData và KHÔNG tự đặt Content-Type", async () => {
  mockFetch(201, { id: "d1", status: "READY" });
  await api.uploadDocument(KEY, "cv", new File(["hello"], "cv.txt"));

  const init = calls[0].init;
  assert.ok(init.body instanceof FormData);
  assert.equal((init.body as FormData).get("application_id"), "cv");
  assert.equal(((init.body as FormData).get("file") as File).name, "cv.txt");
  assert.equal((init.headers as Record<string, string>)["Content-Type"], undefined);
});

test("query string được mã hoá (application_id có ký tự lạ)", async () => {
  mockFetch(200, []);
  await api.listSessions(KEY, "a&b");
  assert.equal(calls[0].url, `${API_URL}/api/v1/chat/sessions?application_id=a%26b&limit=20`);
});

test("DELETE trả 204 -> không đọc body, không lỗi", async () => {
  mockFetch(204, null);
  assert.equal(await api.deleteDocument(KEY, "cv", "d1"), undefined);
  assert.equal(calls[0].init.method, "DELETE");
});

test("HTTP lỗi -> ApiError có status và detail của backend", async () => {
  mockFetch(404, { detail: "Chat session not found" });
  await assert.rejects(api.getSession("cv", "s1"), (err: unknown) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.status, 404);
    assert.equal(err.detail, "Chat session not found");
    return true;
  });
});

test("backend không chạy (fetch ném lỗi) -> ApiError status 0", async () => {
  globalThis.fetch = (async () => {
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  await assert.rejects(api.listApplications(), (err: unknown) => err instanceof ApiError && err.status === 0);
});

test("friendlyError: câu tiếng Việt theo status", () => {
  assert.match(friendlyError(new ApiError(0, "Network error")), /Không kết nối được backend/);
  assert.match(friendlyError(new ApiError(401, "Invalid or missing X-API-Key")), /đăng nhập/);
  assert.match(friendlyError(new ApiError(403, "Editor role required")), /không đủ quyền/);
  assert.match(friendlyError(new ApiError(503, "AI service is temporarily unavailable")), /quota/);
  assert.match(friendlyError(new ApiError(409, "already uploaded")), /already uploaded/);
  assert.match(friendlyError(new Error("boom")), /Console/);
});

// ---- Phase 12b ----

test("Phase 12b: đăng nhập rồi -> gửi Authorization: Bearer, không gửi X-API-Key", async () => {
  const auth: Credentials = { kind: "token", token: "abc.def.ghi", email: "duc@example.com" };
  assert.deepEqual(authHeaders(auth), { Authorization: "Bearer abc.def.ghi" });
  assert.deepEqual(authHeaders(undefined), {});

  mockFetch(200, { session_id: "s1" });
  await api.chat("cv", "hi", null, auth);
  const headers = calls[0].init.headers as Record<string, string>;
  assert.equal(headers.Authorization, "Bearer abc.def.ghi");
  assert.equal(headers["X-API-Key"], undefined);
  assert.equal(headers["Content-Type"], "application/json");
});

test("Phase 12b: login gửi email + mật khẩu dạng JSON, không kèm header đăng nhập", async () => {
  mockFetch(200, { access_token: "t", token_type: "bearer", expires_in: 3600 });
  const res = await api.login("duc@example.com", "a-long-password");
  assert.equal(res.access_token, "t");
  assert.equal(calls[0].url, `${API_URL}/api/v1/auth/login`);
  assert.deepEqual(JSON.parse(String(calls[0].init.body)), { email: "duc@example.com", password: "a-long-password" });
  assert.equal((calls[0].init.headers as Record<string, string>).Authorization, undefined);
});

test("Phase 12b: parseCredentials bỏ qua dữ liệu hỏng trong sessionStorage", () => {
  assert.deepEqual(parseCredentials('{"kind":"key","key":"k"}'), { kind: "key", key: "k" });
  assert.deepEqual(parseCredentials('{"kind":"token","token":"t","email":"a@b.c"}'),
    { kind: "token", token: "t", email: "a@b.c" });
  for (const bad of [null, "", "not json", '{"kind":"token"}', '{"kind":"key","key":""}', '"s3cret"']) {
    assert.equal(parseCredentials(bad), null);
  }
});

test("Phase 12c: 429 -> báo số giây phải chờ lấy từ backend", () => {
  assert.match(friendlyError(new ApiError(429, "Too many requests. Try again in 42 seconds.")), /42 giây/);
  assert.match(friendlyError(new ApiError(429, "Too Many Requests")), /một lát/);
});

test("Phase 12d: lỗi 5xx kèm mã X-Request-ID để tra log; lỗi 4xx thì không", async () => {
  calls = [];
  globalThis.fetch = (async () => new Response(JSON.stringify({ detail: "Internal server error" }), {
    status: 500, headers: { "Content-Type": "application/json", "X-Request-ID": "3f9c2a1b7d4e4f0a" },
  })) as typeof fetch;
  await assert.rejects(api.listApplications(), (err: unknown) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.requestId, "3f9c2a1b7d4e4f0a");
    assert.match(friendlyError(err), /mã lỗi: 3f9c2a1b7d4e4f0a/);
    return true;
  });
  assert.doesNotMatch(friendlyError(new ApiError(404, "not found", "abc12345")), /mã lỗi/);
});
