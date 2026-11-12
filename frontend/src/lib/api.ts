// =====================================================================
// api.ts: CẦU NỐI DUY NHẤT giữa frontend (Next.js) và backend (FastAPI)
// =====================================================================
// Mục đích:
//   - Mọi lần giao diện cần dữ liệu đều đi qua file này. Component KHÔNG tự gọi fetch.
//   - Component chỉ cần viết:  await api.chat(appId, text, sessionId)
//     File này lo hết phần còn lại: ghép URL, thêm header, đóng gói dữ liệu, báo lỗi.
//
// Cấu trúc file (3 lớp):
//   1. Cấu hình    API_URL: địa chỉ backend (NEXT_PUBLIC_API_URL, mặc định localhost:8000)
//   2. Xử lý lỗi   ApiError: một kiểu lỗi chung (status + detail)
//                  parseDetail: lấy câu lỗi từ {"detail": ...} của FastAPI (chuỗi hoặc mảng 422)
//                  friendlyError: đổi mã lỗi thành câu tiếng Việt để hiện lên màn hình
//   3. Gửi request request<T>: hàm chung (đóng gói JSON/FormData, X-API-Key, 204, lỗi)
//                  api: 8 hàm, mỗi hàm ứng với 1 endpoint backend
//                    công khai: listApplications, chat, getSession          (page.tsx)
//                    cần key  : createApplication, listDocuments,
//                               uploadDocument, deleteDocument, listSessions (admin/page.tsx)
//
// Lưu ý:
//   - Trình duyệt gọi THẲNG backend, nên CORS_ORIGINS trong .env phải có địa chỉ frontend.
//   - Admin key chỉ được truyền vào lúc gọi hàm (người dùng nhập ở trang admin),
//     KHÔNG bao giờ ghi cứng trong code hay trong biến NEXT_PUBLIC_*.
//   - Object trong JS/TS giống dict trong Python:
//       const obj = { query: "Dice bao nhiêu?" };
//       JSON.stringify(obj)  ->  '{"query":"Dice bao nhiêu?"}'   (chuỗi để gửi đi)
// =====================================================================

import type {
  Application, ChatResponse, Credentials, DocumentItem, Me, SessionDetail, SessionSummary, TokenResponse,
} from "./types.ts";

// Mục đích: lấy địa chỉ backend, mặc định localhost:8000, và bỏ dấu / thừa ở cuối
// để khi ghép không ra "//api/v1".

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;  // 0 = không có phản hồi (backend tắt, sai URL, bị CORS chặn)
  detail: string;  // thông báo gốc của backend

  requestId: string | null;  // Phase 12d: header X-Request-ID -> tìm đúng dòng log của request lỗi

  constructor(status: number, detail: string, requestId: string | null = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
  }
}

// FastAPI trả {"detail": "..."} cho lỗi nghiệp vụ, nhưng lỗi validate (422) thì detail là 1 MẢNG:
// [{"loc": ["body", "query"], "msg": "String should have at least 1 character", ...}]
export function parseDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : ""))
        .filter(Boolean);
      if (msgs.length) return msgs.join("; ");
    }
  }
  return fallback;
}

// Câu tiếng Việt cho người dùng. Chi tiết kỹ thuật (detail) vẫn giữ trong ApiError để xem ở Console.
export function friendlyError(err: unknown): string {
  if (err instanceof ApiError) {
    // Phase 12d: lỗi phía server -> kèm mã để tra log: docker compose logs backend | grep <mã>
    const code = err.status >= 500 && err.requestId ? ` (mã lỗi: ${err.requestId})` : "";
    return describe(err) + code;
  }
  return "Có lỗi không mong muốn. Xem Console (F12) để biết chi tiết.";
}

function describe(err: ApiError): string {
  switch (err.status) {
    case 0: return "Không kết nối được backend. Kiểm tra `docker compose ps` xem backend có đang chạy không.";
    // Phase 12b: 401 = "bạn là ai?" (chưa đăng nhập / token hết hạn / sai key), 403 = "không đủ quyền"
    case 401: return "Cần đăng nhập (hoặc phiên đăng nhập đã hết hạn, hoặc sai Admin API key).";
    case 403: return "Tài khoản này không đủ quyền làm việc này (hoặc backend chưa đặt ADMIN_API_KEY).";
    case 404: return "Không tìm thấy (application, tài liệu hoặc phiên chat đã bị xoá?).";
    case 409: return `Bị trùng: ${err.detail}`;
    case 413: return "File quá lớn.";
    case 415: return "Loại file không hỗ trợ. Chỉ nhận PDF, DOCX, TXT, MD.";
    case 422: return `Dữ liệu không hợp lệ: ${err.detail}`;
    // Phase 12c: backend gửi "Too many requests. Try again in 42 seconds." -> lấy số giây ra
    case 429: {
      const seconds = /(\d+) seconds?/.exec(err.detail)?.[1];
      return `Bạn gửi hơi nhanh. Đợi ${seconds ? `${seconds} giây` : "một lát"} rồi thử lại nhé.`;
    }
    case 503: return "Dịch vụ AI đang bận hoặc hết quota. Thử lại sau ít phút.";
    default: return `Lỗi ${err.status}: ${err.detail}`;
  }
}

// gom các request vào 1 tham số
type RequestOptions = {
  method?: string;          // ? = NotRequired
  body?: unknown;          // object -> gửi JSON; FormData -> gửi multipart (upload file)
  auth?: Credentials;      // Phase 12b: token đăng nhập (Bearer) hoặc X-API-Key
};

// Đăng nhập bằng email -> "Authorization: Bearer <token>"; dùng key -> "X-API-Key: <key>"
export function authHeaders(auth: Credentials | undefined): Record<string, string> {
  if (!auth) return {};
  if (auth.kind === "token") return { Authorization: `Bearer ${auth.token}` };
  return { "X-API-Key": auth.key };
}


export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = authHeaders(options.auth);
  let body: BodyInit | undefined;
  // Đóng gói dữ liệu trước khi gửi
  if (options.body instanceof FormData) {          
    body = options.body; // KHÔNG tự đặt Content-Type: trình duyệt tự thêm boundary cho multipart
  } else if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";   // ghi nhãn nội dung là JSON
    body = JSON.stringify(options.body);            // đổi object thành chuỗi có chữ
  }

  // Gửi thư đi
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/v1${path}`, { method: options.method ?? "GET", headers, body });
  } catch {
    // fetch chỉ ném lỗi khi KHÔNG có phản hồi: backend tắt, sai URL, bị CORS chặn
    throw new ApiError(0, "Network error");
  }

  // Kiểm tra thư
  if (response.status === 204) return undefined as T; // DELETE thành công: không có body

  // Đọc nội dung thư
  // Nội dung là JSON -> object
  // Không phải JSON -> data thành null
  const data: unknown = await response.json().catch(() => null);

  // Mã (200-299) ->  trả dữ liệu (câu trả lời chat)
  // Mã lỗi (404, 422, 503) -> ApiError(mã, câu lỗi)
  if (!response.ok) {
    throw new ApiError(response.status, parseDetail(data, response.statusText), response.headers.get("X-Request-ID"));
  }
  return data as T;
}

const q = encodeURIComponent;         // gán hàm vào biến

export const api = {
  // ---- đăng nhập (Phase 12b) ----
  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", { method: "POST", body: { email, password } }),
  me: (auth: Credentials) => request<Me>("/auth/me", { auth }),

  // ---- app công khai: ai cũng gọi được; app riêng tư: cần đăng nhập (auth) ----
  listApplications: (auth?: Credentials) => request<Application[]>("/applications", { auth }),  // lấy danh sách application

  // Gửi câu hỏi, kết quả là ChatResponse
  chat: (applicationId: string, query: string, sessionId: string | null, auth?: Credentials) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: { application_id: applicationId, query, session_id: sessionId },      // object cần gửi
      auth,
    }),

  // mở lại phiên chat
  getSession: (applicationId: string, sessionId: string, auth?: Credentials) =>
    request<SessionDetail>(`/chat/sessions/${q(sessionId)}?application_id=${q(applicationId)}`, { auth }),

  // ---- quản trị: editor của application, superadmin, hoặc X-API-Key ----
  createApplication: (auth: Credentials, body: { id: string; name: string; description: string | null }) =>
    request<Application>("/applications", { method: "POST", body, auth }),

  listDocuments: (auth: Credentials, applicationId: string) =>
    request<DocumentItem[]>(`/documents?application_id=${q(applicationId)}`, { auth }),

  uploadDocument: (auth: Credentials, applicationId: string, file: File) => {
    const form = new FormData();
    form.append("application_id", applicationId); // tên ô phải khớp Form(...) / File(...) ở backend
    form.append("file", file);
    return request<DocumentItem>("/documents/upload", { method: "POST", body: form, auth });
  },

  deleteDocument: (auth: Credentials, applicationId: string, documentId: string) =>
    request<void>(`/documents/${q(documentId)}?application_id=${q(applicationId)}`, { method: "DELETE", auth }),

  listSessions: (auth: Credentials, applicationId: string, limit = 20) =>
    request<SessionSummary[]>(`/chat/sessions?application_id=${q(applicationId)}&limit=${limit}`, { auth }),
};
