// =====================================================================
// storage.ts: "SỔ GHI NHỚ" của trình duyệt
// =====================================================================
// Mục đích: lưu vài thứ nhỏ ngay trên máy người dùng để tải lại trang (F5) không bị mất:
//   - đang chat phiên nào, đang ở app nào, lịch sử phiên ở thanh bên
//   - chế độ sáng/tối, màu chủ đạo
//   - admin key (chỉ giữ tới khi đóng tab)
//
// 2 loại "sổ" của trình duyệt (chỉ lưu được CHUỖI, dạng key -> value, giống dict của Python):
//   localStorage   : giữ mãi, đóng trình duyệt mở lại vẫn còn
//   sessionStorage : tự xoá khi đóng tab
//
// Dữ liệu chỉ nằm trên máy này, KHÔNG gửi lên backend. Mỗi trình duyệt có sổ riêng.
// Xem tận mắt: F12 -> tab Application -> Local Storage / Session Storage -> http://localhost:3000
// =====================================================================

import { parseSessions, type SavedSession } from "./history.ts";
import { ACCENT_KEY, THEME_KEY, parseAccent, parseThemePref, type Accent, type ThemePref } from "./theme.ts";
import type { Credentials } from "./types.ts";


// Ghi nhớ nhỏ trên trình duyệt. localStorage có thể bị chặn (chế độ ẩn danh, chặn cookie)
// -> mọi lần đọc/ghi đều bọc try/catch, lỗi thì coi như không có gì, trang vẫn chạy.

// Đọc 1 giá trị. Không có key hoặc bị chặn -> null
// Python: def read(store, key) -> str | None
function read(store: "local" | "session", key: string): string | null {
  try {
    // Chọn sổ theo tham số store, rồi lấy giá trị theo key.  Python: store_dict.get(key)
    return (store === "local" ? window.localStorage : window.sessionStorage).getItem(key);
  } catch {
    return null;              // bị chặn -> coi như chưa lưu gì
  }
}

// Ghi 1 giá trị. value = null nghĩa là XOÁ key đó khỏi sổ
// void = hàm không trả về gì (Python: -> None)
function write(store: "local" | "session", key: string, value: string | null): void {
  try {
    const s = store === "local" ? window.localStorage : window.sessionStorage;
    if (value === null) s.removeItem(key);    // Python: store_dict.pop(key, None)
    else s.setItem(key, value);               // Python: store_dict[key] = value
  } catch {
    /* bỏ qua */                               // bị chặn -> không lưu được, trang vẫn chạy tiếp
  }
}

// Bên dưới là các "ngăn kéo" riêng. Mỗi ngăn có get (lấy) và set (cất).
// Component chỉ gọi  sessionStore.get("cv")  mà không cần nhớ tên key hay loại sổ.

// Phiên chat đang mở của từng application: tải lại trang vẫn thấy cuộc chat cũ
// key: "rag.session.cv", "rag.session.paper"...  (mỗi app một phiên riêng)
// set(app, null) -> xoá, dùng khi bấm "Cuộc trò chuyện mới"
export const sessionStore = {
  get: (applicationId: string) => read("local", `rag.session.${applicationId}`),
  set: (applicationId: string, sessionId: string | null) => write("local", `rag.session.${applicationId}`, sessionId),
};

// App chọn lần trước: mở lại trang thì tự chọn đúng app đó
export const lastAppStore = {
  get: () => read("local", "rag.lastApp"),
  set: (applicationId: string) => write("local", "rag.lastApp", applicationId),
};

// Admin key: sessionStorage -> tự mất khi đóng tab, không nằm lại lâu trên máy
// get: chưa có -> "" (chuỗi rỗng, để ô nhập key hiện trống thay vì null)
// set: key || null -> nếu key là "" thì thành null -> XOÁ key khỏi sổ
// Phase 12b: token đăng nhập HOẶC admin key (thay cho adminKeyStore). sessionStorage -> tự mất khi đóng tab.
// Đánh đổi: JavaScript trên trang đọc được sessionStorage, nên trang KHÔNG được chạy script lạ (XSS).
// Token cũng tự hết hạn sau ACCESS_TOKEN_MINUTES, lộ ra thì thiệt hại có giới hạn.
// Sổ chỉ lưu chuỗi -> set: object -> JSON.stringify; get: chuỗi -> parseCredentials (kiểm kỹ)
export const credentialsStore = {
  get: (): Credentials | null => parseCredentials(read("session", "rag.credentials")),
  set: (c: Credentials | null) => write("session", "rag.credentials", c ? JSON.stringify(c) : null),
};

// Dữ liệu trong storage có thể hỏng / do bản cũ ghi -> kiểm kỹ trước khi dùng
export function parseCredentials(raw: string | null): Credentials | null {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw) as Record<string, unknown>;
    if (v.kind === "token" && typeof v.token === "string" && typeof v.email === "string" && v.token) {
      return { kind: "token", token: v.token, email: v.email };
    }
    if (v.kind === "key" && typeof v.key === "string" && v.key) return { kind: "key", key: v.key };
  } catch {
    /* JSON hỏng */
  }
  return null;
}


// ---- Giao diện nâng cấp: lịch sử chat (thanh bên) + sáng/tối + màu chủ đạo ----

// Danh sách phiên ở thanh bên, mỗi app một danh sách. Key: "rag.history.cv"
// Sổ chỉ lưu được chuỗi, nên:
//   set: mảng -> JSON.stringify -> chuỗi rồi mới cất   (Python: json.dumps)
//   get: chuỗi -> parseSessions -> mảng               (Python: json.loads + lọc)
//        parseSessions trả [] nếu chuỗi hỏng hoặc sai dạng -> trang không bị sập
export const historyStore = {
  get: (applicationId: string): SavedSession[] => parseSessions(read("local", `rag.history.${applicationId}`)),
  set: (applicationId: string, list: SavedSession[]) =>
    write("local", `rag.history.${applicationId}`, JSON.stringify(list)),
};

// Chế độ sáng/tối. Key "rag.theme" (THEME_KEY trong theme.ts)
// get: giá trị lạ hoặc chưa có -> "system" (theo máy)
// set: chọn "system" thì XOÁ key, vì không lưu cũng có nghĩa là theo máy
export const themeStore = {
  get: (): ThemePref => parseThemePref(read("local", THEME_KEY)),
  set: (pref: ThemePref) => write("local", THEME_KEY, pref === "system" ? null : pref),
};

// Màu chủ đạo. Key "rag.accent"
// get: giá trị không nằm trong 5 màu (blue, teal, violet, rose, amber) -> "blue"
export const accentStore = {
  get: (): Accent => parseAccent(read("local", ACCENT_KEY)),
  set: (accent: Accent) => write("local", ACCENT_KEY, accent),
};