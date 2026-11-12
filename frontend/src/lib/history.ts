// Danh sách các cuộc chat của trình duyệt này (thanh bên). Lưu trong localStorage theo từng application.
// Vì sao không gọi GET /chat/sessions? API đó là API QUẢN TRỊ (cần X-API-Key, thấy phiên của mọi người).
// Trang chat công khai chỉ nên thấy phiên của chính mình -> tự nhớ id các phiên đã tạo.


export type SavedSession = {
  id: string;
  title: string;      // câu hỏi đầu tiên của phiên
  updatedAt: string;  // ISO, dùng để sắp xếp và hiện "5 phút trước"
};


export const MAX_SAVED = 30;
export const MAX_TITLE = 80;


export function makeTitle(question: string): string {
  const oneLine = question.replace(/\s+/g, " ").trim();
  return oneLine.length > MAX_TITLE ? `${oneLine.slice(0, MAX_TITLE - 1)}…` : oneLine;
}


// Thêm hoặc cập nhật 1 phiên, đưa lên đầu. Phiên đã có thì GIỮ tiêu đề cũ (= câu hỏi đầu tiên).
export function upsertSession(list: SavedSession[], entry: SavedSession, max = MAX_SAVED): SavedSession[] {
  const old = list.find((s) => s.id === entry.id);
  const updated = old ? { ...old, updatedAt: entry.updatedAt } : entry;
  return [updated, ...list.filter((s) => s.id !== entry.id)].slice(0, max);
}


export function removeSession(list: SavedSession[], id: string): SavedSession[] {
  return list.filter((s) => s.id !== id);
}


// Dữ liệu trong localStorage có thể hỏng hoặc do phiên bản cũ ghi -> chỉ giữ phần tử đúng kiểu
export function parseSessions(raw: string | null): SavedSession[] {
  if (!raw) return [];
  try {
    const data: unknown = JSON.parse(raw);
    if (!Array.isArray(data)) return [];
    return data.filter(
      (s): s is SavedSession =>
        !!s && typeof s.id === "string" && typeof s.title === "string" && typeof s.updatedAt === "string",
    );
  } catch {
    return [];
  }
}


// "vừa xong", "5 phút trước", "3 giờ trước", "hôm qua", "12/10/2026"
export function relativeTime(iso: string, now: Date = new Date()): string {
  const diff = Math.max(0, now.getTime() - new Date(iso).getTime());
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "vừa xong";
  if (minutes < 60) return `${minutes} phút trước`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} giờ trước`;
  if (hours < 48) return "hôm qua";
  return new Date(iso).toLocaleDateString("vi-VN");
}
