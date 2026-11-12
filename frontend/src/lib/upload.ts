// Kiểm tra file NGAY trên trình duyệt trước khi upload -> báo lỗi tức thì, không tốn băng thông.
// Backend vẫn kiểm tra lại (không bao giờ tin client), đây chỉ để trải nghiệm tốt hơn.

import type { DocumentItem } from "./types";

export const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "md"]; // = SUPPORTED_FILE_TYPES ở backend
export  const MAX_UPLOAD_MB = 10;


export function  checkUpload(name: string, sizeBytes: number): string | null {
  const ext = name.includes(".") ? name.split(".").pop()!.toLowerCase() : "";
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `"${name}": chỉ nhận ${ALLOWED_EXTENSIONS.map((e) => e.toUpperCase()).join(", ")}`;
  }
  if (sizeBytes === 0) return `"${name}" là file rỗng`;
  if (sizeBytes > MAX_UPLOAD_MB * 1024 * 1024) return `"${name}" lớn hơn ${MAX_UPLOAD_MB} MB`;
  return null; // hợp lệ
}


export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

// Phase 12a: BACKGROUND_INGESTION=true -> upload trả 202 + PENDING, worker xử lý sau vài giây
export function isProcessing(doc: Pick<DocumentItem, "status">): boolean {
  return doc.status === "PENDING" || doc.status === "PROCESSING";
}

export function uploadNotice(doc: Pick<DocumentItem, "status" | "filename" | "error_message">):
  { kind: "ok" | "error"; text: string } {
  if (doc.status === "READY") return { kind: "ok", text: `Đã nạp “${doc.filename}”.` };
  if (isProcessing(doc)) return { kind: "ok", text: `Đã nhận “${doc.filename}”, đang xử lý nền…` };
  return { kind: "error", text: `“${doc.filename}”: ${doc.status} ${doc.error_message ?? ""}`.trim() };
}