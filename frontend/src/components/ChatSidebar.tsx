// - Điện thoại: `fixed` + `-translate-x-full`, nằm ngoài màn hình bên trái. `open=true` thì `translate-x-0` trượt vào, kèm lớp nền mờ.
// - Laptop: từ 1024px trở lên, các class `lg:static lg:translate-x-0` làm nó thành một cột bình thường, luôn hiện.
// - Nút ✕ trên từng cuộc chat chỉ **ẩn khỏi danh sách** trên trình duyệt này. Dữ liệu trong DB vẫn còn, admin vẫn xem được.

"use client";

// Thanh bên của trang chat: chọn application, tạo chat mới, danh sách các cuộc chat trước.
// Laptop (>= 1024px, "lg:"): luôn hiện bên trái. Điện thoại: ẩn, mở bằng nút ☰ (trượt từ trái ra).

import Link from "next/link";
import type { SavedSession } from "@/lib/history";
import { relativeTime } from "@/lib/history";
import type { Application } from "@/lib/types";
import { ApplicationSelect } from "./ApplicationSelect";

type Props = {
  open: boolean;                 // chỉ có tác dụng trên điện thoại
  onClose: () => void;
  applications: Application[];
  appId: string;
  onAppChange: (applicationId: string) => void;
  sessions: SavedSession[];
  activeId: string | null;
  onSelect: (sessionId: string) => void;
  onRemove: (sessionId: string) => void;
  onNewChat: () => void;
  busy: boolean;                 // đang gửi câu hỏi -> khoá đổi phiên
};

export function ChatSidebar(props: Props) {
  const { open, onClose, applications, appId, onAppChange, sessions, activeId, onSelect, onRemove, onNewChat, busy } = props;

  return (
    <>
      {/* Lớp nền mờ phía sau thanh bên (chỉ điện thoại). Bấm vào -> đóng */}
      <div
        aria-hidden="true"
        onClick={onClose}
        className={`fixed inset-0 z-30 bg-black/40 transition-opacity lg:hidden ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      />

      <aside
        aria-label="Các cuộc chat"
        className={`fixed inset-y-0 left-0 z-40 flex w-[85vw] max-w-80 flex-col border-r border-line bg-surface transition-transform duration-200 lg:static lg:z-auto lg:w-72 lg:max-w-none lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 px-4 pb-3 pt-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand text-sm font-bold text-brand-fg">RAG</div>
          <div className="mr-auto leading-tight">
            <p className="font-bold">Chat với tài liệu</p>
            <p className="text-xs text-muted">Trả lời có trích nguồn</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Đóng danh sách"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:bg-surface-2 lg:hidden"
          >
            ✕
          </button>
        </div>

        <div className="flex flex-col gap-3 px-4">
          <ApplicationSelect applications={applications} value={appId} onChange={onAppChange} />
          <button
            type="button"
            onClick={onNewChat}
            disabled={busy || !appId}
            className="h-10 rounded-lg bg-brand px-3 text-sm font-semibold text-brand-fg hover:opacity-90 disabled:opacity-50"
          >
            + Cuộc chat mới
          </button>
        </div>

        <p className="px-4 pb-2 pt-5 text-xs font-semibold uppercase tracking-wide text-muted">Lịch sử</p>
        <nav className="flex-1 overflow-y-auto px-2 pb-3">
          {sessions.length === 0 ? (
            <p className="px-2 text-sm text-muted">Chưa có cuộc chat nào với application này.</p>
          ) : (
            <ul className="flex flex-col gap-0.5">
              {sessions.map((s) => {
                const active = s.id === activeId;
                return (
                  <li key={s.id} className="group relative">
                    <button
                      type="button"
                      onClick={() => onSelect(s.id)}
                      disabled={busy}
                      aria-current={active ? "true" : undefined}
                      className={`flex w-full flex-col rounded-lg px-3 py-2 pr-9 text-left disabled:opacity-60 ${
                        active ? "bg-brand-soft" : "hover:bg-surface-2"
                      }`}
                    >
                      <span className={`truncate text-sm ${active ? "font-semibold text-ink" : "text-ink"}`}>{s.title}</span>
                      <span className="text-xs text-muted">{relativeTime(s.updatedAt)}</span>
                    </button>
                    {/* Chỉ bỏ khỏi danh sách trên trình duyệt này; dữ liệu ở backend vẫn còn (admin vẫn xem được) */}
                    <button
                      type="button"
                      onClick={() => onRemove(s.id)}
                      aria-label={`Ẩn cuộc chat “${s.title}”`}
                      title="Ẩn khỏi danh sách"
                      className="absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-muted opacity-100 hover:bg-surface hover:text-danger lg:opacity-0 lg:group-hover:opacity-100 lg:focus:opacity-100"
                    >
                      ✕
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </nav>

        <div className="border-t border-line p-3">
          <Link href="/admin" className="block rounded-lg px-3 py-2 text-sm text-muted hover:bg-surface-2 hover:text-ink">
            ⚙ Quản trị tài liệu
          </Link>
        </div>
      </aside>
    </>
  );
}