"use client";

// Phase 12b: nút đăng nhập trên thanh tiêu đề (dùng chung cho trang Chat và Quản trị).
//   - Đăng nhập bằng email + mật khẩu -> backend trả token -> lưu sessionStorage (credentialsStore).
//   - Hoặc dán Admin API key (X-API-Key) như Phase 10, cho bạn khi chưa tạo user.
// Mật khẩu chỉ nằm trong state lúc gõ, gửi đi 1 lần rồi xoá, KHÔNG bao giờ lưu lại.
// Form mở ra dạng bảng nổi (popover) để không chiếm chỗ trên thanh tiêu đề của điện thoại.

import { useState } from "react";
import { api, friendlyError } from "@/lib/api";
import type { Credentials } from "@/lib/types";

type Props = {
  credentials: Credentials | null;
  onChange: (c: Credentials | null) => void;
};

const input =
  "h-10 w-full rounded-lg border border-line bg-surface-2 px-3 text-sm text-ink focus:border-brand focus:outline-none";
const button = "h-10 rounded-lg border border-line px-3 text-sm font-medium text-ink hover:border-brand disabled:opacity-40";

export function LoginBox({ credentials, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"email" | "key">("email");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Đã đăng nhập: hiện ai đang đăng nhập + nút đăng xuất
  if (credentials) {
    return (
      <div className="flex min-w-0 items-center gap-2 text-sm text-muted">
        <span className="hidden max-w-40 truncate sm:inline">
          {credentials.kind === "token" ? credentials.email : "Admin API key"}
        </span>
        <button type="button" className={button} onClick={() => onChange(null)}>Đăng xuất</button>
      </div>
    );
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (mode === "key") {
      if (key) {
        onChange({ kind: "key", key });
        setKey("");
        setOpen(false);
      }
      return;
    }
    setBusy(true);
    try {
      const res = await api.login(email, password);
      onChange({ kind: "token", token: res.access_token, email: email.trim().toLowerCase() });
      setOpen(false);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setPassword(""); // sai hay đúng cũng xoá mật khẩu khỏi bộ nhớ
      setBusy(false);
    }
  }

  return (
    <div className="relative">
      <button type="button" className={button} aria-expanded={open} onClick={() => setOpen(!open)}>
        Đăng nhập
      </button>
      {open && (
        <form
          className="absolute right-0 top-12 z-30 flex w-72 flex-col gap-2 rounded-xl border border-line bg-surface p-3 shadow-lg"
          onSubmit={(e) => void submit(e)}
        >
          {mode === "email" ? (
            <>
              <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                placeholder="Email" autoComplete="username" aria-label="Email" className={input} />
              <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)}
                placeholder="Mật khẩu" autoComplete="current-password" aria-label="Mật khẩu" className={input} />
            </>
          ) : (
            <input type="password" required value={key} onChange={(e) => setKey(e.target.value)}
              placeholder="ADMIN_API_KEY" autoComplete="off" aria-label="X-API-Key" className={input} />
          )}
          <button type="submit" disabled={busy}
            className="h-10 rounded-lg bg-brand px-3 text-sm font-semibold text-brand-fg hover:opacity-90 disabled:opacity-40">
            {mode === "email" ? "Đăng nhập" : "Dùng key"}
          </button>
          <button type="button" className="text-xs text-muted underline"
            onClick={() => { setMode(mode === "email" ? "key" : "email"); setError(null); }}>
            {mode === "email" ? "Dùng Admin API key" : "Dùng email + mật khẩu"}
          </button>
          {error && <span role="alert" className="text-xs text-danger">{error}</span>}
        </form>
      )}
    </div>
  );
}
