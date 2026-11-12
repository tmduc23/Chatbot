"use client";

// Nút "Giao diện": chọn Sáng / Tối / Theo máy và màu chủ đạo. Lưu vào localStorage.
// Giá trị được ghi lên <html data-theme data-accent>, CSS trong globals.css lo phần đổi màu.

import { useEffect, useRef, useState } from "react";
import { accentStore, themeStore } from "@/lib/storage";
import { ACCENTS, THEME_PREFS, resolveTheme, type Accent, type ThemePref } from "@/lib/theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function apply(pref: ThemePref, accent: Accent) {
  const root = document.documentElement;
  root.dataset.theme = resolveTheme(pref, window.matchMedia(DARK_QUERY).matches);
  root.dataset.accent = accent;
}

export function ThemeMenu() {
  const [open, setOpen] = useState(false);
  const [pref, setPref] = useState<ThemePref>("system");
  const [accent, setAccent] = useState<Accent>("blue");
  const boxRef = useRef<HTMLDivElement>(null);

  // Đọc lựa chọn đã lưu SAU khi trang hiện (localStorage chỉ có trên trình duyệt, không có lúc render ở server)
  useEffect(() => {
    setPref(themeStore.get());
    setAccent(accentStore.get());
  }, []);

  // Đang "Theo máy" mà người dùng đổi chế độ của Windows/điện thoại -> đổi theo ngay
  useEffect(() => {
    if (pref !== "system") return;
    const media = window.matchMedia(DARK_QUERY);
    const onChange = () => apply("system", accent);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [pref, accent]);

  // Bấm ra ngoài hoặc Esc -> đóng menu
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function choosePref(next: ThemePref) {
    setPref(next);
    themeStore.set(next);
    apply(next, accent);
  }

  function chooseAccent(next: Accent) {
    setAccent(next);
    accentStore.set(next);
    apply(pref, next);
  }

  return (
    <div ref={boxRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="Giao diện"
        title="Giao diện"
        className="flex h-10 w-10 items-center justify-center rounded-lg border border-line bg-surface text-ink hover:border-brand"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor" />
        </svg>
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Tuỳ chỉnh giao diện"
          className="absolute right-0 top-12 z-50 w-64 rounded-xl border border-line bg-surface p-4 text-ink shadow-xl"
        >
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Chế độ</p>
          <div className="grid grid-cols-3 gap-1 rounded-lg bg-surface-2 p-1">
            {THEME_PREFS.map((t) => (
              <button
                key={t.value}
                type="button"
                onClick={() => choosePref(t.value)}
                aria-pressed={pref === t.value}
                className={`h-9 rounded-md text-sm font-medium ${
                  pref === t.value ? "bg-surface text-ink shadow" : "text-muted hover:text-ink"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <p className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wide text-muted">Màu chủ đạo</p>
          <div className="flex gap-2">
            {ACCENTS.map((a) => (
              <button
                key={a.value}
                type="button"
                onClick={() => chooseAccent(a.value)}
                aria-pressed={accent === a.value}
                aria-label={a.label}
                title={a.label}
                style={{ background: a.swatch }}
                className={`h-9 w-9 rounded-full ring-offset-2 ring-offset-surface ${
                  accent === a.value ? "ring-2 ring-ink" : "hover:scale-110"
                }`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}