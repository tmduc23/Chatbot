// Header của trang quản trị: nền sáng/tối theo theme, có `ThemeMenu`.

import Link from "next/link";
import { ThemeMenu } from "./ThemeMenu";

type Props = {
  title: string;
  active: "chat" | "admin";
  children?: React.ReactNode; // phần bên phải: ô API key...
};

// Header của trang quản trị (trang chat có thanh bên riêng)
export function AppHeader({ title, active, children }: Props) {
  const link = (href: string, label: string, isActive: boolean) => (
    <Link
      href={href}
      aria-current={isActive ? "page" : undefined}
      className={`rounded-lg px-3 py-2 text-sm font-medium ${
        isActive ? "bg-brand-soft text-ink" : "text-muted hover:text-ink"
      }`}
    >
      {label}
    </Link>
  );

  return (
    <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-line bg-surface/90 px-4 py-3 backdrop-blur sm:px-6">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand text-xs font-bold text-brand-fg">RAG</div>
      <h1 className="mr-auto text-lg font-bold">{title}</h1>
      {children}
      <nav className="flex gap-1">
        {link("/", "Chat", active === "chat")}
        {link("/admin", "Quản trị", active === "admin")}
      </nav>
      <ThemeMenu />
    </header>
  );
}