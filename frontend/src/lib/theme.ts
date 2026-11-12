export type ThemePref = "light" | "dark" | "system";
export type Accent = "blue" | "teal" | "violet" | "rose" | "amber";

export const THEME_PREFS: { value: ThemePref; label: string }[] = [
  { value: "light", label: "Sáng" },
  { value: "dark", label: "Tối" },
  { value: "system", label: "Theo máy" },
];

// swatch: màu hiển thị trên nút chọn (giống --brand ở chế độ sáng trong globals.css)
export const ACCENTS: { value: Accent; label: string; swatch: string }[] = [
  { value: "blue", label: "Xanh dương", swatch: "#2563eb" },
  { value: "teal", label: "Xanh ngọc", swatch: "#0f766e" },
  { value: "violet", label: "Tím", swatch: "#6d28d9" },
  { value: "rose", label: "Hồng", swatch: "#be123c" },
  { value: "amber", label: "Cam", swatch: "#b45309" },
];

export const THEME_KEY = "rag.theme";
export const ACCENT_KEY = "rag.accent";

export function parseThemePref(value: string | null): ThemePref {
  return value === "light" || value === "dark" ? value : "system"; // giá trị lạ / chưa chọn -> theo máy
}

export function parseAccent(value: string | null): Accent {
  return ACCENTS.some((a) => a.value === value) ? (value as Accent) : "blue";
}

// "system" -> hỏi hệ điều hành đang sáng hay tối
export function resolveTheme(pref: ThemePref, systemDark: boolean): "light" | "dark" {
  return pref === "system" ? (systemDark ? "dark" : "light") : pref;
}

// Script chạy TRƯỚC khi trang hiện ra (đặt trong <head>), để không bị "nháy trắng" khi đang dùng chế độ tối.
// Phải là chuỗi JS thuần, tự chứa: nó chạy trước khi React và các module được tải.
export const THEME_INIT_SCRIPT = `(function(){try{
var t=localStorage.getItem("${THEME_KEY}"),a=localStorage.getItem("${ACCENT_KEY}");
var d=t==="dark"||(t!=="light"&&matchMedia("(prefers-color-scheme: dark)").matches);
var r=document.documentElement;r.dataset.theme=d?"dark":"light";
if(a)r.dataset.accent=a;}catch(e){}})();`;