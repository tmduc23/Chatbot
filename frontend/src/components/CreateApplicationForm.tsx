"use client";

import { useState } from "react";

type Props = {
  onCreate: (body: { id: string; name: string; description: string | null }) => Promise<boolean>;
};

// Cùng quy tắc với APPLICATION_ID_PATTERN ở backend (schemas/application.py)
const ID_PATTERN = "[a-z0-9][a-z0-9_\\-]{1,49}";

export function CreateApplicationForm({ onCreate }: Props) {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);

  return (
    <details className="rounded-xl border border-line p-4">
      <summary className="cursor-pointer font-semibold">+ Tạo application mới</summary>
      <form
        className="mt-4 flex flex-col gap-3"
        onSubmit={async (e) => {
          e.preventDefault();
          setSaving(true);
          const ok = await onCreate({ id, name, description: description.trim() || null });
          setSaving(false);
          if (ok) {
            setId("");
            setName("");
            setDescription("");
          }
        }}
      >
        <label className="flex flex-col gap-1 text-sm">
          ID (chữ thường, số, - hoặc _; vd. projects)
          <input required pattern={ID_PATTERN} value={id} onChange={(e) => setId(e.target.value)}
            className="h-10 rounded-lg border border-line bg-surface-2 text-ink focus:border-brand focus:outline-none px-3 font-mono" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Tên hiển thị
          <input required maxLength={200} value={name} onChange={(e) => setName(e.target.value)}
            className="h-10 rounded-lg border border-line bg-surface-2 text-ink focus:border-brand focus:outline-none px-3" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Mô tả (reranker và prompt dùng mô tả này để hiểu tài liệu nói về gì)
          <textarea rows={2} maxLength={2000} value={description} onChange={(e) => setDescription(e.target.value)}
            className="rounded-lg border border-line bg-surface-2 text-ink focus:border-brand focus:outline-none px-3 py-2" />
        </label>
        <button type="submit" disabled={saving}
          className="h-10 self-start rounded-lg bg-brand px-4 font-semibold text-brand-fg disabled:opacity-50">
          {saving ? "Đang tạo…" : "Tạo"}
        </button>
      </form>
    </details>
  );
}