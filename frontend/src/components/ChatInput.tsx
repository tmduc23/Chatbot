// Ô nhập dạng thẻ bo tròn. Placeholder theo tên application (prop `placeholder` mới).

"use client";

import { useState } from "react";

const MAX_CHARS = 1000; // = MAX_QUESTION_CHARS ở backend (rag_service.py)

type Props = {
  disabled: boolean;
  placeholder: string;
  onSend: (text: string) => Promise<boolean>; // true = gửi được -> xoá ô nhập
};

export function ChatInput({ disabled, placeholder, onSend }: Props) {
  const [text, setText] = useState("");
  const trimmed = text.trim();
  const canSend = !disabled && trimmed.length > 0 && text.length <= MAX_CHARS;

  async function submit() {
    if (!canSend) return;
    if (await onSend(trimmed)) setText(""); // lỗi -> giữ nguyên câu hỏi để gửi lại
  }

  return (
    <form
      className="mx-auto w-full max-w-3xl px-3 pb-3 sm:px-4 sm:pb-4"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 shadow-sm focus-within:border-brand">
        <label htmlFor="question" className="sr-only">Câu hỏi</label>
        <textarea
          id="question"
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter = gửi, Shift+Enter = xuống dòng. isComposing: đang gõ dấu tiếng Việt (IME) thì không gửi
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void submit();
            }
          }}
          placeholder={placeholder}
          className="min-h-12 flex-1 resize-none bg-transparent px-2 py-1.5 text-ink placeholder:text-muted focus:outline-none"
        />
        <div className="flex flex-col items-end gap-1">
          <span className={`text-xs ${text.length > MAX_CHARS ? "text-danger" : "text-muted"}`}>
            {text.length}/{MAX_CHARS}
          </span>
          <button
            type="submit"
            disabled={!canSend}
            className="h-10 rounded-xl bg-brand px-5 font-semibold text-brand-fg hover:opacity-90 disabled:opacity-40"
          >
            Gửi
          </button>
        </div>
      </div>
      <p className="mt-1.5 hidden text-center text-xs text-muted sm:block">Enter để gửi · Shift+Enter xuống dòng</p>
    </form>
  );
}