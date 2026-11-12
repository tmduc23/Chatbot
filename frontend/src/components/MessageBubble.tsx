"use client";

import { useState } from "react";
import { cleanExcerpt, sourceLabel } from "@/lib/citations";
import type { ChatMessage } from "@/lib/types";
import { AnswerText } from "./AnswerText";

export function MessageBubble({ message }: { message: ChatMessage }) {
  // Phase 10b: số nguồn đang mở đoạn trích (null = không mở cái nào). Mỗi lúc chỉ mở 1 nguồn.
  const [openNumber, setOpenNumber] = useState<number | null>(null);

  if (message.role === "user") {
    return (
      <div className="max-w-[85%] self-end whitespace-pre-wrap rounded-2xl rounded-br-sm bg-brand px-4 py-2.5 text-brand-fg shadow-sm sm:max-w-[75%]">
        {message.content}
      </div>
    );
  }

  const numbers = new Set(message.sources.map((s) => s.number));
  const openSource = message.sources.find((s) => s.number === openNumber);
  const excerpt = cleanExcerpt(openSource?.text);

  // Bấm số [n] trong câu trả lời -> mở đoạn trích n (không đóng lại nếu đang mở)
  const showSource = (n: number) => {
    setOpenNumber(n);
    document.getElementById(`src-${message.id}-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  // Bấm thẻ nguồn -> mở; bấm lần nữa -> đóng
  const toggleSource = (n: number) => setOpenNumber((current) => (current === n ? null : n));

  return (
    <div className="flex max-w-full gap-3 self-start sm:max-w-[90%]">
      <div
        aria-hidden="true"
        className="mt-1 hidden h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-soft text-xs font-bold text-brand sm:flex"
      >
        AI
      </div>
      <div
        className={`flex min-w-0 flex-col gap-3 rounded-2xl rounded-tl-sm border px-4 py-3 shadow-sm ${
          message.abstained ? "border-warn-line bg-warn-bg" : "border-line bg-surface"
        }`}
      >
        {message.abstained && (
          <span className="text-xs font-semibold uppercase tracking-wide text-warn">Không tìm thấy trong tài liệu</span>
        )}
        <AnswerText answer={message.content} validNumbers={numbers} onCite={showSource} />

        {message.sources.length > 0 && (
          <div className="flex flex-wrap gap-2" aria-label="Nguồn">
            {message.sources.map((s) => (
              <button
                key={s.number}
                type="button"
                id={`src-${message.id}-${s.number}`}
                onClick={() => toggleSource(s.number)}
                aria-expanded={openNumber === s.number}
                aria-controls={`excerpt-${message.id}`}
                title="Bấm để xem đoạn trích"
                className={`max-w-full truncate rounded-full border px-3 py-1 text-left text-xs transition-colors ${
                  openNumber === s.number
                    ? "border-brand bg-brand-soft text-ink ring-2 ring-brand/40"
                    : "border-line bg-surface-2 text-muted hover:border-brand hover:text-ink"
                }`}
              >
                📄 [{s.number}] {sourceLabel(s.filename, s.page)} {openNumber === s.number ? "▴" : "▾"}
              </button>
            ))}
          </div>
        )}

        {/* Phase 10b: đoạn trích của nguồn đang mở = đúng đoạn chữ LLM đã đọc để trả lời */}
        {openSource && (
          <figure
            id={`excerpt-${message.id}`}
            className="m-0 flex flex-col gap-1.5 rounded-xl border border-line bg-surface-2 px-3 py-2.5"
          >
            <figcaption className="flex items-center gap-2 text-xs font-semibold text-muted">
              <span className="mr-auto truncate">
                Đoạn trích [{openSource.number}] · {sourceLabel(openSource.filename, openSource.page)}
              </span>
              <button
                type="button"
                onClick={() => setOpenNumber(null)}
                aria-label="Đóng đoạn trích"
                className="rounded px-1.5 text-muted hover:bg-surface hover:text-ink"
              >
                ✕
              </button>
            </figcaption>
            {excerpt ? (
              <blockquote className="m-0 max-h-60 overflow-y-auto whitespace-pre-wrap break-words border-l-2 border-brand pl-3 text-sm leading-relaxed text-ink">
                {excerpt}
              </blockquote>
            ) : (
              <p className="text-sm text-muted">Tin nhắn này được lưu trước khi có tính năng đoạn trích.</p>
            )}
          </figure>
        )}

        {message.cached && (
          <span className="text-xs text-muted">⚡ trả lời từ cache (không gọi AI)</span>
        )}

        {message.standaloneQuestion && (
          <span className="font-mono text-xs text-muted">câu đã viết lại: “{message.standaloneQuestion}”</span>
        )}
      </div>
    </div>
  );
}