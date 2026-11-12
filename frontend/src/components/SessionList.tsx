import { formatDate } from "@/lib/upload";
import type { SessionSummary } from "@/lib/types";

type Props = {
  sessions: SessionSummary[];
  selectedId: string | null;
  onSelect: (sessionId: string) => void;
};

export function SessionList({ sessions, selectedId, onSelect }: Props) {
  if (sessions.length === 0) return <p className="text-sm text-muted">Chưa có phiên chat nào.</p>;
  return (
    <ul className="flex flex-col gap-2">
      {sessions.map((s) => (
        <li key={s.id}>
          <button
            type="button"
            onClick={() => onSelect(s.id)}
            aria-pressed={selectedId === s.id}
            className={`flex w-full flex-col gap-1 rounded-xl border p-3 text-left ${
              selectedId === s.id ? "border-brand bg-brand-soft" : "border-line hover:border-brand"
            }`}
          >
            <span className="text-sm font-semibold">{s.first_question ?? "(trống)"}</span>
            <span className="text-xs text-muted">
              {s.message_count} tin nhắn · {formatDate(s.updated_at)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}