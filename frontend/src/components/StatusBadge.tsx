import type { DocumentStatus } from "@/lib/types";

const STYLES: Record<DocumentStatus, string> = {
  PENDING: "bg-surface-2 text-muted",
  PROCESSING: "bg-brand-soft text-brand",
  READY: "bg-ok-bg text-ok",
  FAILED: "bg-danger-bg text-danger",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${STYLES[status]}`}>{status}</span>;
}