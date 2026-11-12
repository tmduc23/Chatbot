import { formatDate } from "@/lib/upload";
import type { DocumentItem } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

type Props = {
  documents: DocumentItem[];
  onDelete: (doc: DocumentItem) => void;
};

export function DocumentTable({ documents, onDelete }: Props) {
  if (documents.length === 0) {
    return <p className="rounded-xl border border-line p-4 text-sm text-muted">Chưa có tài liệu nào.</p>;
  }
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-140 text-left text-sm">
        <thead className="bg-surface-2 text-muted">
          <tr>
            <th className="px-4 py-3 font-semibold">Tên file</th>
            <th className="px-4 py-3 font-semibold">Trạng thái</th>
            <th className="px-4 py-3 font-semibold">Ngày nạp</th>
            <th className="px-4 py-3"><span className="sr-only">Thao tác</span></th>
          </tr>
        </thead>
        <tbody>
          {documents.map((d) => (
            <tr key={d.id} className="border-t border-line align-top">
              <td className="px-4 py-3 font-medium">
                {d.filename}
                {d.error_message && <div className="mt-1 text-xs font-normal text-danger">{d.error_message}</div>}
              </td>
              <td className="px-4 py-3"><StatusBadge status={d.status} /></td>
              <td className="px-4 py-3 text-muted">{formatDate(d.created_at)}</td>
              <td className="px-4 py-3 text-right">
                <button
                  type="button"
                  onClick={() => onDelete(d)}
                  aria-label={`Xoá ${d.filename}`}
                  className="h-9 rounded-lg border border-danger-line px-3 text-danger hover:bg-danger-bg"
                >
                  Xoá
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}