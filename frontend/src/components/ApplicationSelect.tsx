import type { Application } from "@/lib/types";

type Props = {
  applications: Application[];
  value: string;
  onChange: (applicationId: string) => void;
};

export function ApplicationSelect({ applications, value, onChange }: Props) {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-xs font-medium text-muted">
      Application
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={applications.length === 0}
        className="h-10 w-full min-w-0 rounded-lg border border-line bg-surface px-3 text-sm text-ink focus:border-brand focus:outline-none"
      >
        {applications.length === 0 && <option value="">(chưa có)</option>}
        {applications.map((a) => (
          <option key={a.id} value={a.id}>
            {a.name} ({a.id})
          </option>
        ))}
      </select>
    </label>
  );
}