"use client";

import { useRef, useState } from "react";
import { ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, checkUpload } from "@/lib/upload";

type Props = {
  disabled: boolean;
  onUpload: (file: File) => Promise<void>;
};

export function UploadBox({ disabled, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // tên file đang nạp
  const [problem, setProblem] = useState<string | null>(null);

  async function handle(files: FileList | null) {
    const file = files?.[0];
    if (!file || disabled || busy) return;
    const invalid = checkUpload(file.name, file.size);
    setProblem(invalid);
    if (invalid) return;
    setBusy(file.name);
    try {
      await onUpload(file);
    } finally {
      setBusy(null);
      if (inputRef.current) inputRef.current.value = ""; // cho phép chọn lại đúng file đó
    }
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault(); // bắt buộc, nếu không trình duyệt sẽ mở file thay vì thả vào đây
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void handle(e.dataTransfer.files);
      }}
      className={`flex flex-col items-center gap-2 rounded-2xl border-2 border-dashed p-6 text-center ${
        dragging ? "border-brand bg-brand-soft" : "border-line bg-surface-2"
      }`}
    >
      {busy ? (
        <p className="font-semibold">Đang nạp “{busy}”… (đọc file, chia chunk, embedding)</p>
      ) : (
        <>
          <p className="font-semibold">Kéo thả file vào đây</p>
          <p className="text-sm text-muted">
            {ALLOWED_EXTENSIONS.map((e) => e.toUpperCase()).join(", ")} · tối đa {MAX_UPLOAD_MB} MB
          </p>
          <button
            type="button"
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
            className="h-10 rounded-lg bg-brand px-4 font-semibold text-brand-fg disabled:opacity-50"
          >
            Chọn file
          </button>
        </>
      )}
      <input
        ref={inputRef}
        type="file"
        hidden
        accept={ALLOWED_EXTENSIONS.map((e) => `.${e}`).join(",")}
        onChange={(e) => void handle(e.target.files)}
      />
      {problem && <p role="alert" className="text-sm text-danger">{problem}</p>}
    </div>
  );
}