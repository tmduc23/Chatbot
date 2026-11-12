"use client";

// Trang quản trị: đăng nhập (hoặc dán Admin API key) -> tạo application, upload/xoá tài liệu, xem các phiên chat.
// Phase 12b: backend kiểm quyền theo application (editor / superadmin / X-API-Key).
// Thông tin đăng nhập chỉ nằm trong sessionStorage.

import { useCallback, useEffect, useState } from "react";
import { AppHeader } from "@/components/AppHeader";
import { ApplicationSelect } from "@/components/ApplicationSelect";
import { CreateApplicationForm } from "@/components/CreateApplicationForm";
import { DocumentTable } from "@/components/DocumentTable";
import { LoginBox } from "@/components/LoginBox";
import { MessageBubble } from "@/components/MessageBubble";
import { SessionList } from "@/components/SessionList";
import { UploadBox } from "@/components/UploadBox";
import { api, friendlyError } from "@/lib/api";
import { credentialsStore } from "@/lib/storage";
import { isProcessing, uploadNotice } from "@/lib/upload";
import type { Application, Credentials, DocumentItem, SessionDetail, SessionSummary } from "@/lib/types";

export default function AdminPage() {
  const [credentials, setCredentials] = useState<Credentials | null>(null);   // Phase 12b: token hoặc API key
  const [applications, setApplications] = useState<Application[]>([]);
  const [appId, setAppId] = useState("");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [openSession, setOpenSession] = useState<SessionDetail | null>(null);
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const showError = (err: unknown) => {
    console.error(err);
    setNotice({ kind: "error", text: friendlyError(err) });
  };

  const loadApplications = useCallback(async (selectId?: string) => {
    try {
      const apps = await api.listApplications(credentials ?? undefined);
      setApplications(apps);
      setAppId((current) => selectId ?? (apps.some((a) => a.id === current) ? current : apps[0]?.id ?? ""));
    } catch (err) {
      showError(err);
    }
  }, [credentials]);

  // Tải lại tài liệu + phiên chat của application đang chọn
  const refresh = useCallback(async () => {
    if (!appId || !credentials) return;   // chưa đăng nhập: không gọi API quản trị
    try {
      const [docs, sess] = await Promise.all([
        api.listDocuments(credentials, appId),
        api.listSessions(credentials, appId),
      ]);
      setDocuments(docs);
      setSessions(sess);
    } catch (err) {
      setDocuments([]);
      setSessions([]);
      showError(err);
    }
  }, [credentials, appId]);

  useEffect(() => {
    setCredentials(credentialsStore.get());
  }, []);

  useEffect(() => {
    void loadApplications();
  }, [loadApplications]);

  function changeCredentials(c: Credentials | null) {
    credentialsStore.set(c);
    setCredentials(c);
    if (!c) {
      setDocuments([]);
      setSessions([]);
    }
  }

  useEffect(() => {
    setOpenSession(null);
    setNotice(null);
    void refresh();
  }, [refresh]);

  // Phase 12a: còn document PENDING/PROCESSING -> 3 giây hỏi lại 1 lần, tới khi worker xử lý xong
  const waiting = documents.some(isProcessing);
  useEffect(() => {
    if (!waiting) return;
    const timer = setTimeout(() => void refresh(), 3000);
    return () => clearTimeout(timer);
  }, [waiting, documents, refresh]);

  async function upload(file: File) {
    try {
      if (!credentials) return;
      const doc = await api.uploadDocument(credentials, appId, file);
      setNotice(uploadNotice(doc));
    } catch (err) {
      showError(err);
    }
    await refresh(); // lỗi cũng tải lại: backend có thể đã lưu bản ghi FAILED
  }

  async function remove(doc: DocumentItem) {
    if (!window.confirm(`Xoá “${doc.filename}” và toàn bộ chunk của nó?`)) return;
    try {
      if (!credentials) return;
      await api.deleteDocument(credentials, appId, doc.id);
      setNotice({ kind: "ok", text: `Đã xoá “${doc.filename}”.` });
    } catch (err) {
      showError(err);
    }
    await refresh();
  }

  async function create(body: { id: string; name: string; description: string | null }) {
    try {
      if (!credentials) return false;
      const created = await api.createApplication(credentials, body);
      setNotice({ kind: "ok", text: `Đã tạo application “${created.id}”.` });
      await loadApplications(created.id);
      return true;
    } catch (err) {
      showError(err);
      return false;
    }
  }

  async function selectSession(sessionId: string) {
    try {
      setOpenSession(await api.getSession(appId, sessionId, credentials ?? undefined));
    } catch (err) {
      showError(err);
    }
  }

  return (
    <div className="min-h-screen">
      <AppHeader title="Quản trị tài liệu" active="admin">
        <LoginBox credentials={credentials} onChange={changeCredentials} />
      </AppHeader>

      <div className="mx-auto flex max-w-6xl flex-wrap items-start gap-6 p-4 sm:p-6">
        <main className="flex min-w-0 flex-[999_1_600px] flex-col gap-5 rounded-2xl border border-line bg-surface p-4 shadow-sm sm:p-5">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="mr-auto text-xl font-bold">Tài liệu</h2>
            <div className="w-full sm:w-64"><ApplicationSelect applications={applications} value={appId} onChange={setAppId} /></div>
            <button type="button" onClick={() => void refresh()}
              className="h-10 self-end rounded-lg border border-line px-3 text-sm hover:border-brand">
              Tải lại
            </button>
          </div>

          {!credentials && (
            <p className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-muted">
              Đăng nhập bằng tài khoản editor/superadmin (hoặc dán Admin API key) để quản lý tài liệu.
            </p>
          )}

          {notice && (
            <div role={notice.kind === "error" ? "alert" : "status"}
              className={`rounded-xl border px-4 py-3 text-sm ${
                notice.kind === "ok" ? "border-ok-line bg-ok-bg text-ok" : "border-danger-line bg-danger-bg text-danger"
              }`}>
              {notice.text}
            </div>
          )}

          <UploadBox disabled={!appId || !credentials} onUpload={upload} />
          <DocumentTable documents={documents} onDelete={(d) => void remove(d)} />
          <CreateApplicationForm onCreate={create} />
        </main>

        <aside className="flex flex-[1_1_300px] flex-col gap-3 rounded-2xl border border-line bg-surface p-4 shadow-sm sm:p-5">
          <h2 className="text-lg font-bold">Phiên chat gần đây</h2>
          <SessionList sessions={sessions} selectedId={openSession?.id ?? null} onSelect={(id) => void selectSession(id)} />
          {openSession && (
            <div className="mt-2 flex flex-col gap-3 border-t border-line pt-4">
              {openSession.messages.map((m) => (
                <MessageBubble key={m.id} message={{
                  id: m.id, role: m.role, content: m.content, sources: m.sources, abstained: m.abstain_reason !== null,
                }} />
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
