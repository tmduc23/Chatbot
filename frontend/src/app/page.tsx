"use client";
// Trang chat công khai: thanh bên (application + lịch sử chat) | khung chat.
// "use client": trang chạy trên trình duyệt (cần useState, localStorage, gọi API khi người dùng bấm).

// Luồng của trang:
// 1. Mở trang: tải danh sách application, chọn lại application lần trước.
// 2. Đổi application: nạp lịch sử (thanh bên) của application đó, mở lại phiên đang dở bằng `GET /chat/sessions/{id}`. Backend trả 404 (phiên đã bị xoá) thì bỏ phiên đó khỏi lịch sử.
// 3. Gửi câu hỏi: hiện câu hỏi ngay, chờ `POST /chat`, lưu `session_id`, đưa phiên lên đầu thanh bên. Lỗi thì gỡ câu hỏi khỏi màn hình và giữ nó trong ô nhập để bấm gửi lại.
// 4. Bấm một phiên ở thanh bên hoặc "+ Cuộc chat mới": gọi `openSession(app, id)` (id = null là chat mới).

// `loadToken`: mỗi lần mở phiên thì tăng 1. Bấm 2 phiên (hoặc đổi application 2 lần) thật nhanh thì kết quả API của lần **cũ** về sau sẽ bị bỏ qua, không ghi đè lên phiên đang xem.

import { useEffect, useRef, useState } from "react";
import { ChatInput } from "@/components/ChatInput";
import { ChatSidebar } from "@/components/ChatSidebar";
import { LoginBox } from "@/components/LoginBox";
import { MessageBubble } from "@/components/MessageBubble";
import { ThemeMenu } from "@/components/ThemeMenu";
import { ApiError, api, friendlyError } from "@/lib/api";
import { makeTitle, removeSession, upsertSession, type SavedSession } from "@/lib/history";
import { credentialsStore, historyStore, lastAppStore, sessionStore } from "@/lib/storage";
import type { Application, ChatMessage, Credentials, MessageOut } from "@/lib/types";

// Câu gợi ý theo application; application khác dùng bộ chung
const SUGGESTIONS: Record<string, string[]> = {
  cv: ["Bạn ấy đã làm những dự án nào?", "Bạn ấy có kinh nghiệm gì với Docker?", "Bạn ấy học ở trường nào?"],
  default: ["Tài liệu này nói về gì?", "Tóm tắt các ý chính", "Có những số liệu quan trọng nào?"],
};

const toChatMessage = (m: MessageOut): ChatMessage => ({
  id: m.id,
  role: m.role,
  content: m.content,
  sources: m.sources,
  abstained: m.abstain_reason !== null,
});

export default function ChatPage() {
  const [applications, setApplications] = useState<Application[]>([]);
  const [appId, setAppId] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<SavedSession[]>([]);
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false); // chỉ dùng trên điện thoại
  // Phase 12b: đăng nhập để thấy application riêng tư. undefined = chưa đọc sessionStorage xong
  const [credentials, setCredentials] = useState<Credentials | null | undefined>(undefined);
  const auth = credentials ?? undefined;
  const bottomRef = useRef<HTMLDivElement>(null);
  const loadToken = useRef(0); // mỗi lần mở phiên +1; kết quả API của lần mở CŨ thì bỏ qua

  function saveHistory(list: SavedSession[]) {
    setHistory(list);
    historyStore.set(appId, list);
  }

  // Mở 1 phiên chat (null = cuộc chat mới, trống)
  function openSession(targetApp: string, id: string | null) {
    const token = ++loadToken.current;
    sessionStore.set(targetApp, id);
    setSessionId(id);
    setMessages([]);
    setError(null);
    if (!id) {
      setLoadingHistory(false);
      return;
    }
    setLoadingHistory(true);
    api.getSession(targetApp, id, auth)
      .then((detail) => {
        if (token === loadToken.current) setMessages(detail.messages.map(toChatMessage));
      })
      .catch((err) => {
        if (token !== loadToken.current) return;
        if (err instanceof ApiError && err.status === 404) {
          // phiên không còn ở backend -> bỏ khỏi lịch sử, bắt đầu chat mới
          const list = removeSession(historyStore.get(targetApp), id);
          setHistory(list);
          historyStore.set(targetApp, list);
          sessionStore.set(targetApp, null);
          setSessionId(null);
        } else {
          setError(friendlyError(err));
        }
      })
      .finally(() => token === loadToken.current && setLoadingHistory(false));
  }

  // 0. Phase 12b: đọc thông tin đăng nhập đã lưu (sessionStorage chỉ đọc được trên trình duyệt)
  useEffect(() => {
    setCredentials(credentialsStore.get());
  }, []);

  function changeCredentials(c: Credentials | null) {
    credentialsStore.set(c);
    setCredentials(c);
  }

  // 1. Mở trang (và mỗi lần đăng nhập/đăng xuất): tải danh sách application, chọn lại application lần trước.
  //    Khách chỉ thấy app công khai; đăng nhập rồi thấy thêm app mình là thành viên.
  useEffect(() => {
    if (credentials === undefined) return;
    api.listApplications(credentials ?? undefined)
      .then((apps) => {
        setApplications(apps);
        const last = lastAppStore.get();
        setAppId(apps.find((a) => a.id === last)?.id ?? apps[0]?.id ?? "");
      })
      .catch((err) => setError(friendlyError(err)));
  }, [credentials]);

  // 2. Đổi application: nạp lịch sử của application đó và mở lại phiên đang dở (nếu có)
  useEffect(() => {
    if (!appId) return;
    lastAppStore.set(appId);
    setHistory(historyStore.get(appId));
    openSession(appId, sessionStore.get(appId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appId]);

  // 3. Có tin mới -> cuộn xuống cuối
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  async function send(text: string): Promise<boolean> {
    if (!appId || sending) return false;
    setError(null);
    setSending(true);
    const userMessage: ChatMessage = { id: `local-${Date.now()}`, role: "user", content: text, sources: [], abstained: false };
    setMessages((prev) => [...prev, userMessage]); // hiện câu hỏi ngay, không chờ backend

    try {
      const res = await api.chat(appId, text, sessionId, auth);
      setSessionId(res.session_id);
      sessionStore.set(appId, res.session_id);
      saveHistory(upsertSession(history, {
        id: res.session_id,
        title: makeTitle(text), // phiên đã có thì upsertSession giữ tiêu đề cũ
        updatedAt: new Date().toISOString(),
      }));
      setMessages((prev) => [...prev, {
        id: res.message_id,
        role: "assistant",
        content: res.answer,
        sources: res.sources,
        abstained: res.abstained,
        standaloneQuestion: res.metadata.standalone_question,
        cached: res.metadata.cached,
      }]);
      return true;
    } catch (err) {
      console.error(err);
      setMessages((prev) => prev.filter((m) => m.id !== userMessage.id)); // bỏ câu hỏi chưa được trả lời
      setError(friendlyError(err));
      return false; // ChatInput giữ lại câu hỏi để bấm gửi lại
    } finally {
      setSending(false);
    }
  }

  function selectSession(id: string) {
    setSidebarOpen(false);
    if (id !== sessionId) openSession(appId, id);
  }

  function hideSession(id: string) {
    saveHistory(removeSession(history, id));
    if (id === sessionId) openSession(appId, null);
  }

  function newChat() {
    setSidebarOpen(false);
    openSession(appId, null);
  }

  const app = applications.find((a) => a.id === appId);
  const suggestions = SUGGESTIONS[appId] ?? SUGGESTIONS.default;

  return (
    <div className="flex h-dvh overflow-hidden">
      <ChatSidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        applications={applications}
        appId={appId}
        onAppChange={(id) => {
          setAppId(id);
          setSidebarOpen(false);
        }}
        sessions={history}
        activeId={sessionId}
        onSelect={selectSession}
        onRemove={hideSession}
        onNewChat={newChat}
        busy={sending}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-line bg-surface px-3 py-2.5 sm:px-5">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Mở danh sách cuộc chat"
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-line text-ink lg:hidden"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
            </svg>
          </button>
          <div className="mr-auto min-w-0">
            <h1 className="truncate font-bold">{app?.name ?? "Chat với tài liệu"}</h1>
            <p className="truncate text-xs text-muted">
              {sessionId ? "Đang tiếp tục cuộc chat" : "Cuộc chat mới"}
              {app ? ` · ${app.id}` : ""}
            </p>
          </div>
          {credentials !== undefined && <LoginBox credentials={credentials} onChange={changeCredentials} />}
          <ThemeMenu />
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-3 py-5 sm:px-4">
            {applications.length === 0 && !error && <p className="text-muted">Đang tải…</p>}

            {app && messages.length === 0 && !loadingHistory && (
              <div className="mt-6 flex flex-col items-center gap-4 text-center sm:mt-12">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-soft text-2xl">💬</div>
                <h2 className="text-xl font-bold sm:text-2xl">{app.name}</h2>
                {app.description && <p className="max-w-lg text-muted">{app.description}</p>}
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestions.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => void send(s)}
                      disabled={sending}
                      className="rounded-full border border-line bg-surface px-4 py-2 text-sm text-ink shadow-sm hover:border-brand hover:text-brand"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {loadingHistory && <p className="text-muted">Đang mở lại cuộc chat…</p>}

            {messages.map((m) => <MessageBubble key={m.id} message={m} />)}

            {sending && (
              <div className="flex items-center gap-2 self-start rounded-2xl border border-dashed border-line bg-surface px-4 py-2.5 text-muted">
                <span className="flex gap-1" aria-hidden="true">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand [animation-delay:-0.2s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand [animation-delay:-0.1s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand" />
                </span>
                Đang tìm trong tài liệu và soạn câu trả lời…
              </div>
            )}
            {error && (
              <div role="alert" className="rounded-xl border border-danger-line bg-danger-bg px-4 py-3 text-sm text-danger">
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </main>

        <ChatInput
          disabled={!appId || sending}
          placeholder={app ? `Hỏi về “${app.name}”…` : "Đặt câu hỏi…"}
          onSend={send}
        />
      </div>
    </div>
  );
}