// Kiểu dữ liệu khớp 1-1 với schema Pydantic của backend (app/schemas/*.py, Phase 9).
// Backend đổi schema -> sửa ở đây -> TypeScript báo mọi chỗ dùng sai.

export type Application = {
    id: string;
    name: string;
    description: string | null;
    is_public: boolean;             // Phase 12b: true = khách không đăng nhập cũng chat được
    created_at: string;             // datetime trong JSON là chuỗi ISO,  vd. "2026-10-16T09:30:00Z"
}

export type DocumentStatus = "PENDING" | "PROCESSING" | "READY" | "FAILED";

export type DocumentItem = {
    id: string;
    application_id: string;
    filename: string;
    file_type: string;
    status: DocumentStatus;
    error_message: string | null;
    attempts: number;
    created_at: string;   
}

export type Source = {
  number: number; // khớp với [Source n] trong answer
  document_id: string;
  filename: string;
  page: number | null;
  chunk_id: string;
  text?: string | null; // Phase 10b: đoạn chữ LLM đã đọc. Tin nhắn cũ (trước 10b) không có -> null
};

export type ChatMetadata = {
    retrieval_count: number;
    reranked_count: number;
    relevant_count: number;
    abstain_reason: string | null;
    standalone_question: string | null;
    latency_ms: number;
    cached: boolean;                // Phase 12c: true = lấy từ cache, không gọi AI
};

export type ChatResponse = {
    session_id: string;
    message_id: string;
    answer: string;
    sources: Source[];
    abstained: boolean;
    metadata: ChatMetadata;
};

export type SessionSummary = {
    id: string;
    created_at: string;
    updated_at: string;
    message_count: number;
    first_question: string | null;
};


export type MessageOut = {
    id: string;
    role: "user" | "assistant";
    content: string;
    sources: Source[];
    abstain_reason: string | null;
    created_at: string;
};


export type SessionDetail = {
    id: string;
    application_id: string;
    created_at: string;
    messages: MessageOut[];
};


// Tin nhắn hiển thị trên màn hình chat (gộp cả tin lấy từ lịch sử lẫn tin vừa gửi)
export type ChatMessage = {
    id: string;
    role: "user" | "assistant";
    content: string;
    sources: Source[];
    abstained: boolean;
    standaloneQuestion?: string | null;
    cached?: boolean;               // Phase 12c
};

// ---- Đăng nhập (Phase 12b) ----

// Cách trình duyệt chứng minh "tôi là ai": token sau khi đăng nhập, hoặc Admin API key
export type Credentials =
  | { kind: "token"; token: string; email: string }
  | { kind: "key"; key: string };

export type TokenResponse = {
  access_token: string;
  token_type: "bearer";
  expires_in: number; // giây
};

export type Me = {
  id: string;
  email: string;
  is_superadmin: boolean;
  memberships: { application_id: string; role: "viewer" | "editor" }[];
};

