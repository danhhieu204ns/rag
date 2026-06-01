import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import api from "../api";

// ── Icons ──────────────────────────────────────────────────────────────────

const IconMenu = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="3" y1="6" x2="21" y2="6" />
    <line x1="3" y1="12" x2="21" y2="12" />
    <line x1="3" y1="18" x2="21" y2="18" />
  </svg>
);

const IconEdit = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);

const IconSearch = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const IconPlus = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const IconTrash = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6l-1 14H6L5 6" />
    <path d="M10 11v6M14 11v6" />
    <path d="M9 6V4h6v2" />
  </svg>
);

const IconSend = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);

const IconChevronDown = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const IconDocument = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const IconClose = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const IconThink = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4l3 3" />
  </svg>
);

const BotAvatar = () => (
  <div className="cgpt-avatar">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z" />
    </svg>
  </div>
);

// ── Citation source helpers ────────────────────────────────────────────────

function getSourceDisplay(source, fallbackIndex) {
  const sourceMetadata = source?.source_metadata;
  const sourceInfo = sourceMetadata?.source_info;
  const headingPath = Array.isArray(sourceMetadata?.heading_path)
    ? sourceMetadata.heading_path
    : Array.isArray(sourceMetadata?.context?.heading_path)
      ? sourceMetadata.context.heading_path
      : [];
  const sectionTitle =
    sourceMetadata?.section_title ||
    sourceMetadata?.title ||
    sourceMetadata?.context?.h6 ||
    sourceMetadata?.context?.h5 ||
    sourceMetadata?.context?.h4 ||
    sourceMetadata?.context?.h3 ||
    sourceMetadata?.context?.h2 ||
    sourceMetadata?.context?.h1 ||
    headingPath[headingPath.length - 1];
  const fileName =
    sourceInfo?.file_name ||
    sourceMetadata?.file_name ||
    sourceMetadata?.original_filename ||
    sourceMetadata?.filename ||
    sourceMetadata?.title ||
    (source?.document_id != null ? `Tài liệu #${source.document_id}` : `Chunk ${fallbackIndex}`);
  const page =
    source?.page ??
    sourceMetadata?.page_start ??
    sourceMetadata?.source_page ??
    sourceInfo?.page_number;
  const pageEnd = sourceMetadata?.page_end;

  return {
    fileName,
    page,
    pageEnd,
    sectionTitle,
    headingPath,
    chunkId: source?.chunk_id ?? sourceMetadata?.parent_chunk_id ?? sourceMetadata?.parent_id,
    chunkIndex: source?.chunk_index,
  };
}

function citationKey(source, chunkNum) {
  return [
    source?.document_id ?? "doc",
    source?.chunk_id ?? source?.source_metadata?.parent_chunk_id ?? "chunk",
    source?.chunk_index ?? "idx",
    source?.page ?? "page",
    chunkNum,
  ].join(":");
}

function isPdfDocument(document) {
  const contentType = String(document?.content_type || "").toLowerCase();
  const filename = String(document?.original_filename || "").toLowerCase();
  return contentType.includes("pdf") || filename.endsWith(".pdf");
}

function normalizeForMatch(value) {
  return String(value || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function findWhitespaceInsensitiveRange(text, query) {
  const source = String(text || "");
  const needle = normalizeForMatch(query);
  if (!source || !needle) return null;

  let normalized = "";
  const indexMap = [];
  let inWhitespace = false;

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (/\s/.test(char)) {
      if (normalized && !inWhitespace) {
        normalized += " ";
        indexMap.push(i);
        inWhitespace = true;
      }
      continue;
    }
    normalized += char.toLowerCase();
    indexMap.push(i);
    inWhitespace = false;
  }

  const matchIndex = normalized.indexOf(needle);
  if (matchIndex === -1) return null;

  const start = indexMap[matchIndex] ?? 0;
  const endMapIndex = matchIndex + needle.length - 1;
  const end = (indexMap[endMapIndex] ?? source.length - 1) + 1;
  return { start, end };
}

function HighlightedSourceText({ text, highlight }) {
  const range = findWhitespaceInsensitiveRange(text, highlight);
  if (!range) return <>{text}</>;

  return (
    <>
      {text.slice(0, range.start)}
      <mark>{text.slice(range.start, range.end)}</mark>
      {text.slice(range.end)}
    </>
  );
}

// ── Citation badge ─────────────────────────────────────────────────────────

function CitationBadge({ chunkNum, source, isActive, onSelect }) {
  const display = getSourceDisplay(source, chunkNum);

  return (
    <span className="cgpt-citation-wrap">
      <button
        className={`cgpt-citation-badge${isActive ? " cgpt-citation-active" : ""}`}
        onClick={() => onSelect?.({ source, chunkNum, key: citationKey(source, chunkNum) })}
        title={`Mở nguồn ${display.fileName}`}
      >
        {chunkNum}
      </button>
    </span>
  );
}

// ── Markdown renderer with inline citations ────────────────────────────────

function makeMdComponents(sources, onCitationSelect, activeCitationKey) {
  function injectCitations(children) {
    if (!sources?.length) return children;
    const items = Array.isArray(children) ? children : [children];
    return items.flatMap((item, i) => {
      if (typeof item !== "string") return [item];
      // Match [Chunk N] or [Chunk N, Chunk M, ...] patterns only
      const parts = item.split(/(\[Chunk \d+(?:,\s*Chunk \d+)*\])/g);
      if (parts.length === 1) return [item];
      return parts.flatMap((part, j) => {
        const chunkNums = [...part.matchAll(/Chunk (\d+)/g)].map((m) => m[1]);
        if (chunkNums.length > 0) {
          return chunkNums
            .map((num, k) => {
              const src = sources[parseInt(num, 10) - 1];
              if (!src) return null;
              const key = citationKey(src, num);
              return (
                <CitationBadge
                  key={`${i}-${j}-${k}`}
                  chunkNum={num}
                  source={src}
                  isActive={key === activeCitationKey}
                  onSelect={onCitationSelect}
                />
              );
            })
            .filter(Boolean);
        }
        return part ? [part] : [];
      });
    });
  }

  return {
    p:      ({ children }) => <p className="md-p">{injectCitations(children)}</p>,
    h1:     ({ children }) => <h1 className="md-h1">{injectCitations(children)}</h1>,
    h2:     ({ children }) => <h2 className="md-h2">{injectCitations(children)}</h2>,
    h3:     ({ children }) => <h3 className="md-h3">{injectCitations(children)}</h3>,
    h4:     ({ children }) => <h4 className="md-h4">{injectCitations(children)}</h4>,
    h5:     ({ children }) => <h5 className="md-h5">{injectCitations(children)}</h5>,
    li:     ({ children }) => <li>{injectCitations(children)}</li>,
    strong: ({ children }) => <strong>{injectCitations(children)}</strong>,
    em:     ({ children }) => <em>{injectCitations(children)}</em>,
    code:   ({ children, className }) => (
      <code className={`md-code${className ? " " + className : ""}`}>{children}</code>
    ),
    pre:    ({ children }) => <pre className="md-pre">{children}</pre>,
  };
}

function MarkdownWithCitations({ content, sources, onCitationSelect, activeCitationKey }) {
  // Strip any stray <think> tags that may leak into the answer
  const clean = content.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<\/?think>/g, "").trim();
  return (
    <div className="md-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={makeMdComponents(sources, onCitationSelect, activeCitationKey)}
      >
        {clean}
      </ReactMarkdown>
    </div>
  );
}

// ── Think block ────────────────────────────────────────────────────────────

function ThinkBlock({ content, isStreaming }) {
  return (
    <details className="think-block" open={isStreaming || undefined}>
      <summary className="think-block-summary">
        {isStreaming ? (
          <>
            <span className="think-spinner" />
            <span>Đang phân tích...</span>
          </>
        ) : (
          <>
            <IconThink />
            <span>Xem quá trình phân tích</span>
          </>
        )}
      </summary>
      <div className="think-block-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
        >
          {content}
        </ReactMarkdown>
      </div>
    </details>
  );
}

// ── Source card (NotebookLM style) ─────────────────────────────────────────

function SourceCard({ source, index }) {
  const sourceMetadata = source?.source_metadata;
  const sourceInfo = sourceMetadata?.source_info;
  const fileName =
    sourceInfo?.file_name ||
    sourceMetadata?.file_name ||
    sourceMetadata?.original_filename ||
    sourceMetadata?.filename ||
    sourceMetadata?.title ||
    (source?.document_id != null ? `Tài liệu #${source.document_id}` : `Nguồn ${index}`);
  const page = source?.page ?? sourceInfo?.page_number;

  return (
    <div className="source-card">
      <div className="source-card-header">
        <div className="source-card-icon">
          <IconDocument />
        </div>
        <div className="source-card-info">
          <div className="source-card-name" title={fileName}>{fileName}</div>
          <div className="source-card-page">{page != null ? `Vị trí: trang ${page}` : "Vị trí: chưa rõ trang"}</div>
        </div>
        <span className="source-card-num">{index}</span>
      </div>
    </div>
  );
}

function SourcesPanel({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="sources-panel">
      <div className="sources-panel-label">
        <IconDocument />
        <span>{sources.length} NGUỒN THAM KHẢO</span>
      </div>
      <div className="sources-cards">
        {sources.map((src, i) => (
          <SourceCard key={i} source={src} index={i + 1} />
        ))}
      </div>
    </div>
  );
}

function SourceViewer({
  activeCitation,
  detail,
  isLoading,
  error,
  filePreviewUrl,
  isFileLoading,
  onClose,
}) {
  if (!activeCitation) return null;

  const fallbackDisplay = getSourceDisplay(activeCitation.source, activeCitation.chunkNum);
  const document = detail?.document;
  const fileName = document?.original_filename || fallbackDisplay.fileName;
  const page = detail?.page ?? fallbackDisplay.page;
  const pageEnd = detail?.page_end ?? fallbackDisplay.pageEnd;
  const pageLabel = page != null
    ? `Trang ${page}${pageEnd && pageEnd !== page ? `-${pageEnd}` : ""}`
    : "Không rõ trang";
  const locationLabel = `Vị trí trích dẫn: ${pageLabel}`;
  const canPreviewPdf = filePreviewUrl && isPdfDocument(document);
  const pdfSrc = canPreviewPdf && page != null
    ? `${filePreviewUrl}#page=${page}`
    : filePreviewUrl;

  return (
    <aside className="source-viewer" aria-label="Nguồn trích dẫn">
      <div className="source-viewer-head">
        <div className="source-viewer-title-wrap">
          <div className="source-viewer-eyebrow">Nguồn {activeCitation.chunkNum}</div>
          <h2 className="source-viewer-title" title={fileName}>{fileName}</h2>
          <div className="source-viewer-subtitle">{locationLabel}</div>
        </div>
        <button className="source-viewer-close" onClick={onClose} aria-label="Đóng nguồn">
          <IconClose />
        </button>
      </div>

      <div className="source-viewer-body">
        {isLoading ? (
          <div className="source-viewer-state">Đang tải vị trí nguồn...</div>
        ) : error ? (
          <div className="source-viewer-state source-viewer-state-error">
            <strong>Không thể tải vị trí trích dẫn.</strong>
            <span>{error}</span>
          </div>
        ) : (
          <>
            <div className="source-viewer-locator">
              <span>{locationLabel}</span>
            </div>

            {document && isPdfDocument(document) && (
              <div className="source-viewer-preview">
                {isFileLoading ? (
                  <div className="source-viewer-state">Đang mở file PDF...</div>
                ) : canPreviewPdf ? (
                  <iframe
                    title={`Preview ${fileName}`}
                    src={pdfSrc}
                    className="source-viewer-frame"
                  />
                ) : (
                  <div className="source-viewer-state">Không thể mở file PDF.</div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}

function normalizeChatMessages(items) {
  return items.map((item) =>
    item.role === "assistant"
      ? {
          ...item,
          isComplete: true,
          error: item.content?.trim()
            ? ""
            : "Câu trả lời trước đó không có nội dung. Bạn vui lòng gửi lại câu hỏi.",
        }
      : item
  );
}

// ── Parse think block from message content ─────────────────────────────────

function parseContent(content) {
  const ts = content.indexOf("<think>");
  const te = content.indexOf("</think>");
  if (ts === -1) return { thinking: null, answer: content, isThinking: false };
  if (te === -1) return { thinking: content.substring(ts + 7), answer: "", isThinking: true };

  const thinking = content.substring(ts + 7, te).trim();
  // Strip any extra <think>...</think> blocks the model may output in the answer
  const answer = content
    .substring(te + 8)
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<\/?think>/g, "")
    .trim();

  return { thinking, answer, isThinking: false };
}

// ── AssistantMessage ───────────────────────────────────────────────────────

function AssistantMessage({ msg, onCitationSelect, activeCitationKey }) {
  const { thinking, answer, isThinking } = parseContent(msg.content);
  const sources = msg.sources ?? [];
  const answerText = thinking !== null ? answer : msg.content;
  const status = msg.error || msg.status || (msg.isComplete ? "" : "Đang xử lý...");

  return (
    <div className="cgpt-assistant-row">
      <BotAvatar />
      <div className="cgpt-assistant-content">
        <div className="cgpt-assistant-name">VTAca RAG</div>

        {thinking !== null && (
          <ThinkBlock content={thinking} isStreaming={isThinking} />
        )}

        {answerText && (
          <MarkdownWithCitations
            content={answerText}
            sources={sources}
            onCitationSelect={onCitationSelect}
            activeCitationKey={activeCitationKey}
          />
        )}

        {!answerText && !thinking && status && (
          <div className={`cgpt-stream-status${msg.error ? " cgpt-stream-error" : ""}`}>
            {!msg.error && (
              <div className="cgpt-typing cgpt-typing-inline">
                <span /><span /><span />
              </div>
            )}
            <span>{status}</span>
          </div>
        )}

        {answerText && msg.isComplete && (
          <div className="cgpt-answer-complete" title="Câu trả lời đã hoàn tất">
            <span className="cgpt-complete-check" aria-hidden="true">✓</span>
            <span>Hoàn tất</span>
          </div>
        )}

      </div>
    </div>
  );
}

// ── ChatPage ───────────────────────────────────────────────────────────────

function ChatPage({ user, onLogout }) {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isReceiving, setIsReceiving] = useState(false);
  const [error, setError] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth >= 769);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCitation, setActiveCitation] = useState(null);
  const [sourceDetail, setSourceDetail] = useState(null);
  const [sourceError, setSourceError] = useState("");
  const [isSourceLoading, setIsSourceLoading] = useState(false);
  const [sourceFileUrl, setSourceFileUrl] = useState("");
  const [isSourceFileLoading, setIsSourceFileLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const sendLockRef = useRef(false);
  const activeCitationKey = activeCitation?.key || "";

  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;
    return sessions.filter((s) =>
      s.title.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [sessions, searchQuery]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  useEffect(() => {
    if (!activeCitation?.source?.document_id) {
      setSourceDetail(null);
      setSourceError("");
      setIsSourceLoading(false);
      return;
    }

    let cancelled = false;
    const source = activeCitation.source;
    setIsSourceLoading(true);
    setSourceError("");
    setSourceDetail(null);

    api
      .get("/chat/source", {
        params: {
          document_id: source.document_id,
          chunk_id: source.chunk_id || undefined,
          chunk_index: source.chunk_index ?? undefined,
          page: source.page || undefined,
        },
      })
      .then((response) => {
        if (!cancelled) {
          setSourceDetail(response.data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          const detail = err?.response?.data?.detail;
          setSourceError(typeof detail === "string" ? detail : "Không tìm thấy vị trí nguồn.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsSourceLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeCitationKey]);

  useEffect(() => {
    if (!sourceDetail?.document || !sourceDetail.file_available || !isPdfDocument(sourceDetail.document)) {
      setSourceFileUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return "";
      });
      setIsSourceFileLoading(false);
      return;
    }

    let cancelled = false;
    let objectUrl = "";
    setIsSourceFileLoading(true);
    setSourceFileUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return "";
    });

    api
      .get(`/chat/source/file/${sourceDetail.document.id}`, { responseType: "blob" })
      .then((response) => {
        if (cancelled) return;
        const contentType = response.headers?.["content-type"] || sourceDetail.document.content_type || "application/pdf";
        objectUrl = URL.createObjectURL(new Blob([response.data], { type: contentType }));
        setSourceFileUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setSourceFileUrl("");
      })
      .finally(() => {
        if (!cancelled) setIsSourceFileLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [sourceDetail?.document?.id, sourceDetail?.file_available]);

  function closeSourceViewer() {
    setActiveCitation(null);
    setSourceDetail(null);
    setSourceError("");
  }

  function handleCitationSelect(nextCitation) {
    setActiveCitation(nextCitation);
  }

  async function fetchSessions() {
    const response = await api.get("/chat/sessions");
    setSessions(response.data);
  }

  async function fetchMessages(sessionId) {
    if (!sessionId) { setMessages([]); return; }
    const response = await api.get(`/chat/sessions/${sessionId}/messages`);
    setMessages(normalizeChatMessages(response.data));
  }

  function createSession() {
    setActiveSessionId(null);
    setMessages([]);
    closeSourceViewer();
  }

  async function deleteSession(sessionId) {
    await api.delete(`/chat/sessions/${sessionId}`);
    const remaining = sessions.filter((item) => item.id !== sessionId);
    setSessions(remaining);
    if (activeSessionId === sessionId) {
      closeSourceViewer();
      const next = remaining[0]?.id || null;
      setActiveSessionId(next);
      if (next) await fetchMessages(next);
      else setMessages([]);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    if (!input.trim() || isSending || sendLockRef.current) return;
    sendLockRef.current = true;
    setError("");
    setIsSending(true);
    setIsReceiving(false);
    const userText = input.trim();
    setInput("");
    closeSourceViewer();
    setMessages((prev) => [
      ...prev,
      { id: `pending-${Date.now()}`, role: "user", content: userText, sources: [] },
    ]);

    const assistantMessageId = `assistant-${Date.now()}`;
    let isAssistantMessageAdded = false;

    try {
      const headers = { "Content-Type": "application/json" };
      const token = localStorage.getItem("rag_admin_token");
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const body = { session_id: activeSessionId, message: userText };

      const response = await fetch(`${api.defaults.baseURL}/chat/query`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let doneReading = false;
      let newSessionId = activeSessionId;
      let buffer = "";

      let pendingTokenText = "";
      let streamFrameId = null;
      let isStreamFinished = false;
      let shouldMarkComplete = false;
      let resolveQueue = null;
      const queuePromise = new Promise((resolve) => { resolveQueue = resolve; });

      function ensureAssistantMessage(status = "Đang xử lý...") {
        if (isAssistantMessageAdded) return;
        setMessages((prev) => [
          ...prev,
          { id: assistantMessageId, role: "assistant", content: "", sources: [], status, isComplete: false },
        ]);
        isAssistantMessageAdded = true;
        setIsReceiving(true);
      }

      function updateAssistantStatus(status) {
        ensureAssistantMessage(status);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, status } : msg
          )
        );
      }

      function scheduleTokenFlush() {
        if (streamFrameId !== null) return;
        streamFrameId = window.requestAnimationFrame(flushPendingTokens);
      }

      function flushPendingTokens() {
        streamFrameId = null;
        if (pendingTokenText) {
          const chunk = pendingTokenText;
          pendingTokenText = "";
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, content: msg.content + chunk, status: "", isComplete: shouldMarkComplete }
                : msg
            )
          );
          shouldMarkComplete = false;
        }
        if (pendingTokenText) {
          scheduleTokenFlush();
        } else if (isStreamFinished) {
          resolveQueue();
        }
      }

      function appendStreamToken(content) {
        ensureAssistantMessage("");
        pendingTokenText += content || "";
        scheduleTokenFlush();
      }

      while (!doneReading) {
        const { value, done } = await reader.read();
        doneReading = done;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() || "";

          for (const eventBlock of events) {
            const lines = eventBlock.split(/\r?\n/);
            const dataLines = lines
              .map((line) => line.trim())
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.replace(/^data:\s?/, ""));
            if (dataLines.length === 0) continue;
            try {
              const data = JSON.parse(dataLines.join("\n"));
              if (data.type === "session") {
                ensureAssistantMessage("Đang xử lý...");
                newSessionId = data.session_id;
              } else if (data.type === "status") {
                updateAssistantStatus(data.message || "Đang xử lý...");
              } else if (data.type === "sources") {
                ensureAssistantMessage("Đang chuẩn bị câu trả lời...");
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId ? { ...msg, sources: data.sources } : msg
                  )
                );
              } else if (data.type === "output_mode") {
                ensureAssistantMessage("Đang chuẩn bị câu trả lời...");
              } else if (data.type === "token") {
                appendStreamToken(data.content);
              } else if (data.type === "done") {
                shouldMarkComplete = true;
                if (!pendingTokenText) {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, status: "", isComplete: true }
                        : msg
                    )
                  );
                }
              } else if (data.type === "error") {
                const detail = data.detail || "Có lỗi xảy ra trong quá trình tạo câu trả lời.";
                setError(detail);
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId
                      ? { ...msg, status: "", error: detail, isComplete: true }
                      : msg
                  )
                );
              }
            } catch (e) {
              console.error("Error parsing SSE data", e, eventBlock);
            }
          }
        }
      }

      isStreamFinished = true;
      if (pendingTokenText) scheduleTokenFlush();
      else if (streamFrameId === null) resolveQueue();
      await queuePromise;

      await fetchSessions();
      if (newSessionId && newSessionId !== activeSessionId) {
        setActiveSessionId(newSessionId);
      } else if (newSessionId) {
        await fetchMessages(newSessionId);
      }
    } catch (err) {
      console.error(err);
      setError("Không thể gửi câu hỏi hoặc mất kết nối.");
    } finally {
      sendLockRef.current = false;
      setIsSending(false);
      setIsReceiving(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key !== "Enter") return;
    if (event.shiftKey || event.nativeEvent.isComposing) return;
    sendMessage(event);
  }

  useEffect(() => {
    fetchSessions().catch(() => setError("Không thể tải danh sách phiên chat."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchMessages(activeSessionId).catch(() => setError("Không thể tải tin nhắn."));
  }, [activeSessionId]);

  return (
    <div className={`cgpt-shell${isSidebarOpen ? "" : " cgpt-sidebar-hidden"}`}>
      {/* ── SIDEBAR ── */}
      <aside className="cgpt-sidebar">
        <div className="cgpt-sidebar-top">
          <button className="cgpt-icon-btn" onClick={() => setIsSidebarOpen(false)} aria-label="Đóng sidebar">
            <IconMenu />
          </button>
          <button className="cgpt-icon-btn" onClick={createSession} aria-label="Đoạn chat mới">
            <IconEdit />
          </button>
        </div>

        <div className="cgpt-search-box">
          <IconSearch />
          <input
            type="text"
            className="cgpt-search-input"
            placeholder="Tìm kiếm đoạn chat"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <nav className="cgpt-nav">
          <button className="cgpt-new-chat-btn" onClick={createSession}>
            <IconEdit />
            <span>Đoạn chat mới</span>
          </button>

          {filteredSessions.length > 0 && (
            <>
              <div className="cgpt-section-label">Gần đây</div>
              {filteredSessions.map((session) => (
                <div
                  key={session.id}
                  className={`cgpt-session${session.id === activeSessionId ? " cgpt-session-active" : ""}`}
                >
                  <button
                    className="cgpt-session-btn"
                    onClick={() => {
                      closeSourceViewer();
                      setActiveSessionId(session.id);
                    }}
                  >
                    <span className="cgpt-session-title">{session.title}</span>
                  </button>
                  <button className="cgpt-session-del" onClick={() => deleteSession(session.id)} aria-label="Xóa">
                    <IconTrash />
                  </button>
                </div>
              ))}
            </>
          )}

          {filteredSessions.length === 0 && searchQuery && (
            <p className="cgpt-no-results">Không tìm thấy đoạn chat</p>
          )}
        </nav>
      </aside>

      {/* ── MAIN ── */}
      <main className={`cgpt-main${activeCitation ? " cgpt-main-source-open" : ""}`}>
        {isSidebarOpen && (
          <div className="cgpt-backdrop" onClick={() => setIsSidebarOpen(false)} aria-hidden="true" />
        )}

        <header className="cgpt-topbar">
          <div className="cgpt-topbar-left">
            <button
              className="cgpt-icon-btn cgpt-topbar-menu"
              onClick={() => setIsSidebarOpen((v) => !v)}
              aria-label={isSidebarOpen ? "Đóng sidebar" : "Mở sidebar"}
            >
              <IconMenu />
            </button>
            <button className="cgpt-icon-btn cgpt-topbar-edit" onClick={createSession} aria-label="Đoạn chat mới">
              <IconEdit />
            </button>
          </div>
          <div className="cgpt-model-pill">
            <span>Trợ lý tri thức VTAca</span>
            <IconChevronDown />
          </div>
          <div className="cgpt-topbar-right">
            {user?.role === "admin" && (
              <Link to="/admin" className="btn-outline ghost-link" style={{ marginRight: "1rem", padding: "0.4rem 0.8rem", fontSize: "0.9rem" }}>
                Admin Dashboard
              </Link>
            )}
            <button
              className="ghost-link"
              onClick={onLogout}
              aria-label="Đăng xuất"
              style={{ border: "none", background: "transparent", cursor: "pointer", padding: "0.4rem 0.8rem", fontSize: "0.9rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}
            >
              Đăng xuất
            </button>
          </div>
        </header>

        <div className="cgpt-content-split">
          <section className="cgpt-chat-pane">
            {/* Messages */}
            <div className="cgpt-messages-wrap">
              {messages.length === 0 ? (
                <div className="cgpt-empty">
                  <h1>Chúng ta nên bắt đầu từ đâu?</h1>
                </div>
              ) : (
                <div className="cgpt-messages">
                  {messages.map((msg) => (
                    <div key={msg.id} className={`cgpt-msg cgpt-msg-${msg.role}`}>
                      {msg.role === "user" ? (
                        <div className="cgpt-user-bubble">{msg.content}</div>
                      ) : (
                        <AssistantMessage
                          msg={msg}
                          onCitationSelect={handleCitationSelect}
                          activeCitationKey={activeCitationKey}
                        />
                      )}
                    </div>
                  ))}

                  {isSending && !isReceiving && (
                    <div className="cgpt-msg cgpt-msg-assistant">
                      <div className="cgpt-assistant-row">
                        <BotAvatar />
                        <div className="cgpt-assistant-content">
                          <div className="cgpt-assistant-name">VTAca RAG</div>
                          <div className="cgpt-typing">
                            <span /><span /><span />
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Composer */}
            <div className="cgpt-composer-wrap">
              <form className="cgpt-composer" onSubmit={sendMessage}>
                <div className="cgpt-composer-inner">
                  <button type="button" className="cgpt-attach-btn" aria-label="Thêm nội dung">
                    <IconPlus />
                  </button>
                  <textarea
                    ref={textareaRef}
                    className="cgpt-textarea"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Hỏi bất kỳ điều gì"
                    rows={1}
                  />
                  <button
                    type="submit"
                    className={`cgpt-send-btn${input.trim() && !isSending ? " cgpt-send-active" : ""}`}
                    disabled={isSending || !input.trim()}
                    aria-label="Gửi"
                  >
                    <IconSend />
                  </button>
                </div>
              </form>
              {error && <p className="cgpt-error">{error}</p>}
              <p className="cgpt-disclaimer">Trợ lý tri thức có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.</p>
            </div>
          </section>

          <SourceViewer
            activeCitation={activeCitation}
            detail={sourceDetail}
            isLoading={isSourceLoading}
            error={sourceError}
            filePreviewUrl={sourceFileUrl}
            isFileLoading={isSourceFileLoading}
            onClose={closeSourceViewer}
          />
        </div>
      </main>
    </div>
  );
}

export default ChatPage;
