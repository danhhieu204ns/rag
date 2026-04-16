import { useEffect, useMemo, useRef, useState } from "react";
import api from "../api";

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

const BotAvatar = () => (
  <div className="cgpt-avatar">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z" />
    </svg>
  </div>
);

function ChatPage() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth >= 769);
  const [searchQuery, setSearchQuery] = useState("");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

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

  async function fetchSessions() {
    const response = await api.get("/chat/sessions");
    setSessions(response.data);
    if (!activeSessionId && response.data.length > 0) {
      setActiveSessionId(response.data[0].id);
    }
  }

  async function fetchMessages(sessionId) {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    const response = await api.get(`/chat/sessions/${sessionId}/messages`);
    setMessages(response.data);
  }

  async function createSession() {
    const response = await api.post("/chat/sessions", { title: "New chat" });
    await fetchSessions();
    setActiveSessionId(response.data.id);
    setMessages([]);
  }

  async function deleteSession(sessionId) {
    await api.delete(`/chat/sessions/${sessionId}`);
    const remaining = sessions.filter((item) => item.id !== sessionId);
    setSessions(remaining);
    if (activeSessionId === sessionId) {
      const next = remaining[0]?.id || null;
      setActiveSessionId(next);
      if (next) await fetchMessages(next);
      else setMessages([]);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    if (!input.trim() || isSending) return;
    setError("");
    setIsSending(true);
    const userText = input.trim();
    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: `pending-${Date.now()}`, role: "user", content: userText, sources: [] },
    ]);
    try {
      const response = await api.post("/chat/query", {
        session_id: activeSessionId,
        message: userText,
      });
      const sessionId = response.data.session_id;
      setActiveSessionId(sessionId);
      await fetchSessions();
      await fetchMessages(sessionId);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể gửi câu hỏi.");
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key !== "Enter") return;
    if (event.shiftKey || event.nativeEvent.isComposing) return;
    sendMessage(event);
  }

  function buildSourceLabel(source) {
    const segments = [`Doc #${source.document_id ?? "?"}`];
    if (source.page) {
      segments.push(`Trang ${source.page}`);
    }
    if (source.chunk_index !== null && source.chunk_index !== undefined) {
      segments.push(`Chunk ${source.chunk_index}`);
    }
    return segments.join(" | ");
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
          <button
            className="cgpt-icon-btn"
            onClick={() => setIsSidebarOpen(false)}
            aria-label="Đóng sidebar"
          >
            <IconMenu />
          </button>
          <button
            className="cgpt-icon-btn"
            onClick={createSession}
            aria-label="Đoạn chat mới"
          >
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
                    onClick={() => setActiveSessionId(session.id)}
                  >
                    <span className="cgpt-session-title">{session.title}</span>
                  </button>
                  <button
                    className="cgpt-session-del"
                    onClick={() => deleteSession(session.id)}
                    aria-label="Xóa"
                  >
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
      <main className="cgpt-main">
        {/* Backdrop – chỉ hiện trên mobile khi sidebar mở */}
        {isSidebarOpen && (
          <div
            className="cgpt-backdrop"
            onClick={() => setIsSidebarOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* Topbar */}
        <header className="cgpt-topbar">
          <div className="cgpt-topbar-left">
            <button
              className="cgpt-icon-btn cgpt-topbar-menu"
              onClick={() => setIsSidebarOpen((v) => !v)}
              aria-label={isSidebarOpen ? "Đóng sidebar" : "Mở sidebar"}
            >
              <IconMenu />
            </button>
            <button
              className="cgpt-icon-btn cgpt-topbar-edit"
              onClick={createSession}
              aria-label="Đoạn chat mới"
            >
              <IconEdit />
            </button>
          </div>
          <div className="cgpt-model-pill">
            <span>ViettelRAG</span>
            <IconChevronDown />
          </div>
          <div className="cgpt-topbar-right" />
        </header>

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
                    <div className="cgpt-assistant-row">
                      <BotAvatar />
                      <div className="cgpt-assistant-content">
                        <div className="cgpt-assistant-name">ViettelRAG</div>
                        <div className="cgpt-msg-text">{msg.content}</div>
                        {msg.sources?.length > 0 && (
                          <details className="cgpt-sources">
                            <summary>
                              Nguồn tham khảo ({msg.sources.length})
                            </summary>
                            <ul>
                              {msg.sources.map((src, i) => (
                                <li key={`${msg.id}-${i}`}>
                                  <strong>Tài liệu #{src.document_id ?? "?"}</strong>:{" "}
                                  {src.excerpt}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isSending && (
                <div className="cgpt-msg cgpt-msg-assistant">
                  <div className="cgpt-assistant-row">
                    <BotAvatar />
                    <div className="cgpt-assistant-content">
                      <div className="cgpt-assistant-name">ViettelRAG</div>
                      <div className="cgpt-typing">
                        <span />
                        <span />
                        <span />
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
              <button
                type="button"
                className="cgpt-attach-btn"
                aria-label="Thêm nội dung"
              >
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
          <p className="cgpt-disclaimer">
            ViettelRAG có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.
          </p>
        </div>
      </main>
    </div>
  );
}

export default ChatPage;
