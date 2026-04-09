import { useEffect, useMemo, useState } from "react";
import api from "../api";

function ChatPage() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");

  const activeSession = useMemo(
    () => sessions.find((item) => item.id === activeSessionId) || null,
    [sessions, activeSessionId]
  );

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
      if (next) {
        await fetchMessages(next);
      } else {
        setMessages([]);
      }
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    if (!input.trim()) return;

    setError("");
    setIsSending(true);
    const userText = input.trim();
    setInput("");

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

  function handleChatInputKeyDown(event) {
    if (event.key !== "Enter") {
      return;
    }

    if (event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

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
    <div className="layout-grid">
      <section className="panel panel-side">
        <div className="panel-head">
          <h2>Chat sessions</h2>
          <button onClick={createSession}>+ New</button>
        </div>

        <div className="session-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={session.id === activeSessionId ? "session-item active" : "session-item"}
            >
              <button className="session-select" onClick={() => setActiveSessionId(session.id)}>
                <span className="session-title">{session.title}</span>
              </button>
              <button className="session-delete" onClick={() => deleteSession(session.id)}>
                x
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel panel-main">
        <div className="panel-head">
          <h2>{activeSession ? activeSession.title : "Hỏi đáp"}</h2>
        </div>

        <div className="messages">
          {messages.length === 0 ? (
            <p className="muted">Chưa có tin nhắn. Hãy đặt câu hỏi đầu tiên.</p>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <header>{message.role === "user" ? "Ban" : "Assistant"}</header>
                <p>{message.content}</p>
                {message.role === "assistant" && message.sources?.length > 0 ? (
                  <details>
                    <summary>Nguon ({message.sources.length})</summary>
                    <ul>
                      {message.sources.map((source, index) => (
                        <li key={`${message.id}-${index}`}>
                          Doc #{source.document_id ?? "?"}: {source.excerpt}
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : null}
              </article>
            ))
          )}
        </div>

        <form className="chat-form" onSubmit={sendMessage}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleChatInputKeyDown}
            placeholder="Nhập câu hỏi..."
            rows={3}
          />
          <button type="submit" disabled={isSending}>
            {isSending ? "Đang gửi..." : "Gửi"}
          </button>
        </form>

        {error ? <p className="error-text">{error}</p> : null}
      </section>
    </div>
  );
}

export default ChatPage;
