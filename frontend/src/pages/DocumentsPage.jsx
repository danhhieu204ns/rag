import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api";

const CHUNK_PAGE_SIZE = 10;
const DOCUMENT_PROGRESS_POLL_INTERVAL_MS = 2500;

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadTitle, setUploadTitle] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState("");
  const [activeDocumentAction, setActiveDocumentAction] = useState({
    documentId: null,
    action: "",
  });
  const [error, setError] = useState("");

  const [selectedChunkDocumentId, setSelectedChunkDocumentId] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [chunkOffset, setChunkOffset] = useState(0);
  const [isChunksBusy, setIsChunksBusy] = useState(false);
  const [chunksError, setChunksError] = useState("");
  const [showReindexConfirm, setShowReindexConfirm] = useState(false);
  const isPollingRef = useRef(false);

  const totalChunks = useMemo(
    () => documents.reduce((sum, item) => sum + (item.chunk_count || 0), 0),
    [documents]
  );

  const selectedChunkDocument = useMemo(
    () => documents.find((item) => item.id === selectedChunkDocumentId) || null,
    [documents, selectedChunkDocumentId]
  );

  const hasIndexingDocuments = useMemo(
    () => documents.some((item) => item.status === "indexing"),
    [documents]
  );

  const chunkRangeStart = chunkTotal === 0 ? 0 : chunkOffset + 1;
  const chunkRangeEnd = Math.min(chunkOffset + CHUNK_PAGE_SIZE, chunkTotal);
  const canGoChunkPrev = chunkOffset > 0;
  const canGoChunkNext = chunkOffset + CHUNK_PAGE_SIZE < chunkTotal;

  function closeChunksInspector() {
    setSelectedChunkDocumentId(null);
    setChunks([]);
    setChunkTotal(0);
    setChunkOffset(0);
    setChunksError("");
  }

  async function fetchDocuments() {
    const response = await api.get("/documents");
    setDocuments(response.data);

    if (selectedChunkDocumentId && !response.data.some((item) => item.id === selectedChunkDocumentId)) {
      closeChunksInspector();
    }
  }

  async function fetchDocumentChunks(documentId, offset = 0) {
    setIsChunksBusy(true);
    setChunksError("");

    try {
      const response = await api.get(`/documents/${documentId}/chunks`, {
        params: {
          offset,
          limit: CHUNK_PAGE_SIZE,
        },
      });

      setSelectedChunkDocumentId(documentId);
      setChunks(response.data.items || []);
      setChunkTotal(response.data.total_chunks || 0);
      setChunkOffset(response.data.offset || 0);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setChunksError(typeof detail === "string" ? detail : "Không thể tải danh sách chunks.");
    } finally {
      setIsChunksBusy(false);
    }
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!selectedFile) return;

    setIsBusy(true);
    setBusyMessage("Đang tải tài liệu lên...");
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      if (uploadTitle.trim()) {
        formData.append("title", uploadTitle.trim());
      }
      await api.post("/documents/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSelectedFile(null);
      setUploadTitle("");
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể tải tài liệu lên.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  async function processDocument(documentId) {
    setIsBusy(true);
    setBusyMessage("Đang bắt đầu index tài liệu...");
    setActiveDocumentAction({ documentId, action: "process" });
    setError("");
    try {
      await api.post(`/documents/${documentId}/process`);
      setBusyMessage("Đã đưa tài liệu vào hàng đợi index, đang theo dõi tiến độ...");
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể bắt đầu index tài liệu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
      setActiveDocumentAction({ documentId: null, action: "" });
    }
  }


  async function deleteDocument(documentId) {
    setIsBusy(true);
    setBusyMessage("Đang xóa tài liệu...");
    setError("");
    try {
      await api.delete(`/documents/${documentId}`);
      if (selectedChunkDocumentId === documentId) {
        closeChunksInspector();
      }
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể xóa tài liệu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  function openReindexConfirm() {
    setShowReindexConfirm(true);
  }

  async function handleConfirmReindex() {
    setShowReindexConfirm(false);
    setIsBusy(true);
    setBusyMessage("Đang đưa các tài liệu cần index vào hàng đợi...");
    setError("");
    try {
      await api.post("/documents/reindex");
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể index lại tài liệu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  function handleCancelReindex() {
    setShowReindexConfirm(false);
  }

  function handleChunkToggle(doc) {
    if (selectedChunkDocumentId === doc.id) {
      closeChunksInspector();
      return;
    }
    fetchDocumentChunks(doc.id, 0);
  }

  function formatChunkDate(value) {
    if (!value) return "";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return "";
    }
    return parsed.toLocaleString();
  }

  function getStatusLabel(status) {
    const labels = {
      uploaded: "Chưa index",
      indexing: "Đang index",
      embedded: "Đã index",
      parsed: "Đã parse",
      index_failed: "Index lỗi",
    };
    return labels[status] || status || "Không rõ";
  }

  function getStatusClass(status) {
    if (status === "embedded") return "success";
    if (status === "indexing") return "progress";
    if (status === "index_failed") return "danger";
    return "neutral";
  }

  useEffect(() => {
    fetchDocuments().catch(() => setError("Không thể tải danh sách tài liệu."));
  }, []);

  useEffect(() => {
    if (!hasIndexingDocuments) {
      return;
    }

    const poll = async () => {
      if (isPollingRef.current) {
        return;
      }

      isPollingRef.current = true;
      try {
        await fetchDocuments();
      } catch {
        // Polling should stay silent; user-facing errors are handled by explicit actions.
      } finally {
        isPollingRef.current = false;
      }
    };

    poll();
    const timer = window.setInterval(poll, DOCUMENT_PROGRESS_POLL_INTERVAL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [hasIndexingDocuments]);

  return (

    <div className="panel panel-main admin-docs-panel">
      <div className="panel-head">
        <h2>Quản lý tài liệu</h2>
        <div className="panel-actions">
          <span className="metric-pill">Tài liệu: {documents.length}</span>
          <span className="metric-pill">Tổng chunks: {totalChunks}</span>
          {hasIndexingDocuments ? <span className="status-pill progress">Đang cập nhật tiến độ</span> : null}
          <button className="soft-button" onClick={openReindexConfirm} disabled={isBusy}>
            Index tài liệu còn thiếu
          </button>
        </div>
      </div>

      <form className="upload-form" onSubmit={uploadDocument}>
        <input
          type="file"
          onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
          accept=".pdf,.txt,.md"
        />
        <input
          type="text"
          placeholder="Tiêu đề (không bắt buộc)"
          value={uploadTitle}
          onChange={(event) => setUploadTitle(event.target.value)}
        />
        <button className="primary" type="submit" disabled={isBusy || !selectedFile}>
          {isBusy ? "Đang xử lý..." : "Tải lên"}
        </button>
      </form>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Tiêu đề</th>
              <th>File gốc</th>
              <th>Trạng thái</th>
              <th>Chunks</th>
              <th>Thao tác</th>

            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.id}>
                <td>{doc.id}</td>
                <td className="document-title-cell">{doc.title}</td>
                <td>{doc.original_filename}</td>
                <td>
                  <span className={`status-pill ${getStatusClass(doc.status)}`}>
                    {getStatusLabel(doc.status)}
                  </span>
                </td>
                <td>{doc.chunk_count}</td>
                <td className="row-actions">
                  {doc.status !== "embedded" && doc.status !== "indexing" ? (
                    <button
                      onClick={() => processDocument(doc.id)}
                      disabled={isBusy}
                      className="primary"
                    >
                      {isBusy && activeDocumentAction.documentId === doc.id && activeDocumentAction.action === "process"
                        ? "Đang index..."
                        : "Index"}
                    </button>
                  ) : null}
                  <button onClick={() => handleChunkToggle(doc)} disabled={isBusy || isChunksBusy}>
                    {selectedChunkDocumentId === doc.id ? "Ẩn chunks" : "Xem chunks"}
                  </button>
                  <button className="danger" onClick={() => deleteDocument(doc.id)} disabled={isBusy}>
                    Xóa
                  </button>
                </td>
              </tr>
            ))}
            {documents.length === 0 ? (
              <tr>
                <td colSpan="6" className="empty-cell">Chưa có tài liệu nào.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {selectedChunkDocumentId ? (
        <section className="chunk-inspector">
          <div className="chunk-inspector-head">
            <div>
              <h3>
                Chunks - Tài liệu #{selectedChunkDocumentId}
                {selectedChunkDocument ? `: ${selectedChunkDocument.title}` : ""}
              </h3>
              <p className="muted">
                Tổng: {chunkTotal}
                {chunkTotal > 0 ? ` | Đang hiển thị ${chunkRangeStart}-${chunkRangeEnd}` : ""}
              </p>
            </div>
            <div className="chunk-inspector-actions">
              <button
                onClick={() =>
                  fetchDocumentChunks(
                    selectedChunkDocumentId,
                    Math.max(0, chunkOffset - CHUNK_PAGE_SIZE)
                  )
                }
                disabled={!canGoChunkPrev || isChunksBusy}
              >
                Trước
              </button>
              <button
                onClick={() =>
                  fetchDocumentChunks(selectedChunkDocumentId, chunkOffset + CHUNK_PAGE_SIZE)
                }
                disabled={!canGoChunkNext || isChunksBusy}
              >
                Sau
              </button>
              <button className="danger" onClick={closeChunksInspector}>
                Đóng
              </button>
            </div>
          </div>

          {isChunksBusy ? <p className="muted">Đang tải chunks...</p> : null}
          {chunksError ? <p className="error-text">{chunksError}</p> : null}
          {!isChunksBusy && !chunksError && chunks.length === 0 ? (
            <p className="muted">Tài liệu này chưa có chunk. Bấm Index để tạo chunks.</p>
          ) : null}

          <div className="chunk-list">
            {chunks.map((chunk) => {
              const metadata = chunk.source_metadata || {};
              const hasMetadata = Object.keys(metadata).length > 0;

              return (
                <article className="chunk-card" key={chunk.id}>
                  <div className="chunk-card-head">
                    <strong>Chunk #{chunk.chunk_index}</strong>
                    <span className="chunk-pill">id {chunk.id}</span>
                    {chunk.source_page ? <span className="chunk-pill">trang {chunk.source_page}</span> : null}
                    {chunk.source_kind ? <span className="chunk-pill">{chunk.source_kind}</span> : null}
                    {chunk.created_at ? (
                      <span className="chunk-pill">{formatChunkDate(chunk.created_at)}</span>
                    ) : null}
                  </div>

                  <details open>
                    <summary>Nội dung</summary>
                    <pre className="chunk-pre">{chunk.content}</pre>
                  </details>

                  <details>
                    <summary>Metadata</summary>
                    <pre className="chunk-pre">
                      {hasMetadata ? JSON.stringify(metadata, null, 2) : "{}"}
                    </pre>
                  </details>
                </article>
              );
            })}
          </div>
        </section>
      ) : null}

      {error ? <p className="error-text">{error}</p> : null}
      {isBusy && busyMessage ? <p className="muted busy-text">{busyMessage}</p> : null}

      {showReindexConfirm ? (
        <div className="modal-overlay" onClick={handleCancelReindex}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Xác nhận index lại tài liệu</h3>
            </div>
            <div className="modal-body">
              <p>Bạn có chắc muốn index lại các tài liệu còn thiếu không?</p>
              <p className="muted" style={{ fontSize: "0.9em", marginTop: "8px" }}>
                Các tài liệu sẽ được đưa vào hàng đợi xử lý.
              </p>
            </div>
            <div className="modal-footer">
              <button className="soft-button" onClick={handleCancelReindex}>
                Hủy
              </button>
              <button className="primary" onClick={handleConfirmReindex}>
                Xác nhận
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default DocumentsPage;
