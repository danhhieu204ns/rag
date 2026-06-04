import { useEffect, useRef, useState } from "react";
import api from "../api";

const DOCUMENT_PROGRESS_POLL_INTERVAL_MS = 2500;
const EMPTY_FILE_VIEWER = {
  document: null,
  objectUrl: "",
  textContent: "",
  contentType: "",
  isLoading: false,
  error: "",
};

function isPdfDocument(document, contentTypeOverride = "") {
  const contentType = String(contentTypeOverride || document?.content_type || "").toLowerCase();
  const filename = String(document?.original_filename || "").toLowerCase();
  return contentType.includes("pdf") || filename.endsWith(".pdf");
}

function isTextDocument(document, contentTypeOverride = "") {
  const contentType = String(contentTypeOverride || document?.content_type || "").toLowerCase();
  const filename = String(document?.original_filename || "").toLowerCase();
  return (
    contentType.startsWith("text/") ||
    contentType.includes("markdown") ||
    filename.endsWith(".txt") ||
    filename.endsWith(".md")
  );
}

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
  const [fileViewer, setFileViewer] = useState(EMPTY_FILE_VIEWER);
  const [showReindexConfirm, setShowReindexConfirm] = useState(false);
  const isPollingRef = useRef(false);
  const fileViewerRequestRef = useRef(0);
  const fileViewerObjectUrlRef = useRef("");

  const hasIndexingDocuments = documents.some((item) => item.status === "indexing");

  async function fetchDocuments() {
    const response = await api.get("/documents");
    setDocuments(response.data);
  }

  function closeFileViewer() {
    fileViewerRequestRef.current += 1;
    setFileViewer((current) => {
      if (current.objectUrl) {
        URL.revokeObjectURL(current.objectUrl);
      }
      fileViewerObjectUrlRef.current = "";
      return EMPTY_FILE_VIEWER;
    });
  }

  async function openDocumentFile(doc) {
    const requestId = fileViewerRequestRef.current + 1;
    fileViewerRequestRef.current = requestId;
    setError("");
    setFileViewer((current) => {
      if (current.objectUrl) {
        URL.revokeObjectURL(current.objectUrl);
      }
      fileViewerObjectUrlRef.current = "";
      return {
        ...EMPTY_FILE_VIEWER,
        document: doc,
        isLoading: true,
      };
    });

    try {
      const response = await api.get(`/documents/${doc.id}/file`, { responseType: "blob" });
      if (fileViewerRequestRef.current !== requestId) {
        return;
      }

      const contentType =
        response.headers?.["content-type"] ||
        response.data?.type ||
        doc.content_type ||
        "application/octet-stream";
      const blob = response.data instanceof Blob
        ? response.data
        : new Blob([response.data], { type: contentType });
      const typedBlob = blob.type ? blob : new Blob([blob], { type: contentType });

      if (isTextDocument(doc, contentType)) {
        const textContent = await typedBlob.text();
        if (fileViewerRequestRef.current !== requestId) {
          return;
        }
        setFileViewer({
          ...EMPTY_FILE_VIEWER,
          document: doc,
          textContent,
          contentType,
        });
        return;
      }

      const objectUrl = URL.createObjectURL(typedBlob);
      if (fileViewerRequestRef.current !== requestId) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      fileViewerObjectUrlRef.current = objectUrl;
      setFileViewer({
        ...EMPTY_FILE_VIEWER,
        document: doc,
        objectUrl,
        contentType,
      });
    } catch (err) {
      if (fileViewerRequestRef.current !== requestId) {
        return;
      }
      const detail = err?.response?.data?.detail;
      setFileViewer({
        ...EMPTY_FILE_VIEWER,
        document: doc,
        error: typeof detail === "string" ? detail : "Không thể mở file tài liệu.",
      });
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
      if (fileViewer.document?.id === documentId) {
        closeFileViewer();
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
    return () => {
      fileViewerRequestRef.current += 1;
      if (fileViewerObjectUrlRef.current) {
        URL.revokeObjectURL(fileViewerObjectUrlRef.current);
        fileViewerObjectUrlRef.current = "";
      }
    };
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
                  <button onClick={() => openDocumentFile(doc)} disabled={isBusy || fileViewer.isLoading}>
                    {fileViewer.isLoading && fileViewer.document?.id === doc.id ? "Đang mở..." : "Xem file"}
                  </button>
                  <button className="danger" onClick={() => deleteDocument(doc.id)} disabled={isBusy}>
                    Xóa
                  </button>
                </td>
              </tr>
            ))}
            {documents.length === 0 ? (
              <tr>
                <td colSpan="5" className="empty-cell">Chưa có tài liệu nào.</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {fileViewer.document ? (
        <div className="modal-overlay document-viewer-overlay" onClick={closeFileViewer}>
          <div
            className="modal-dialog document-viewer-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Xem file tài liệu"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="document-viewer-header">
              <div className="document-viewer-title-wrap">
                <div className="document-viewer-eyebrow">Tài liệu #{fileViewer.document.id}</div>
                <h3 title={fileViewer.document.title}>{fileViewer.document.title}</h3>
                <p title={fileViewer.document.original_filename}>
                  {fileViewer.document.original_filename}
                </p>
              </div>
              <div className="document-viewer-actions">
                {fileViewer.objectUrl ? (
                  <a className="soft-link-button" href={fileViewer.objectUrl} target="_blank" rel="noreferrer">
                    Mở tab mới
                  </a>
                ) : null}
                <button className="danger" onClick={closeFileViewer}>
                  Đóng
                </button>
              </div>
            </div>

            <div className="document-viewer-body">
              {fileViewer.isLoading ? (
                <div className="document-viewer-state">Đang mở file tài liệu...</div>
              ) : null}

              {!fileViewer.isLoading && fileViewer.error ? (
                <div className="document-viewer-state document-viewer-state-error">
                  {fileViewer.error}
                </div>
              ) : null}

              {!fileViewer.isLoading &&
              !fileViewer.error &&
              isPdfDocument(fileViewer.document, fileViewer.contentType) &&
              fileViewer.objectUrl ? (
                <iframe
                  title={`Xem ${fileViewer.document.original_filename}`}
                  src={fileViewer.objectUrl}
                  className="document-viewer-frame"
                />
              ) : null}

              {!fileViewer.isLoading &&
              !fileViewer.error &&
              isTextDocument(fileViewer.document, fileViewer.contentType) ? (
                <pre className="document-text-preview">
                  {fileViewer.textContent || "File không có nội dung."}
                </pre>
              ) : null}

              {!fileViewer.isLoading &&
              !fileViewer.error &&
              !isPdfDocument(fileViewer.document, fileViewer.contentType) &&
              !isTextDocument(fileViewer.document, fileViewer.contentType) ? (
                <div className="document-viewer-state">
                  Định dạng này không hỗ trợ xem trực tiếp trong trình duyệt.
                </div>
              ) : null}
            </div>
          </div>
        </div>
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
