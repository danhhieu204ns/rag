import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api";

function DocumentsPage({ onAdminLogout }) {
  const [documents, setDocuments] = useState([]);
  const [titleDrafts, setTitleDrafts] = useState({});
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadTitle, setUploadTitle] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState("");

  const totalChunks = useMemo(
    () => documents.reduce((sum, item) => sum + (item.chunk_count || 0), 0),
    [documents]
  );

  async function fetchDocuments() {
    const response = await api.get("/documents");
    setDocuments(response.data);

    const nextDrafts = {};
    response.data.forEach((item) => {
      nextDrafts[item.id] = item.title;
    });
    setTitleDrafts(nextDrafts);
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!selectedFile) return;

    setIsBusy(true);
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
      setError(typeof detail === "string" ? detail : "Không thể upload tài liệu.");
    } finally {
      setIsBusy(false);
    }
  }

  async function saveTitle(documentId) {
    const title = (titleDrafts[documentId] || "").trim();
    if (!title) return;

    setIsBusy(true);
    setError("");
    try {
      await api.put(`/documents/${documentId}`, { title });
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể cập nhật tiêu đề.");
    } finally {
      setIsBusy(false);
    }
  }

  async function embedDocument(documentId) {
    setIsBusy(true);
    setError("");
    try {
      await api.post(`/documents/${documentId}/embed`);
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể embedding tài liệu.");
    } finally {
      setIsBusy(false);
    }
  }

  async function deleteDocument(documentId) {
    setIsBusy(true);
    setError("");
    try {
      await api.delete(`/documents/${documentId}`);
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể xóa tài liệu.");
    } finally {
      setIsBusy(false);
    }
  }

  async function rebuildIndex() {
    setIsBusy(true);
    setError("");
    try {
      await api.post("/documents/reindex");
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Không thể rebuild index.");
    } finally {
      setIsBusy(false);
    }
  }

  useEffect(() => {
    fetchDocuments().catch(() => setError("Không thể tải danh sách tài liệu."));
  }, []);

  return (
    <div className="admin-docs-shell">
      <header className="admin-docs-head">
        <div>
          <h1>Admin - Quan ly tai lieu</h1>
          <p>Upload, embedding va cap nhat kho tri thuc cho chatbot user.</p>
        </div>
        <div className="admin-docs-top-actions">
          <Link to="/chat" className="ghost-link">
            Mo user chat
          </Link>
          <button className="danger" onClick={onAdminLogout}>
            Dang xuat
          </button>
        </div>
      </header>

      <div className="panel panel-main admin-docs-panel">
        <div className="panel-head">
          <h2>Danh sach tai lieu</h2>
          <div className="panel-actions">
            <span className="muted">Tai lieu: {documents.length}</span>
            <span className="muted">Tong chunks: {totalChunks}</span>
            <button onClick={rebuildIndex} disabled={isBusy}>
            Rebuild index
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
            placeholder="Title (optional)"
            value={uploadTitle}
            onChange={(event) => setUploadTitle(event.target.value)}
          />
          <button type="submit" disabled={isBusy || !selectedFile}>
            {isBusy ? "Dang xu ly..." : "Upload"}
          </button>
        </form>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Tieu de</th>
                <th>File goc</th>
                <th>Status</th>
                <th>Chunks</th>
                <th>Hanh dong</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.id}</td>
                  <td>
                    <input
                      value={titleDrafts[doc.id] || ""}
                      onChange={(event) =>
                        setTitleDrafts((prev) => ({ ...prev, [doc.id]: event.target.value }))
                      }
                    />
                  </td>
                  <td>{doc.original_filename}</td>
                  <td>{doc.status}</td>
                  <td>{doc.chunk_count}</td>
                  <td className="row-actions">
                    <button onClick={() => saveTitle(doc.id)} disabled={isBusy}>
                      Save
                    </button>
                    <button onClick={() => embedDocument(doc.id)} disabled={isBusy}>
                      Embed
                    </button>
                    <button className="danger" onClick={() => deleteDocument(doc.id)} disabled={isBusy}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
      </div>
    </div>
  );
}

export default DocumentsPage;
