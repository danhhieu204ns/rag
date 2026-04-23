import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api";

const CHUNK_PAGE_SIZE = 10;
const DOCUMENT_PROGRESS_POLL_INTERVAL_MS = 2500;

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [titleDrafts, setTitleDrafts] = useState({});
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

  async function fetchDocuments(options = {}) {
    const { syncDrafts = true } = options;
    const response = await api.get("/documents");
    setDocuments(response.data);

    if (syncDrafts) {
      const nextDrafts = {};
      response.data.forEach((item) => {
        nextDrafts[item.id] = item.title;
      });
      setTitleDrafts(nextDrafts);
    } else {
      setTitleDrafts((prev) => {
        const merged = { ...prev };
        const aliveIds = new Set(response.data.map((item) => item.id));

        response.data.forEach((item) => {
          if (merged[item.id] === undefined) {
            merged[item.id] = item.title;
          }
        });

        Object.keys(merged).forEach((key) => {
          const id = Number(key);
          if (!aliveIds.has(id)) {
            delete merged[key];
          }
        });

        return merged;
      });
    }

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
      setChunksError(typeof detail === "string" ? detail : "Khong the tai danh sach chunks.");
    } finally {
      setIsChunksBusy(false);
    }
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!selectedFile) return;

    setIsBusy(true);
    setBusyMessage("Dang upload tai lieu...");
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
      setError(typeof detail === "string" ? detail : "Khong the upload tai lieu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  async function saveTitle(documentId) {
    const title = (titleDrafts[documentId] || "").trim();
    if (!title) return;

    setIsBusy(true);
    setBusyMessage("Dang cap nhat tieu de...");
    setError("");
    try {
      await api.put(`/documents/${documentId}`, { title });
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Khong the cap nhat tieu de.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  async function parseDocument(documentId) {
    setIsBusy(true);
    setBusyMessage("Buoc A: Dang parse PDF sang markdown va luu disk...");
    setActiveDocumentAction({ documentId, action: "parse" });
    setError("");
    try {
      await api.post(`/documents/${documentId}/parse`);
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Khong the parse tai lieu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
      setActiveDocumentAction({ documentId: null, action: "" });
    }
  }

  async function embedDocument(documentId) {
    setIsBusy(true);
    setBusyMessage("Buoc B: Dang load markdown da parse de chunking + embedding...");
    setActiveDocumentAction({ documentId, action: "embed" });
    setError("");
    try {
      await api.post(`/documents/${documentId}/embed`);
      setBusyMessage("Buoc B: Da queue embedding/upsert, dang dong bo giao dien...");
      await fetchDocuments();
      if (selectedChunkDocumentId === documentId) {
        await fetchDocumentChunks(documentId, 0);
      }
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Khong the embed tai lieu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
      setActiveDocumentAction({ documentId: null, action: "" });
    }
  }

  async function deleteDocument(documentId) {
    setIsBusy(true);
    setBusyMessage("Dang xoa tai lieu...");
    setError("");
    try {
      await api.delete(`/documents/${documentId}`);
      if (selectedChunkDocumentId === documentId) {
        closeChunksInspector();
      }
      await fetchDocuments();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Khong the xoa tai lieu.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
  }

  async function rebuildIndex() {
    setIsBusy(true);
    setBusyMessage("Dang queue rebuild index tu markdown da parse...");
    setError("");
    try {
      await api.post("/documents/reindex");
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Khong the rebuild index.");
    } finally {
      setIsBusy(false);
      setBusyMessage("");
    }
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

  useEffect(() => {
    fetchDocuments().catch(() => setError("Khong the tai danh sach tai lieu."));
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
        await fetchDocuments({ syncDrafts: false });
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
          <h2>Danh sach tai lieu</h2>
          <div className="panel-actions">
            <span className="muted">Tai lieu: {documents.length}</span>
            <span className="muted">Tong chunks: {totalChunks}</span>
            {hasIndexingDocuments ? <span className="muted">Dang polling tien trinh...</span> : null}
            <button onClick={rebuildIndex} disabled={isBusy}>
              Rebuild index
            </button>
          </div>
        </div>

        <p className="muted step-hint">
          Luong 2 buoc: Buoc A Parse 1 lan (PDF -&gt; markdown, luu disk) -&gt; Buoc B tai su dung markdown de chunking,
          parent-child, summary/questions/keywords, embedding va upsert.
        </p>

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
                <th>Buoc A</th>
                <th>Buoc B</th>
                <th>Khac</th>
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
                    <button onClick={() => parseDocument(doc.id)} disabled={isBusy}>
                      {isBusy && activeDocumentAction.documentId === doc.id && activeDocumentAction.action === "parse"
                        ? "Dang parse..."
                        : "Parse markdown"}
                    </button>
                  </td>
                  <td className="row-actions">
                    <button onClick={() => embedDocument(doc.id)} disabled={isBusy}>
                      {isBusy && activeDocumentAction.documentId === doc.id && activeDocumentAction.action === "embed"
                        ? "Dang load markdown..."
                        : "Embed tu markdown"}
                    </button>
                  </td>
                  <td className="row-actions">
                    <button onClick={() => handleChunkToggle(doc)} disabled={isBusy || isChunksBusy}>
                      {selectedChunkDocumentId === doc.id ? "Hide chunks" : "View chunks"}
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

        {selectedChunkDocumentId ? (
          <section className="chunk-inspector">
            <div className="chunk-inspector-head">
              <div>
                <h3>
                  Chunks - Document #{selectedChunkDocumentId}
                  {selectedChunkDocument ? `: ${selectedChunkDocument.title}` : ""}
                </h3>
                <p className="muted">
                  Total: {chunkTotal}
                  {chunkTotal > 0 ? ` | Showing ${chunkRangeStart}-${chunkRangeEnd}` : ""}
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
                  Prev
                </button>
                <button
                  onClick={() =>
                    fetchDocumentChunks(selectedChunkDocumentId, chunkOffset + CHUNK_PAGE_SIZE)
                  }
                  disabled={!canGoChunkNext || isChunksBusy}
                >
                  Next
                </button>
                <button className="danger" onClick={closeChunksInspector}>
                  Close
                </button>
              </div>
            </div>

            {isChunksBusy ? <p className="muted">Dang tai chunks...</p> : null}
            {chunksError ? <p className="error-text">{chunksError}</p> : null}
            {!isChunksBusy && !chunksError && chunks.length === 0 ? (
              <p className="muted">Tai lieu nay chua co chunk. Bam Embed de tao chunks.</p>
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
                      {chunk.source_page ? <span className="chunk-pill">page {chunk.source_page}</span> : null}
                      {chunk.source_kind ? <span className="chunk-pill">{chunk.source_kind}</span> : null}
                      {chunk.created_at ? (
                        <span className="chunk-pill">{formatChunkDate(chunk.created_at)}</span>
                      ) : null}
                    </div>

                    <details open>
                      <summary>Content</summary>
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
      </div>
  );
}

export default DocumentsPage;
