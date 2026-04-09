import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import DocumentsPage from "./pages/DocumentsPage";

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>RAG Assistant</h1>
          <p>FastAPI + React application</p>
        </div>
        <nav className="tabs">
          <NavLink to="/chat" className={({ isActive }) => (isActive ? "tab active" : "tab")}>
            Hỏi đáp
          </NavLink>
          <NavLink to="/documents" className={({ isActive }) => (isActive ? "tab active" : "tab")}>
            Quản lý tài liệu
          </NavLink>
        </nav>
      </header>

      <main className="page-content">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
