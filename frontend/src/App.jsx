import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import AdminLoginPage from "./pages/AdminLoginPage";
import ChatPage from "./pages/ChatPage";
import DocumentsPage from "./pages/DocumentsPage";
import api, { TOKEN_KEY } from "./api";

function App() {
  const [isAdminLoggedIn, setIsAdminLoggedIn] = useState(
    () => Boolean(localStorage.getItem(TOKEN_KEY))
  );

  // Verify token validity once on mount
  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setIsAdminLoggedIn(false);
      return;
    }

    api
      .get("/auth/me")
      .then(() => setIsAdminLoggedIn(true))
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        setIsAdminLoggedIn(false);
      });
  }, []);

  function handleAdminLogin(token) {
    localStorage.setItem(TOKEN_KEY, token);
    setIsAdminLoggedIn(true);
  }

  function handleAdminLogout() {
    localStorage.removeItem(TOKEN_KEY);
    setIsAdminLoggedIn(false);
  }

  return (
    <Routes>
      <Route path="/" element={<Navigate to="/chat" replace />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route
        path="/admin/login"
        element={
          isAdminLoggedIn ? (
            <Navigate to="/admin/documents" replace />
          ) : (
            <AdminLoginPage onLogin={handleAdminLogin} />
          )
        }
      />
      <Route
        path="/admin/documents"
        element={
          isAdminLoggedIn ? (
            <DocumentsPage onAdminLogout={handleAdminLogout} />
          ) : (
            <Navigate to="/admin/login" replace />
          )
        }
      />
      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}

export default App;
