import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import api from "../api";

function LoginPage({ onLogin }) {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function submitLogin(event) {
    event.preventDefault();
    setError("");

    if (!username.trim() || !password.trim()) {
      setError("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await api.post("/auth/login", { username, password });
      const token = response.data.access_token;
      
      // Fetch user details to get role
      const userRes = await api.get("/auth/me", {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      onLogin(token, userRes.data);
      navigate("/chat", { replace: true });
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Đăng nhập thất bại. Vui lòng thử lại."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="admin-login-page">
      <div className="admin-bg-overlay" />

      <form className="admin-login-card" onSubmit={submitLogin}>
        <div className="brand-wrap">
          <h1>
            <span className="brand-red">viettel</span>
            <span className="brand-black">Chatbot</span>
          </h1>
          <h2>Đăng nhập Hệ thống</h2>
        </div>

        <label htmlFor="admin-user">Tên đăng nhập *</label>
        <input
          id="admin-user"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Nhập tên đăng nhập"
          autoComplete="username"
          disabled={isLoading}
        />

        <label htmlFor="admin-password">Mật khẩu *</label>
        <input
          id="admin-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Nhập mật khẩu"
          autoComplete="current-password"
          disabled={isLoading}
        />

        <div className="admin-login-actions">
          <button type="submit" className="btn-primary" disabled={isLoading}>
            {isLoading ? "Đang đăng nhập..." : "Đăng nhập"}
          </button>
        </div>

        {error && <p className="error-text">{error}</p>}
      </form>
    </div>
  );
}

export default LoginPage;
