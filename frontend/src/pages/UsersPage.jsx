import { useEffect, useState } from "react";
import api from "../api";

function UsersPage() {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("user");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function fetchUsers() {
    try {
      setIsLoading(true);
      const res = await api.get("/users");
      setUsers(res.data);
    } catch (err) {
      setError("Không thể tải danh sách người dùng.");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    fetchUsers();
  }, []);

  async function handleCreateUser(e) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.");
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      await api.post("/users", { username, password, role });
      setUsername("");
      setPassword("");
      setRole("user");
      await fetchUsers();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Lỗi khi tạo người dùng.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleDeleteUser(id) {
    if (!window.confirm("Bạn có chắc chắn muốn xoá người dùng này?")) return;
    setError("");
    try {
      await api.delete(`/users/${id}`);
      await fetchUsers();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Lỗi khi xoá người dùng.");
    }
  }

  return (
    <div className="panel panel-main admin-docs-panel">
      <div className="panel-head">
        <h2>Quản lí người dùng</h2>
        <div className="panel-actions">
          <span className="muted">Người dùng: {users.length}</span>
        </div>
      </div>

      <form className="upload-form" onSubmit={handleCreateUser} style={{ marginBottom: "2rem" }}>
        <input
          type="text"
          placeholder="Tên đăng nhập"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Mật khẩu"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <select value={role} onChange={(e) => setRole(e.target.value)} style={{ padding: "0.5rem", borderRadius: "6px", border: "1px solid var(--border-light)" }}>
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        <button type="submit" disabled={isSubmitting || !username || !password}>
          {isSubmitting ? "Đang tạo..." : "Tạo người dùng"}
        </button>
      </form>

      {error && <p className="error-text" style={{ marginBottom: "1rem" }}>{error}</p>}

      {isLoading ? (
        <p>Đang tải...</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Tên đăng nhập</th>
                <th>Role</th>
                <th>Trạng thái</th>
                <th>Hành động</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.id}</td>
                  <td>{u.username}</td>
                  <td>
                    <span style={{ 
                      padding: "2px 8px", 
                      borderRadius: "12px", 
                      fontSize: "0.8rem",
                      backgroundColor: u.role === "admin" ? "var(--viettel-red)" : "#e0e0e0",
                      color: u.role === "admin" ? "white" : "black"
                    }}>
                      {u.role}
                    </span>
                  </td>
                  <td>{u.is_active ? "Hoạt động" : "Bị khoá"}</td>
                  <td className="row-actions">
                    <button className="danger" onClick={() => handleDeleteUser(u.id)}>
                      Xoá
                    </button>
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan="5" style={{ textAlign: "center", padding: "1rem" }}>Không có người dùng nào.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default UsersPage;
