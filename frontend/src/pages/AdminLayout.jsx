import { Link, NavLink, Outlet } from "react-router-dom";

function AdminLayout({ onLogout }) {
  return (
    <div className="admin-layout">
      <aside className="admin-sidebar">
        <div className="brand-wrap" style={{ padding: "1.5rem", borderBottom: "1px solid var(--border-light)", marginBottom: "1rem" }}>
          <h1 style={{ fontSize: "1.25rem", margin: 0 }}>
            <span className="brand-red">viettel</span>
            <span className="brand-black">Admin</span>
          </h1>
        </div>
        <nav className="admin-nav" style={{ display: "flex", flexDirection: "column", gap: "0.5rem", padding: "0 1rem" }}>
          <NavLink 
            to="/admin/documents" 
            className={({ isActive }) => `admin-nav-link ${isActive ? "active" : ""}`}
            style={({ isActive }) => ({
              padding: "0.75rem 1rem",
              borderRadius: "6px",
              textDecoration: "none",
              color: isActive ? "var(--viettel-red)" : "var(--text-main)",
              backgroundColor: isActive ? "var(--bg-light)" : "transparent",
              fontWeight: isActive ? "600" : "400"
            })}
          >
            Quản lí tài liệu
          </NavLink>
          <NavLink 
            to="/admin/users" 
            className={({ isActive }) => `admin-nav-link ${isActive ? "active" : ""}`}
            style={({ isActive }) => ({
              padding: "0.75rem 1rem",
              borderRadius: "6px",
              textDecoration: "none",
              color: isActive ? "var(--viettel-red)" : "var(--text-main)",
              backgroundColor: isActive ? "var(--bg-light)" : "transparent",
              fontWeight: isActive ? "600" : "400"
            })}
          >
            Quản lí người dùng
          </NavLink>
        </nav>
        <div style={{ marginTop: "auto", padding: "1.5rem" }}>
          <Link to="/chat" className="btn-outline ghost-link" style={{ width: "100%", marginBottom: "1rem", textAlign: "center", display: "block" }}>
            Quay lại Chat
          </Link>
          <button onClick={onLogout} className="btn-outline" style={{ width: "100%" }}>
            Đăng xuất
          </button>
        </div>
      </aside>
      
      <main className="admin-content" style={{ flex: 1, padding: "2rem", overflowY: "auto", backgroundColor: "var(--bg-main)" }}>
        <Outlet />
      </main>
      
      <style>{`
        .admin-layout {
          display: flex;
          height: 100vh;
          width: 100vw;
          overflow: hidden;
        }
        .admin-sidebar {
          width: 250px;
          flex-shrink: 0;
          background: #fff;
          border-right: 1px solid var(--border-light);
          display: flex;
          flex-direction: column;
        }
        .admin-nav-link:hover {
          background-color: var(--bg-light) !important;
        }
      `}</style>
    </div>
  );
}

export default AdminLayout;
