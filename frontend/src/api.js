import axios from "axios";

const TOKEN_KEY = "rag_admin_token";

function resolveApiBaseUrl() {
  const raw = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api").trim();
  const noTrailingSlash = raw.replace(/\/+$/, "");
  return noTrailingSlash.endsWith("/api") ? noTrailingSlash : `${noTrailingSlash}/api`;
}

const api = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: 120000,
});

console.log("API base URL:", api.defaults.baseURL);

// Attach token to every request if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// On 401, clear token and redirect to admin login
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const isAdminRoute = window.location.pathname.startsWith("/admin");
      if (isAdminRoute) {
        localStorage.removeItem(TOKEN_KEY);
        window.location.href = "/admin/login";
      }
    }
    return Promise.reject(error);
  }
);

export { TOKEN_KEY };
export default api;
