import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://10.20.2.60:8000/api",
  timeout: 120000,
});

export default api;
