export const API_BASE_URL = String(
  import.meta.env.VITE_API_BASE_URL ||
    "https://forensight-criminal-face-generation-and.onrender.com",
).replace(/\/+$/, "");
