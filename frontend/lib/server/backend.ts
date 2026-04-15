function normalizeBaseUrl(url: string) {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

export function getBackendApiUrl() {
  const configuredUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (configuredUrl) {
    return normalizeBaseUrl(configuredUrl);
  }
  if (process.env.NODE_ENV !== "production") {
    return "http://localhost:8000";
  }
  throw new Error("BACKEND_API_URL is not configured for the frontend server.");
}
