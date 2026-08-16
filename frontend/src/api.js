const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const message =
      payload?.detail ||
      payload?.message ||
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  return payload;
}

export function fetchSources() {
  return request("/api/sources");
}

export function indexSources(urls) {
  return request("/api/sources/index", {
    method: "POST",
    body: JSON.stringify({ urls }),
  });
}

export function sendQuestion(question, activeUrls) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, activeUrls }),
  });
}

