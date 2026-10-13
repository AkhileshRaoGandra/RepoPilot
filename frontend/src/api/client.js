/**
 * Centralized API client for RepoPilot backend.
 * Base URL is read from the VITE_API_BASE_URL environment variable,
 * falling back to local FastAPI on http://127.0.0.1:8000.
 */

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/**
 * Check if the FastAPI backend is running and reachable.
 */
export async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${API_BASE_URL}/openapi.json`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Validate that a URL matches a valid public GitHub repository format:
 * https://github.com/:owner/:repo
 */
export function isValidGitHubUrl(url) {
  if (!url || typeof url !== "string") return false;
  const trimmed = url.trim();
  const pattern = /^https:\/\/github\.com\/[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+(?:\.git)?\/?$/;
  return pattern.test(trimmed);
}

/**
 * Format error message from API response or thrown Error.
 * Handles FastAPI string detail, Pydantic validation array detail, or network errors.
 */
export function formatErrorMessage(err, payload = null) {
  if (payload && payload.detail) {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((item) => (item.msg ? `${item.loc?.join(".") || "field"}: ${item.msg}` : JSON.stringify(item)))
        .join("; ");
    }
  }
  if (err && err.message) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      return `Unable to reach the RepoPilot API at ${API_BASE_URL}. Ensure the FastAPI backend is running.`;
    }
    return err.message;
  }
  return "An unexpected error occurred while contacting the server.";
}

/**
 * Ingest and analyze a public GitHub repository.
 * Endpoint: POST /repository/analyze
 */
export async function analyzeRepository(githubUrl) {
  const trimmed = (githubUrl || "").trim();
  if (!trimmed) {
    throw new Error("Please enter a GitHub repository URL.");
  }
  if (!isValidGitHubUrl(trimmed)) {
    throw new Error("Please enter a valid HTTPS GitHub repository URL (e.g. https://github.com/owner/repository).");
  }

  let response;
  let payload;
  try {
    response = await fetch(`${API_BASE_URL}/repository/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ github_url: trimmed }),
    });
    payload = await response.json();
  } catch (err) {
    throw new Error(formatErrorMessage(err));
  }

  if (!response.ok) {
    throw new Error(formatErrorMessage(null, payload));
  }
  return payload;
}

/**
 * Run deterministic standard RAG query.
 * Endpoint: POST /query
 */
export async function queryRAG(query, topK = 5) {
  const trimmed = (query || "").trim();
  if (!trimmed) {
    throw new Error("Please enter a non-empty question.");
  }

  let response;
  let payload;
  try {
    response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed, top_k: Number(topK) }),
    });
    payload = await response.json();
  } catch (err) {
    throw new Error(formatErrorMessage(err));
  }

  if (!response.ok) {
    throw new Error(formatErrorMessage(null, payload));
  }
  return payload;
}

/**
 * Run agentic repository investigation using LangGraph with graph tool & source inspection.
 * Endpoint: POST /agent/query
 */
export async function queryAgent(query, topK = 5) {
  const trimmed = (query || "").trim();
  if (!trimmed) {
    throw new Error("Please enter a non-empty question.");
  }

  let response;
  let payload;
  try {
    response = await fetch(`${API_BASE_URL}/agent/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed, top_k: Number(topK) }),
    });
    payload = await response.json();
  } catch (err) {
    throw new Error(formatErrorMessage(err));
  }

  if (!response.ok) {
    throw new Error(formatErrorMessage(null, payload));
  }
  return payload;
}

/**
 * Directly retrieve top semantic chunks without LLM generation.
 * Endpoint: POST /retrieve
 */
export async function retrieveChunks(query, topK = 5) {
  const trimmed = (query || "").trim();
  if (!trimmed) {
    throw new Error("Please enter a non-empty query.");
  }

  let response;
  let payload;
  try {
    response = await fetch(`${API_BASE_URL}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed, top_k: Number(topK) }),
    });
    payload = await response.json();
  } catch (err) {
    throw new Error(formatErrorMessage(err));
  }

  if (!response.ok) {
    throw new Error(formatErrorMessage(null, payload));
  }
  return payload;
}

/**
 * Retrieve the current repository dependency graph nodes and edges.
 * Endpoint: GET /repository/graph
 */
export async function getRepositoryGraph() {
  let response;
  let payload;
  try {
    response = await fetch(`${API_BASE_URL}/repository/graph`);
    payload = await response.json();
  } catch (err) {
    throw new Error(formatErrorMessage(err));
  }

  if (!response.ok) {
    throw new Error(formatErrorMessage(null, payload));
  }
  return payload;
}
