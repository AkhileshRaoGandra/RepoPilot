/**
 * Centralized API client for RepoPilot backend.
 * Base URL is read from the VITE_API_BASE_URL environment variable,
 * falling back to local FastAPI on http://127.0.0.1:8000.
 */

export const API_BASE_URL = (() => {
  if (import.meta.env.VITE_API_BASE_URL !== undefined && import.meta.env.VITE_API_BASE_URL !== "") {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, "");
  }
  // When deployed on Vercel or production domain, route requests through the /api rewrite
  if (
    typeof window !== "undefined" &&
    window.location &&
    window.location.hostname !== "localhost" &&
    window.location.hostname !== "127.0.0.1"
  ) {
    return "/api";
  }
  // Default fallback for standalone local FastAPI server
  return "http://127.0.0.1:8000";
})();

/**
 * Check if the FastAPI backend is running and reachable.
 * Uses an 8-second timeout to accommodate serverless (e.g. Vercel) cold starts,
 * and tries the lightweight /health endpoint before falling back to /openapi.json.
 */
export async function checkBackendHealth() {
  const endpoints = [
    `${API_BASE_URL}/health`,
    `${API_BASE_URL}/openapi.json`,
  ];
  if (API_BASE_URL === "/api") {
    endpoints.push("/health");
    endpoints.push("/openapi.json");
  }

  for (const endpoint of endpoints) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);
      const res = await fetch(endpoint, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        return true;
      }
    } catch {
      // Continue to next fallback endpoint
    }
  }
  return false;
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
 * Supports async background task queuing with polling to eliminate HTTP gateway timeouts.
 *
 * @param {string} githubUrl - GitHub repository URL
 * @param {(progress: { progress: number, step: string }) => void} [onProgress] - Optional progress callback
 * @returns {Promise<object>} The final repository analysis result
 */
export async function analyzeRepository(githubUrl, onProgress) {
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

  // If the server answered synchronously (e.g. sync=True mode), return the payload directly
  if (!payload || !payload.task_id) {
    return payload;
  }

  // Asynchronous background task: Poll GET /repository/tasks/{task_id} until completed or failed
  const taskId = payload.task_id;
  const pollIntervalMs = 1500;
  const maxAttempts = 240; // ~6 minutes maximum polling window
  let attempts = 0;

  if (onProgress) {
    onProgress({
      progress: payload.progress || 0,
      step: payload.step || "Queued for processing...",
    });
  }

  while (attempts < maxAttempts) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    attempts++;

    try {
      const taskRes = await fetch(`${API_BASE_URL}/repository/tasks/${taskId}`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      if (!taskRes.ok) {
        const errorData = await taskRes.json().catch(() => null);
        throw new Error(formatErrorMessage(null, errorData) || `Failed to fetch task status (${taskRes.status})`);
      }

      const taskData = await taskRes.json();

      if (onProgress) {
        onProgress({
          progress: taskData.progress ?? 0,
          step: taskData.step || "Processing repository...",
        });
      }

      if (taskData.status === "completed") {
        return taskData.result;
      }

      if (taskData.status === "failed") {
        throw new Error(taskData.error || "Repository analysis failed.");
      }
    } catch (pollErr) {
      // Re-throw genuine domain failures immediately
      if (pollErr.message && !pollErr.message.includes("Failed to fetch") && !pollErr.message.includes("NetworkError")) {
        throw pollErr;
      }
      // If a transient network glitch happens while polling, keep polling
    }
  }

  throw new Error("Repository analysis timed out after 6 minutes. Please try again.");
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
