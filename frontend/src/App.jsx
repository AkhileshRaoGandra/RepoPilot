import React, { useState, useEffect, useMemo } from "react";
import {
  analyzeRepository,
  queryRAG,
  queryAgent,
  isValidGitHubUrl,
  checkBackendHealth,
  API_BASE_URL,
} from "./api/client";

// Popular repository presets for quick testing
const PRESET_REPOS = [
  { label: "fastapi/fastapi", url: "https://github.com/fastapi/fastapi", desc: "Python API Framework" },
  { label: "pallets/flask", url: "https://github.com/pallets/flask", desc: "Python Microframework" },
  { label: "psf/requests", url: "https://github.com/psf/requests", desc: "HTTP Library" },
];

// Curated question suggestions with categories
const SUGGESTIONS = [
  {
    category: "Architecture",
    icon: "🏗️",
    title: "Layer Flow",
    question: "How does a request move from the controller to the repository layer?",
  },
  {
    category: "Authentication",
    icon: "🔐",
    title: "Auth & Security",
    question: "How does authentication work in this codebase?",
  },
  {
    category: "Database",
    icon: "🗄️",
    title: "Data Persistence",
    question: "Where is the database connection handled and how are queries executed?",
  },
  {
    category: "Dependencies",
    icon: "🕸️",
    title: "Module Imports",
    question: "What are the core service dependencies and how do they interact?",
  },
];

// Lightweight Markdown Renderer (Zero external dependencies)
function FormattedAnswer({ text }) {
  const parts = useMemo(() => {
    if (!text) return [];
    // Split by code blocks first
    const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
    const segments = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        segments.push({
          type: "text",
          content: text.slice(lastIndex, match.index),
        });
      }
      segments.push({
        type: "code",
        language: match[1] || "code",
        content: match[2].trim(),
      });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      segments.push({
        type: "text",
        content: text.slice(lastIndex),
      });
    }

    return segments;
  }, [text]);

  const [copiedIndex, setCopiedIndex] = useState(null);

  function copyCode(content, idx) {
    navigator.clipboard.writeText(content);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  }

  // Format regular markdown text (headers, bold, inline code, list items)
  function renderTextBlock(content, segIdx) {
    const lines = content.split("\n");
    return lines.map((line, lIdx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return <div key={`${segIdx}-${lIdx}`} className="text-spacer" />;
      }

      // Headers
      if (trimmed.startsWith("### ")) {
        return (
          <h4 key={`${segIdx}-${lIdx}`} className="answer-h4">
            {formatInlineText(trimmed.slice(4))}
          </h4>
        );
      }
      if (trimmed.startsWith("## ")) {
        return (
          <h3 key={`${segIdx}-${lIdx}`} className="answer-h3">
            {formatInlineText(trimmed.slice(3))}
          </h3>
        );
      }

      // Bullet items
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        return (
          <div key={`${segIdx}-${lIdx}`} className="answer-list-item">
            <span className="bullet-point">▸</span>
            <span>{formatInlineText(trimmed.slice(2))}</span>
          </div>
        );
      }

      // Numbered items
      const numMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
      if (numMatch) {
        return (
          <div key={`${segIdx}-${lIdx}`} className="answer-list-item numbered">
            <span className="num-point">{numMatch[1]}.</span>
            <span>{formatInlineText(numMatch[2])}</span>
          </div>
        );
      }

      // Regular paragraph
      return (
        <p key={`${segIdx}-${lIdx}`} className="answer-p">
          {formatInlineText(line)}
        </p>
      );
    });
  }

  // Parse inline `code` and **bold**
  function formatInlineText(raw) {
    const tokens = [];
    const inlineRegex = /(`[^`]+`|\*\*[^*]+\*\*)/g;
    let last = 0;
    let m;

    while ((m = inlineRegex.exec(raw)) !== null) {
      if (m.index > last) {
        tokens.push(raw.slice(last, m.index));
      }
      const tok = m[1];
      if (tok.startsWith("`") && tok.endsWith("`")) {
        tokens.push(
          <code key={tokens.length} className="inline-code-token">
            {tok.slice(1, -1)}
          </code>
        );
      } else if (tok.startsWith("**") && tok.endsWith("**")) {
        tokens.push(
          <strong key={tokens.length} className="inline-bold-token">
            {tok.slice(2, -2)}
          </strong>
        );
      }
      last = m.index + tok.length;
    }

    if (last < raw.length) {
      tokens.push(raw.slice(last));
    }

    return tokens;
  }

  return (
    <div className="formatted-answer-container">
      {parts.map((segment, idx) => {
        if (segment.type === "code") {
          return (
            <div key={idx} className="answer-code-block-wrapper">
              <div className="code-block-header">
                <span className="code-block-lang">{segment.language}</span>
                <button
                  type="button"
                  className="code-block-copy-btn"
                  onClick={() => copyCode(segment.content, idx)}
                >
                  {copiedIndex === idx ? "Copied ✓" : "Copy"}
                </button>
              </div>
              <pre className="code-block-pre">
                <code>{segment.content}</code>
              </pre>
            </div>
          );
        }
        return <div key={idx}>{renderTextBlock(segment.content, idx)}</div>;
      })}
    </div>
  );
}

// Language color mapping for tags
const EXT_COLOR_MAP = {
  PY: { bg: "rgba(59, 130, 246, 0.15)", text: "#60a5fa", border: "rgba(59, 130, 246, 0.3)" },
  JS: { bg: "rgba(234, 179, 8, 0.15)", text: "#facc15", border: "rgba(234, 179, 8, 0.3)" },
  TS: { bg: "rgba(14, 165, 233, 0.15)", text: "#38bdf8", border: "rgba(14, 165, 233, 0.3)" },
  JAVA: { bg: "rgba(239, 68, 68, 0.15)", text: "#f87171", border: "rgba(239, 68, 68, 0.3)" },
  CPP: { bg: "rgba(236, 72, 153, 0.15)", text: "#f472b6", border: "rgba(236, 72, 153, 0.3)" },
  C: { bg: "rgba(236, 72, 153, 0.15)", text: "#f472b6", border: "rgba(236, 72, 153, 0.3)" },
  H: { bg: "rgba(236, 72, 153, 0.15)", text: "#f472b6", border: "rgba(236, 72, 153, 0.3)" },
  HPP: { bg: "rgba(236, 72, 153, 0.15)", text: "#f472b6", border: "rgba(236, 72, 153, 0.3)" },
  JSON: { bg: "rgba(168, 85, 247, 0.15)", text: "#c084fc", border: "rgba(168, 85, 247, 0.3)" },
  TXT: { bg: "rgba(161, 161, 170, 0.15)", text: "#e4e4e7", border: "rgba(161, 161, 170, 0.3)" },
  MD: { bg: "rgba(16, 185, 129, 0.15)", text: "#34d399", border: "rgba(16, 185, 129, 0.3)" },
  YAML: { bg: "rgba(245, 158, 11, 0.15)", text: "#fbbf24", border: "rgba(245, 158, 11, 0.3)" },
  YML: { bg: "rgba(245, 158, 11, 0.15)", text: "#fbbf24", border: "rgba(245, 158, 11, 0.3)" },
  GO: { bg: "rgba(20, 184, 166, 0.15)", text: "#2dd4bf", border: "rgba(20, 184, 166, 0.3)" },
};

function FileBadge({ filepath }) {
  const ext = (filepath ? filepath.split(".").pop() : "FILE").toUpperCase();
  const theme = EXT_COLOR_MAP[ext] || {
    bg: "rgba(255, 255, 255, 0.08)",
    text: "#d4d4d8",
    border: "rgba(255, 255, 255, 0.12)",
  };

  return (
    <span
      className="file-extension-badge"
      style={{
        backgroundColor: theme.bg,
        color: theme.text,
        borderColor: theme.border,
      }}
    >
      {ext}
    </span>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("UI rendering error caught by ErrorBoundary:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "40px", color: "#f87171", background: "#0b0f19", minHeight: "100vh", fontFamily: "sans-serif" }}>
          <h2>Application Rendering Error</h2>
          <p>{this.state.error?.message || "An unexpected error occurred."}</p>
          <button
            onClick={() => window.location.reload()}
            style={{ padding: "8px 16px", marginTop: "12px", background: "#3b82f6", color: "white", border: "none", borderRadius: "6px", cursor: "pointer" }}
          >
            Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function MainApp() {
  // Backend health status
  const [backendOnline, setBackendOnline] = useState(null);

  // Step 1: Repository Ingestion State
  const [githubUrl, setGithubUrl] = useState("");
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState({ progress: 0, step: "" });
  const [analyzeError, setAnalyzeError] = useState("");
  const [indexedRepo, setIndexedRepo] = useState(null);

  // Question & Query State
  const [question, setQuestion] = useState("How does authentication work?");
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState("");
  const [queryResult, setQueryResult] = useState(null);

  // UI state for excerpts and traces
  const [expandedExcerptIndex, setExpandedExcerptIndex] = useState(null);
  const [expandAllExcerpts, setExpandAllExcerpts] = useState(false);
  const [traceExpanded, setTraceExpanded] = useState(true);
  const [copiedAnswer, setCopiedAnswer] = useState(false);
  const [copiedSnippetIndex, setCopiedSnippetIndex] = useState(null);

  // Check backend health periodically
  useEffect(() => {
    let active = true;
    async function ping() {
      const ok = await checkBackendHealth();
      if (active) setBackendOnline(ok);
    }
    ping();
    const interval = setInterval(ping, 15000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // Handle repository analysis
  async function handleAnalyzeSubmit(e) {
    if (e) e.preventDefault();
    setAnalyzeError("");

    const trimmed = (githubUrl || "").trim();
    if (!trimmed) {
      setAnalyzeError("Please enter a GitHub repository URL.");
      return;
    }
    if (!isValidGitHubUrl(trimmed)) {
      setAnalyzeError(
        "Please provide a valid HTTPS GitHub repository URL (e.g. https://github.com/owner/repository)."
      );
      return;
    }

    setAnalyzeLoading(true);
    setIndexedRepo(null);
    setAnalyzeProgress({ progress: 5, step: "Queued for processing..." });

    try {
      const data = await analyzeRepository(trimmed, (p) => setAnalyzeProgress(p));
      setIndexedRepo(data);
    } catch (err) {
      setAnalyzeError(err.message || "Failed to analyze repository.");
    } finally {
      setAnalyzeLoading(false);
    }
  }

  // Handle asking questions
  async function handleQuerySubmit(e) {
    if (e) e.preventDefault();
    setQueryError("");

    // Validation 1: Require repository ingestion first
    if (!indexedRepo) {
      setQueryError(
        "Please ingest a repository first! Enter a GitHub repository URL above and click 'Analyze' before asking questions."
      );
      const repoInput = document.getElementById("github-repo-input");
      if (repoInput) {
        repoInput.scrollIntoView({ behavior: "smooth", block: "center" });
        repoInput.focus();
      }
      return;
    }

    // Validation 2: Require non-empty question
    const trimmed = (question || "").trim();
    if (!trimmed) {
      setQueryError("Please enter a question to ask.");
      return;
    }

    setQueryLoading(true);
    setQueryResult(null);
    setExpandedExcerptIndex(null);
    setExpandAllExcerpts(false);

    try {
      const data = await queryAgent(trimmed, 5);
      setQueryResult(data);
    } catch (err) {
      const msg = (err && err.message) || "";
      if (
        msg.toLowerCase().includes("ingest") ||
        msg.toLowerCase().includes("not found") ||
        msg.toLowerCase().includes("no dependency graph") ||
        msg.toLowerCase().includes("collection")
      ) {
        setQueryError(
          "Please ingest a repository first. No indexed repository is currently active on the server. Analyze a repository above to index the codebase."
        );
      } else {
        setQueryError(err.message || "Failed to execute query.");
      }
    } finally {
      setQueryLoading(false);
    }
  }

  function handleCopyAnswer() {
    if (!queryResult?.answer) return;
    navigator.clipboard.writeText(queryResult.answer);
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 2000);
  }

  function handleCopySnippet(content, idx) {
    navigator.clipboard.writeText(content);
    setCopiedSnippetIndex(idx);
    setTimeout(() => setCopiedSnippetIndex(null), 2000);
  }

  return (
    <div className="app-root-layout">
      {/* Background Ambience */}
      <div className="bg-aurora-glow" aria-hidden="true" />
      <div className="bg-grid-overlay" aria-hidden="true" />

      {/* Main Structural Card Container */}
      <main className="repopilot-unified-card">
        {/* ====================================================================
            PAGE HEADER
            ==================================================================== */}
        <header className="app-header-bar">
          <div className="header-brand-cluster">
            <div className="brand-logo-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="18" cy="18" r="3" />
                <circle cx="6" cy="6" r="3" />
                <circle cx="18" cy="6" r="3" />
                <path d="M6 9v12" />
                <path d="M18 9a9 9 0 0 1-9 9" />
              </svg>
            </div>
            <div className="brand-text-cluster">
              <div className="brand-title-line">
                <h1>RepoPilot</h1>
                <span className="brand-chip">v0.8 · Agentic RAG</span>
              </div>
              <p className="brand-subtitle">
                Onboard & navigate any GitHub repository through semantic vector retrieval & autonomous agent reasoning.
              </p>
            </div>
          </div>

          <div className="header-controls-cluster">
            <div
              className={`backend-status-pill ${
                backendOnline === true
                  ? "is-online"
                  : backendOnline === false
                  ? "is-offline"
                  : "is-checking"
              }`}
              title={`API endpoint: ${API_BASE_URL}`}
            >
              <span className="pulse-indicator-dot" />
              <span>
                {backendOnline === true
                  ? "Backend Online"
                  : backendOnline === false
                  ? "Offline"
                  : "Connecting…"}
              </span>
            </div>
          </div>
        </header>

        {/* ====================================================================
            ANALYZE REPOSITORY
            ==================================================================== */}
        <section className="structural-step-container" aria-labelledby="step-analyze-title">
          <div className="step-title-row">
            <div className="step-number-title">
              <div className="step-icon-badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
                  <path d="M9 18c-4.51 2-5-2-7-2" />
                </svg>
              </div>
              <h2 id="step-analyze-title" className="step-heading">Analyze repository</h2>
            </div>
            {indexedRepo && (
              <span className="step-status-chip ready">
                <span className="check-icon">✓</span> Ready for queries
              </span>
            )}
          </div>

          <form onSubmit={handleAnalyzeSubmit}>
            <div className={`step-input-wrapper ${analyzeError ? "has-input-error" : ""}`}>
              <div className="input-field-prefix">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
                </svg>
              </div>

              <input
                id="github-repo-input"
                type="text"
                className="step-text-input"
                placeholder="https://github.com/owner/repository"
                value={githubUrl}
                disabled={analyzeLoading}
                onChange={(e) => {
                  setGithubUrl(e.target.value);
                  if (analyzeError) setAnalyzeError("");
                }}
              />

              {githubUrl && !analyzeLoading && (
                <button
                  type="button"
                  className="input-clear-btn"
                  title="Clear input"
                  onClick={() => {
                    setGithubUrl("");
                    setIndexedRepo(null);
                  }}
                >
                  ✕
                </button>
              )}

              <button
                type="submit"
                className="step-submit-btn primary"
                disabled={analyzeLoading}
              >
                {analyzeLoading ? (
                  <>
                    <span className="spinner-icon" />
                    <span>Analyzing…</span>
                  </>
                ) : (
                  <>
                    <span>Analyze</span>
                    <span className="arrow-icon">↗</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Quick preset chips */}
          <div className="preset-suggestions-bar">
            <span className="preset-label">Preset repositories:</span>
            {PRESET_REPOS.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                className="preset-tag-button"
                disabled={analyzeLoading}
                onClick={() => {
                  setGithubUrl(preset.url);
                  if (analyzeError) setAnalyzeError("");
                }}
                title={preset.desc}
              >
                <code>{preset.label}</code>
              </button>
            ))}
          </div>

          {/* Analysis loading state with real-time task progress */}
          {analyzeLoading && (
            <div className="processing-feedback-card ingestion-progress-card">
              <div className="feedback-spinner-orbit" />
              <div className="feedback-text-group" style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <strong>{analyzeProgress.step || "Cloning & Indexing Repository..."}</strong>
                  <span className="progress-percent-pill">{analyzeProgress.progress || 10}%</span>
                </div>
                <div className="ingestion-progress-track">
                  <div
                    className="ingestion-progress-fill"
                    style={{ width: `${Math.max(8, analyzeProgress.progress || 10)}%` }}
                  />
                </div>
                <p>
                  Running asynchronous background ingestion: Git clone, Tree-sitter AST symbol parsing, relationship graph generation, BGE-M3 embedding, and Qdrant vector indexing.
                </p>
              </div>
            </div>
          )}

          {/* Indexed repository details card */}
          {indexedRepo && !analyzeLoading && (
            <div className="indexed-overview-card">
              <div className="overview-header-row">
                <div className="overview-repo-badge">
                  <span className="repo-verified-icon">✔</span>
                  <div>
                    <span className="overview-title-text">Repository Indexed Successfully</span>
                    <div className="overview-repo-path">
                      <code>{indexedRepo.repository}</code>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  className="reindex-action-btn"
                  onClick={handleAnalyzeSubmit}
                  title="Re-run indexing"
                >
                  Re-index ↻
                </button>
              </div>

              <div className="overview-metrics-grid">
                <div className="metric-cell">
                  <span className="metric-label">Files Discovered</span>
                  <span className="metric-value">{indexedRepo.files_found || 0}</span>
                </div>
                <div className="metric-cell">
                  <span className="metric-label">AST Chunks Stored</span>
                  <span className="metric-value">{indexedRepo.chunks_stored || 0}</span>
                </div>
                <div className="metric-cell">
                  <span className="metric-label">Vector Store</span>
                  <span className="metric-value">Qdrant (BGE-M3)</span>
                </div>
                <div className="metric-cell">
                  <span className="metric-label">Dependency Graph</span>
                  <span className="metric-value accent">Active & Linked</span>
                </div>
              </div>
            </div>
          )}

          {/* Step 1 error */}
          {analyzeError && (
            <div className="step-error-banner" role="alert">
              <span className="error-icon">⚠️</span>
              <span>{analyzeError}</span>
            </div>
          )}
        </section>

        {/* ====================================================================
            ASK A QUESTION (LANGGRAPH AGENT)
            ==================================================================== */}
        <section className="structural-step-container" aria-labelledby="step-query-title">
          <div className="step-title-row">
            <div className="step-number-title">
              <div className="step-icon-badge accent">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
              </div>
              <h2 id="step-query-title" className="step-heading">Ask a question</h2>
            </div>
            <div className="step-header-meta-group">
              {!indexedRepo ? (
                <span className="step-ingestion-warn-badge">
                  ⚠️ Ingestion Required
                </span>
              ) : (
                <span className="step-ingestion-ready-badge">
                  ✓ Ready ({indexedRepo.repository})
                </span>
              )}
              <span className="step-route-hint">
                Autonomous Agent · LangGraph
              </span>
            </div>
          </div>

          <form onSubmit={handleQuerySubmit}>
            <div className={`step-input-wrapper ${queryError ? "has-input-error" : ""}`}>
              <div className="input-field-prefix prompt-arrow">›</div>

              <input
                id="question-input"
                type="text"
                className="step-text-input"
                placeholder={
                  !indexedRepo
                    ? "Please ingest a repository above first..."
                    : "Ask anything about the codebase (e.g. How does authentication work?)"
                }
                value={question}
                disabled={queryLoading}
                onChange={(e) => {
                  setQuestion(e.target.value);
                  if (queryError) setQueryError("");
                }}
              />

              {question && !queryLoading && (
                <button
                  type="button"
                  className="input-clear-btn"
                  title="Clear question"
                  onClick={() => setQuestion("")}
                >
                  ✕
                </button>
              )}

              <button
                type="submit"
                className="step-submit-btn accent"
                disabled={queryLoading}
                title={!indexedRepo ? "Please ingest a repository in Step 1 first" : "Submit question"}
              >
                {queryLoading ? (
                  <>
                    <span className="spinner-icon" />
                    <span>Reasoning…</span>
                  </>
                ) : (
                  <>
                    <span>Ask</span>
                    <span className="key-return-badge">↵</span>
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Suggested question chips */}
          <div className="question-suggestions-row">
            {SUGGESTIONS.map((item, idx) => (
              <button
                key={idx}
                type="button"
                className="suggestion-chip-btn"
                disabled={queryLoading}
                onClick={() => {
                  setQuestion(item.question);
                  if (queryError) setQueryError("");
                }}
              >
                <span className="chip-category-icon">{item.icon}</span>
                <span className="chip-title-text">{item.title}</span>
              </button>
            ))}
          </div>

          {/* Query in-flight feedback */}
          {queryLoading && (
            <div className="processing-feedback-card">
              <div className="feedback-spinner-orbit" />
              <div className="feedback-text-group">
                <strong>LangGraph Agent Executing Investigation Plan</strong>
                <p>
                  Executing dynamic tool calls across semantic vector space, repository graph nodes, and source AST code inspection.
                </p>
              </div>
            </div>
          )}

          {/* Query error banner with ingestion call-to-action */}
          {queryError && (
            <div
              className={`step-error-banner ${
                queryError.includes("ingest a repository first") ? "is-ingestion-required" : ""
              }`}
              role="alert"
            >
              <div className="error-message-content">
                <span className="error-icon">⚠️</span>
                <span>{queryError}</span>
              </div>
              {queryError.includes("ingest a repository first") && (
                <button
                  type="button"
                  className="jump-to-step1-action-btn"
                  onClick={() => {
                    const el = document.getElementById("github-repo-input");
                    if (el) {
                      el.scrollIntoView({ behavior: "smooth", block: "center" });
                      el.focus();
                    }
                  }}
                >
                  Go to Step 1 ↗
                </button>
              )}
            </div>
          )}
        </section>

        {/* ====================================================================
            RESULTS: ANSWER & EVIDENCE (MATCHING PIC'S STRUCTURE)
            ==================================================================== */}
        {queryResult ? (
          <>
            <div className="structural-section-divider" />

            {/* Answer Section */}
            <section className="structural-output-section" aria-labelledby="answer-header">
              <div className="output-section-header">
                <div className="output-header-left">
                  <span className="output-icon-badge">✨</span>
                  <h3 id="answer-header" className="output-heading">Answer</h3>
                  <span className="output-badge-grounded">100% Grounded</span>
                </div>

                <button
                  type="button"
                  className="action-copy-btn"
                  onClick={handleCopyAnswer}
                  title="Copy full response"
                >
                  {copiedAnswer ? (
                    <>
                      <span>Copied</span>
                      <span className="copy-check">✓</span>
                    </>
                  ) : (
                    <>
                      <span>Copy Answer</span>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </svg>
                    </>
                  )}
                </button>
              </div>

              <div className="answer-content-panel">
                <FormattedAnswer text={queryResult.answer || "No response generated."} />
              </div>
            </section>

            {/* Agent Investigation Pipeline */}
            {queryResult.tools_used && queryResult.tools_used.length > 0 && (
              <div className="agent-investigation-accordion">
                <div
                  className="accordion-header-row"
                  onClick={() => setTraceExpanded((prev) => !prev)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") setTraceExpanded((prev) => !prev); }}
                >
                  <div className="accordion-title-left">
                    <span className="accordion-indicator-icon">⚡</span>
                    <strong>LangGraph Execution Trail</strong>
                    <span className="accordion-steps-count">
                      ({queryResult.tools_used.length} tools orchestrated)
                    </span>
                  </div>

                  <span className="accordion-toggle-hint">
                    {traceExpanded ? "Hide Trail ▲" : "View Trail ▼"}
                  </span>
                </div>

                {traceExpanded && (
                  <div className="accordion-body-content">
                    {/* Visual pipeline progression */}
                    <div className="pipeline-steps-track">
                      {queryResult.tools_used.map((tool, idx) => (
                        <React.Fragment key={idx}>
                          <div className="pipeline-node">
                            <span className="node-index">{idx + 1}</span>
                            <span className="node-name">{tool}</span>
                          </div>
                          {idx < queryResult.tools_used.length - 1 && (
                            <span className="node-connector-arrow">→</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>

                    {/* Discovered Graph Relationships */}
                    {queryResult.graph_investigations && queryResult.graph_investigations.length > 0 && (
                      <div className="graph-traversed-box">
                        <span className="graph-box-label">Repository Graph Nodes Traversed:</span>
                        {queryResult.graph_investigations.map((inv, idx) => (
                          <div key={idx} className="graph-entry-row">
                            {inv.file && inv.dependencies && inv.dependencies.length > 0 && (
                              <span>
                                🔗 <code>{inv.file}</code> depends on:{" "}
                                {inv.dependencies.map((dep, dIdx) => (
                                  <span key={dIdx} className="dep-symbol-tag">
                                    {dep}
                                  </span>
                                ))}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Evidence Section */}
            <section className="structural-output-section" aria-labelledby="evidence-header">
              <div className="output-section-header">
                <div className="output-header-left">
                  <h3 id="evidence-header" className="output-heading">Evidence</h3>
                  {queryResult.evidence && queryResult.evidence.length > 0 && (
                    <span className="evidence-count-pill">
                      {queryResult.evidence.length} citations
                    </span>
                  )}
                </div>

                {queryResult.evidence && queryResult.evidence.length > 1 && (
                  <button
                    type="button"
                    className="toggle-all-excerpts-btn"
                    onClick={() => setExpandAllExcerpts((prev) => !prev)}
                  >
                    {expandAllExcerpts ? "Collapse All Excerpts" : "Expand All Excerpts"}
                  </button>
                )}
              </div>

              {queryResult.evidence && queryResult.evidence.length > 0 ? (
                <div className="evidence-cards-stream">
                  {queryResult.evidence.map((item, index) => {
                    const lineInfo =
                      item.lines && item.lines !== "Unknown" ? `Lines ${item.lines}` : null;
                    const isExpanded = expandAllExcerpts || expandedExcerptIndex === index;

                    // Locate matching code content from retrieved chunks or source inspections
                    const matchingChunk = (queryResult.retrieved_chunks || []).find(
                      (c) => c.metadata && c.metadata.file === item.file
                    );
                    const matchingInsp = (queryResult.source_inspections || []).find(
                      (i) => i.file === item.file && i.content
                    );
                    const codeExcerpt =
                      (matchingInsp && matchingInsp.content) ||
                      (matchingChunk && matchingChunk.content) ||
                      null;

                    return (
                      <article key={index} className="evidence-citation-card">
                        <div className="citation-header-bar">
                          <div className="citation-file-meta">
                            <FileBadge filepath={item.file} />
                            <span className="citation-filepath">{item.file}</span>
                            {lineInfo && (
                              <>
                                <span className="citation-sep">·</span>
                                <span className="citation-lines">{lineInfo}</span>
                              </>
                            )}
                            {item.symbol && item.symbol !== "dependency" && (
                              <>
                                <span className="citation-sep">·</span>
                                <span className="citation-symbol">{item.symbol}</span>
                              </>
                            )}
                          </div>

                          {codeExcerpt && (
                            <button
                              type="button"
                              className="view-excerpt-action-btn"
                              onClick={() =>
                                setExpandedExcerptIndex(isExpanded ? null : index)
                              }
                            >
                              {isExpanded ? "Hide excerpt ▲" : "View excerpt ▼"}
                            </button>
                          )}
                        </div>

                        {/* Expandable IDE code excerpt box */}
                        {isExpanded && codeExcerpt && (
                          <div className="citation-code-viewer">
                            <div className="code-viewer-topbar">
                              <span className="viewer-file-tag">
                                {item.file} {lineInfo ? `(${lineInfo})` : ""}
                              </span>
                              <button
                                type="button"
                                className="viewer-copy-btn"
                                onClick={() => handleCopySnippet(codeExcerpt, index)}
                              >
                                {copiedSnippetIndex === index ? "Copied ✓" : "Copy Snippet"}
                              </button>
                            </div>
                            <pre className="viewer-pre-block">
                              <code>{codeExcerpt}</code>
                            </pre>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              ) : (
                <p className="no-evidence-text">No direct repository citations returned.</p>
              )}

              <p className="evidence-grounding-disclaimer">
                Evidence is grounded in repository source code and dependency graph relationships.
              </p>
            </section>
          </>
        ) : (
          /* Default state matching reference image's structural placeholder */
          <>
            <div className="structural-section-divider" />

            <section className="structural-output-section">
              <div className="output-section-header">
                <div className="output-header-left">
                  <span className="output-icon-badge">✨</span>
                  <h3 className="output-heading">Answer</h3>
                </div>
              </div>
              <div className="answer-content-panel placeholder-state">
                <p className="placeholder-description">
                  Display the actual response from your backend here.
                </p>
              </div>
            </section>

            <section className="structural-output-section">
              <div className="output-section-header">
                <div className="output-header-left">
                  <h3 className="output-heading">Evidence</h3>
                </div>
              </div>
              <div className="evidence-empty-state">
                <p className="empty-evidence-hint">
                  No evidence citations yet. Ingest a repository and submit a query to view code citations.
                </p>
              </div>
            </section>
          </>
        )}
      </main>

      {/* Modern Developer Tool Footer */}
      <footer className="app-system-footer">
        <div className="footer-specs-list">
          <span>Tree-sitter Parser</span>
          <span className="sep">•</span>
          <span>BGE-M3 Embeddings</span>
          <span className="sep">•</span>
          <span>Qdrant Vector DB</span>
          <span className="sep">•</span>
          <span>LangGraph Orchestrator</span>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  );
}
