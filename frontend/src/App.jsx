import React, { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function EvidenceCard({ evidence, matchingChunk }) {
  const [expanded, setExpanded] = useState(false);
  const codeContent = matchingChunk ? matchingChunk.content : null;

  return (
    <article className="evidence-card">
      <div className="evidence-header">
        <div className="citation-title">
          <code>{evidence.citation}</code>
          {evidence.symbol && <span className="symbol-badge">{evidence.symbol}</span>}
          {evidence.score !== null && evidence.score !== undefined && (
            <span className="score-badge">Similarity: {(evidence.score * 100).toFixed(1)}%</span>
          )}
        </div>
        {codeContent && (
          <button className="secondary" onClick={() => setExpanded((val) => !val)}>
            {expanded ? "Hide chunk" : "View chunk"}
          </button>
        )}
      </div>
      {expanded && codeContent && (
        <pre><code>{codeContent}</code></pre>
      )}
    </article>
  );
}

function FileCard({ file }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="file-card">
      <div className="file-heading">
        <div>
          <code>{file.path}</code>
          <p>{file.language} · {file.lines} lines · {file.size.toLocaleString()} bytes</p>
        </div>
        <button className="secondary" onClick={() => setExpanded((val) => !val)}>
          {expanded ? "Hide content" : "View content"}
        </button>
      </div>
      {expanded && <pre><code>{file.content}</code></pre>}
    </article>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("ask");

  // Ingestion State
  const [githubUrl, setGithubUrl] = useState("");
  const [ingestResult, setIngestResult] = useState(null);
  const [ingestError, setIngestError] = useState("");
  const [ingestLoading, setIngestLoading] = useState(false);

  // RAG Query State
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [queryMode, setQueryMode] = useState("agent");
  const [queryResult, setQueryResult] = useState(null);
  const [queryError, setQueryError] = useState("");
  const [queryLoading, setQueryLoading] = useState(false);

  // Sample prompt suggestions
  const suggestions = [
    "Where is authentication implemented?",
    "How does token generation work?",
    "What classes and methods are in the codebase?",
    "Inspect surrounding lines around login in auth/service.py",
  ];

  async function handleIngestSubmit(event) {
    event.preventDefault();
    setIngestLoading(true);
    setIngestError("");
    setIngestResult(null);

    try {
      const response = await fetch(`${API_BASE}/repository/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: githubUrl }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Repository analysis failed.");
      }
      setIngestResult(payload);
    } catch (err) {
      setIngestError(err.message || "Unable to reach the API. Is FastAPI running on port 8000?");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleQuerySubmit(event) {
    event.preventDefault();
    if (!question.trim()) return;

    setQueryLoading(true);
    setQueryError("");
    setQueryResult(null);

    const endpoint = queryMode === "agent" ? `${API_BASE}/agent/query` : `${API_BASE}/query`;

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question.trim(), top_k: Number(topK) }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Query failed.");
      }
      setQueryResult(payload);
    } catch (err) {
      setQueryError(err.message || "Unable to reach the API. Is FastAPI running on port 8000?");
    } finally {
      setQueryLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="hero">
        <p className="eyebrow">DAY 6 · LANGGRAPH AGENTIC ORCHESTRATION</p>
        <h1>RepoPilot</h1>
        <p className="subtitle">
          AI-powered repository understanding system. Ingest codebases, index semantic vectors in Qdrant,
          and use LangGraph agentic orchestration with tool investigation to produce grounded answers.
        </p>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs" aria-label="Sections">
        <button
          type="button"
          className={`tab-btn ${activeTab === "ask" ? "active" : ""}`}
          onClick={() => setActiveTab("ask")}
        >
          Ask Questions (Day 5 RAG)
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === "ingest" ? "active" : ""}`}
          onClick={() => setActiveTab("ingest")}
        >
          Ingest Repository (Days 1–4)
        </button>
      </nav>

      {/* TAB 1: ASK REPOPILOT (RAG Q&A) */}
      {activeTab === "ask" && (
        <section aria-labelledby="ask-heading">
          <div className="panel">
            <h2 id="ask-heading" style={{ marginBottom: "14px" }}>Ask the Codebase</h2>
            <form onSubmit={handleQuerySubmit}>
              <label htmlFor="repo-question">Ask a question about the repository</label>
              <div className="query-form">
                <input
                  id="repo-question"
                  type="text"
                  required
                  placeholder="e.g. How does authentication work? Where is login defined?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
                <button type="submit" disabled={queryLoading}>
                  {queryLoading ? "Retrieving & Answering…" : "Ask RepoPilot"}
                </button>
              </div>

              <div className="query-controls">
                <label htmlFor="top-k-select">
                  Top-K context chunks:
                  <select
                    id="top-k-select"
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                  >
                    <option value={3}>3 chunks</option>
                    <option value={5}>5 chunks</option>
                    <option value={8}>8 chunks</option>
                    <option value={10}>10 chunks</option>
                  </select>
                </label>

                <div className="query-mode-selector">
                  <span style={{ fontSize: "0.85rem", color: "#aab5cd" }}>Mode:</span>
                  <button
                    type="button"
                    className={`mode-btn ${queryMode === "agent" ? "active" : ""}`}
                    onClick={() => setQueryMode("agent")}
                  >
                    Agentic RAG (/agent/query)
                  </button>
                  <button
                    type="button"
                    className={`mode-btn ${queryMode === "rag" ? "active" : ""}`}
                    onClick={() => setQueryMode("rag")}
                  >
                    Standard RAG (/query)
                  </button>
                </div>
              </div>
            </form>

            <div className="suggestions">
              <span>Suggested questions:</span>
              {suggestions.map((s, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="chip"
                  onClick={() => setQuestion(s)}
                >
                  {s}
                </button>
              ))}
            </div>

            {queryError && <p className="error" role="alert">{queryError}</p>}
          </div>

          {/* Q&A Result */}
          {queryResult && (
            <section className="results" aria-live="polite">
              <div className="results-heading">
                <div>
                  <p className="eyebrow">
                    {queryMode === "agent" ? "LANGGRAPH AGENT INVESTIGATION" : "ANSWER GROUNDED IN REPOSITORY"}
                  </p>
                  <h2>Question: &ldquo;{queryResult.question}&rdquo;</h2>
                </div>
              </div>

              {queryResult.tools_used && queryResult.tools_used.length > 0 && (
                <div className="tools-used-bar">
                  <span className="tools-label">Agent Tools Invoked:</span>
                  {queryResult.tools_used.map((tool, idx) => (
                    <span key={idx} className="tool-badge">
                      {tool === "semantic_search" ? "🔍 semantic_search" : "📄 " + tool}
                    </span>
                  ))}
                  {queryResult.investigation_status && (
                    <span className="status-badge">status: {queryResult.investigation_status}</span>
                  )}
                </div>
              )}

              <div className="answer-box">
                <h3>Grounded Answer</h3>
                <div className="answer-content">{queryResult.answer}</div>
              </div>


              {queryResult.evidence && queryResult.evidence.length > 0 && (
                <div className="evidence-section">
                  <p className="evidence-heading">SOURCE EVIDENCE CITATIONS ({queryResult.evidence.length})</p>
                  <div className="evidence-list">
                    {queryResult.evidence.map((item, index) => {
                      // Find matching raw chunk for code inspection
                      const matchingChunk = queryResult.retrieved_chunks
                        ? queryResult.retrieved_chunks.find(
                            (c) =>
                              c.metadata &&
                              c.metadata.file === item.file &&
                              c.metadata.start_line === item.start_line
                          ) || queryResult.retrieved_chunks[index]
                        : null;

                      return (
                        <EvidenceCard
                          key={`${item.citation}-${index}`}
                          evidence={item}
                          matchingChunk={matchingChunk}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </section>
          )}
        </section>
      )}

      {/* TAB 2: REPOSITORY INGESTION & INDEXING */}
      {activeTab === "ingest" && (
        <section aria-labelledby="ingest-heading">
          <div className="panel">
            <h2 id="ingest-heading" style={{ marginBottom: "14px" }}>Ingest & Index Repository</h2>
            <form onSubmit={handleIngestSubmit}>
              <label htmlFor="github-url">Public GitHub repository URL</label>
              <div className="url-form">
                <input
                  id="github-url"
                  type="url"
                  required
                  placeholder="https://github.com/owner/repository"
                  value={githubUrl}
                  onChange={(event) => setGithubUrl(event.target.value)}
                />
                <button type="submit" disabled={ingestLoading}>
                  {ingestLoading ? "Ingesting & Indexing…" : "Analyze & Index"}
                </button>
              </div>
            </form>
            {ingestError && <p className="error" role="alert">{ingestError}</p>}
          </div>

          {ingestResult && (
            <section className="results" aria-live="polite">
              <div className="results-heading">
                <div>
                  <p className="eyebrow">INDEXING COMPLETE · STORED IN QDRANT</p>
                  <h2>{ingestResult.repository}</h2>
                </div>
                <button
                  type="button"
                  onClick={() => setActiveTab("ask")}
                  style={{ whiteSpace: "nowrap" }}
                >
                  Ask questions about this repo &rarr;
                </button>
              </div>

              {ingestResult.skipped_unreadable_files > 0 && (
                <p className="notice">Skipped {ingestResult.skipped_unreadable_files} unreadable file(s).</p>
              )}

              <div className="stats">
                <div><strong>{ingestResult.files_found}</strong><span>files found</span></div>
                <div><strong>{ingestResult.code_files}</strong><span>code files</span></div>
                <div><strong>{ingestResult.parsed_files || 0}</strong><span>parsed files</span></div>
                <div><strong>{ingestResult.chunks_created || 0}</strong><span>chunks created</span></div>
                <div><strong>{ingestResult.chunks_stored || 0}</strong><span>chunks in Qdrant</span></div>
              </div>

              {ingestResult.files && (
                <div>
                  <p className="eyebrow">EXTRACTED FILES ({ingestResult.files.length})</p>
                  <div className="file-list">
                    {ingestResult.files.map((file) => (
                      <FileCard file={file} key={file.path} />
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}
        </section>
      )}
    </main>
  );
}
