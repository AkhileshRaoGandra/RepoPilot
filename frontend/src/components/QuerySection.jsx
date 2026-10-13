import React, { useState } from "react";
import { queryRAG, queryAgent } from "../api/client";
import AnswerCard from "./AnswerCard";

export default function QuerySection({ analyzedRepoName, onNavigateToIngest }) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [queryMode, setQueryMode] = useState("agent");
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState("");
  const [queryResult, setQueryResult] = useState(null);

  // Suggested prompt chips tailored to repository onboarding
  const suggestions = [
    "Where is authentication implemented?",
    "How does a request move from the controller to the repository layer?",
    "What classes and methods are in the codebase?",
    "What dependencies and imports does the service layer use?",
    "Inspect the database connection and repository queries",
  ];

  async function handleSubmit(event) {
    event.preventDefault();
    setQueryError("");

    const trimmed = question.trim();
    if (!trimmed) {
      setQueryError("Please enter a question about the repository.");
      return;
    }

    setQueryLoading(true);
    setQueryResult(null);

    try {
      let payload;
      if (queryMode === "agent") {
        payload = await queryAgent(trimmed, topK);
      } else {
        payload = await queryRAG(trimmed, topK);
      }
      setQueryResult(payload);
    } catch (err) {
      setQueryError(err.message || "Failed to execute query.");
    } finally {
      setQueryLoading(false);
    }
  }

  return (
    <section aria-labelledby="ask-heading">
      <div className="panel">
        <div className="ask-header-row">
          <div>
            <h2 id="ask-heading" style={{ marginBottom: "6px" }}>
              Ask the Codebase
            </h2>
            <p className="panel-desc">
              Ask questions about architecture, code flow, functions, or dependencies.
              Answers are strictly grounded in repository source code with line-level evidence.
            </p>
          </div>
          {analyzedRepoName && (
            <div className="active-repo-badge">
              <span>Active Repository:</span>
              <strong>{analyzedRepoName}</strong>
            </div>
          )}
        </div>

        {!analyzedRepoName && (
          <div className="repo-notice-banner">
            <span>ℹ️ No repository has been ingested in this session yet.</span>
            {onNavigateToIngest && (
              <button
                type="button"
                className="secondary small-btn"
                onClick={onNavigateToIngest}
              >
                Ingest a repository first &rarr;
              </button>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <label htmlFor="repo-question">Question about the repository</label>
          <div className="query-form">
            <input
              id="repo-question"
              type="text"
              required
              disabled={queryLoading}
              placeholder="e.g. How does authentication work? How does a request reach the database?"
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (queryError) setQueryError("");
              }}
            />
            <button type="submit" disabled={queryLoading}>
              {queryLoading ? "Investigating & Answering…" : "Ask RepoPilot"}
            </button>
          </div>

          <div className="query-controls">
            <label htmlFor="top-k-select">
              Context chunks limit (Top-K):
              <select
                id="top-k-select"
                disabled={queryLoading}
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
              <span className="mode-selector-label">Pipeline Mode:</span>
              <button
                type="button"
                className={`mode-btn ${queryMode === "agent" ? "active" : ""}`}
                disabled={queryLoading}
                onClick={() => setQueryMode("agent")}
              >
                🤖 Agent Mode (/agent/query)
              </button>
              <button
                type="button"
                className={`mode-btn ${queryMode === "rag" ? "active" : ""}`}
                disabled={queryLoading}
                onClick={() => setQueryMode("rag")}
              >
                ⚡ RAG Mode (/query)
              </button>
            </div>
          </div>

          <div className="mode-explanation-card">
            {queryMode === "agent" ? (
              <p>
                <strong>Agent Mode (LangGraph):</strong> Multi-step stateful investigation using semantic search,
                Tree-sitter dependency graph exploration, and targeted source code file inspection.
              </p>
            ) : (
              <p>
                <strong>RAG Mode:</strong> Direct single-pass semantic similarity retrieval with BGE-M3 + Qdrant,
                followed by grounded LLM response generation.
              </p>
            )}
          </div>
        </form>

        <div className="suggestions">
          <span>Suggested queries:</span>
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className="chip"
              disabled={queryLoading}
              onClick={() => {
                setQuestion(s);
                if (queryError) setQueryError("");
              }}
            >
              {s}
            </button>
          ))}
        </div>

        {queryLoading && (
          <div className="loading-banner" role="status" aria-live="polite">
            <div className="spinner" />
            <div>
              <strong>
                {queryMode === "agent"
                  ? "LangGraph agent is orchestrating repository tools..."
                  : "Retrieving semantic chunks and compiling grounded response..."}
              </strong>
              <p style={{ margin: "4px 0 0", color: "#aab5cd", fontSize: "0.85rem" }}>
                {queryMode === "agent"
                  ? "Running semantic search, exploring dependency graph topology, and inspecting source files."
                  : "Embedding query with BGE-M3, querying Qdrant vectors, and calling the LLM."}
              </p>
            </div>
          </div>
        )}

        {queryError && (
          <div className="error" role="alert">
            <strong>Query Error:</strong> {queryError}
          </div>
        )}
      </div>

      {queryResult && (
        <AnswerCard queryResult={queryResult} queryMode={queryMode} />
      )}
    </section>
  );
}
