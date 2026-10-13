import React, { useState } from "react";
import { analyzeRepository, isValidGitHubUrl } from "../api/client";

function FileCard({ file }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="file-card">
      <div className="file-heading">
        <div>
          <code>{file.path}</code>
          <p>
            {file.language || "Unknown"} · {file.lines ?? 0} lines ·{" "}
            {(file.size ?? 0).toLocaleString()} bytes
          </p>
        </div>
        <button
          type="button"
          className="secondary"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Hide content" : "View content"}
        </button>
      </div>
      {expanded && (
        <pre>
          <code>{file.content}</code>
        </pre>
      )}
    </article>
  );
}

export default function RepositoryInput({
  onRepositoryAnalyzed,
  onNavigateToAsk,
  currentRepoInfo,
}) {
  const [githubUrl, setGithubUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(currentRepoInfo);
  const [filesExpanded, setFilesExpanded] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const trimmed = githubUrl.trim();
    if (!trimmed) {
      setError("Please enter a GitHub repository URL.");
      return;
    }

    if (!isValidGitHubUrl(trimmed)) {
      setError("Please provide a valid HTTPS GitHub repository URL (e.g. https://github.com/owner/repository).");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const payload = await analyzeRepository(trimmed);
      setResult(payload);
      if (onRepositoryAnalyzed) {
        onRepositoryAnalyzed(payload);
      }
    } catch (err) {
      setError(err.message || "Failed to analyze repository.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section aria-labelledby="ingest-heading">
      <div className="panel">
        <h2 id="ingest-heading" style={{ marginBottom: "14px" }}>
          Ingest & Index GitHub Repository
        </h2>
        <p className="panel-desc">
          Provide any public GitHub repository URL. RepoPilot will clone the repository,
          parse source code with Tree-sitter, split symbols with a code-aware chunker,
          generate 1024-dim dense vectors with BGE-M3, store them in Qdrant, and build a dependency graph.
        </p>

        <form onSubmit={handleSubmit}>
          <label htmlFor="github-url">Public GitHub repository URL</label>
          <div className="url-form">
            <input
              id="github-url"
              type="url"
              required
              disabled={loading}
              placeholder="https://github.com/owner/repository"
              value={githubUrl}
              onChange={(e) => {
                setGithubUrl(e.target.value);
                if (error) setError("");
              }}
            />
            <button type="submit" disabled={loading}>
              {loading ? "Ingesting & Indexing…" : "Analyze & Index"}
            </button>
          </div>
        </form>

        {loading && (
          <div className="loading-banner" role="status" aria-live="polite">
            <div className="spinner" />
            <div>
              <strong>Cloning, parsing, and embedding repository...</strong>
              <p style={{ margin: "4px 0 0", color: "#aab5cd", fontSize: "0.85rem" }}>
                BGE-M3 is generating 1024-dimensional dense vectors and Qdrant is indexing chunks.
                This typically takes 15–30 seconds on CPU.
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="error" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {result && (
        <section className="results" aria-live="polite">
          <div className="results-heading">
            <div>
              <p className="eyebrow">INDEXING COMPLETE · STORED IN QDRANT</p>
              <h2>{result.repository}</h2>
            </div>
            {onNavigateToAsk && (
              <button
                type="button"
                onClick={onNavigateToAsk}
                style={{ whiteSpace: "nowrap" }}
              >
                Ask questions about this repo &rarr;
              </button>
            )}
          </div>

          {result.skipped_unreadable_files > 0 && (
            <p className="notice">
              Skipped {result.skipped_unreadable_files} unreadable or sensitive file(s).
            </p>
          )}

          {result.parsing_failures && result.parsing_failures.length > 0 && (
            <p className="notice">
              Skipped {result.parsing_failures.length} file(s) due to parse errors.
            </p>
          )}

          <div className="stats">
            <div>
              <strong>{result.files_found ?? 0}</strong>
              <span>files found</span>
            </div>
            <div>
              <strong>{result.code_files ?? 0}</strong>
              <span>code files</span>
            </div>
            <div>
              <strong>{result.parsed_files ?? 0}</strong>
              <span>parsed files</span>
            </div>
            <div>
              <strong>{result.chunks_created ?? 0}</strong>
              <span>chunks created</span>
            </div>
            <div>
              <strong>{result.chunks_stored ?? 0}</strong>
              <span>chunks in Qdrant</span>
            </div>
            <div>
              <strong>{result.graph_nodes ?? 0}</strong>
              <span>graph nodes</span>
            </div>
            <div>
              <strong>{result.graph_edges ?? 0}</strong>
              <span>graph edges</span>
            </div>
          </div>

          {result.files && result.files.length > 0 && (
            <div className="extracted-files-section">
              <div className="extracted-files-header">
                <p className="eyebrow">
                  EXTRACTED REPOSITORY FILES ({result.files.length})
                </p>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setFilesExpanded((prev) => !prev)}
                >
                  {filesExpanded ? "Collapse file list" : "Expand file list"}
                </button>
              </div>

              {filesExpanded && (
                <div className="file-list">
                  {result.files.map((file) => (
                    <FileCard file={file} key={file.path} />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </section>
  );
}
