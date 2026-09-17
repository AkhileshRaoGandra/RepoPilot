import React, { useState } from "react";

const API_URL = "http://127.0.0.1:8000/repository/analyze";

function FileCard({ file }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="file-card">
      <div className="file-heading">
        <div>
          <code>{file.path}</code>
          <p>{file.language} · {file.lines} lines · {file.size.toLocaleString()} bytes</p>
        </div>
        <button className="secondary" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Hide content" : "View content"}
        </button>
      </div>
      {expanded && <pre><code>{file.content}</code></pre>}
    </article>
  );
}

export default function App() {
  const [githubUrl, setGithubUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: githubUrl }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Repository analysis failed.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the API. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">DAY 1 · REPOSITORY INGESTION</p>
        <h1>RepoPilot</h1>
        <p className="subtitle">Analyze a public GitHub repository and inspect its supported source files.</p>
      </section>

      <section className="panel">
        <form onSubmit={handleSubmit}>
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
            <button type="submit" disabled={loading}>{loading ? "Analyzing…" : "Analyze repository"}</button>
          </div>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {result && (
        <section className="results" aria-live="polite">
          <div className="results-heading">
            <div>
              <p className="eyebrow">ANALYSIS COMPLETE</p>
              <h2>{result.repository}</h2>
            </div>
            {result.skipped_unreadable_files > 0 && (
              <p className="notice">Skipped {result.skipped_unreadable_files} unreadable file(s).</p>
            )}
          </div>
          <div className="stats">
            <div><strong>{result.files_found}</strong><span>files found</span></div>
            <div><strong>{result.code_files}</strong><span>code files</span></div>
            <div><strong>{result.documentation_files}</strong><span>documentation files</span></div>
          </div>
          <div className="file-list">
            {result.files.map((file) => <FileCard file={file} key={file.path} />)}
          </div>
        </section>
      )}
    </main>
  );
}
