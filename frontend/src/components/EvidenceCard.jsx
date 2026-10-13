import React, { useState } from "react";

export default function EvidenceCard({ evidence, matchingChunk, matchingInspection }) {
  const [expanded, setExpanded] = useState(false);

  // Extract source code excerpt from matching chunk or matching source inspection
  const codeContent =
    (matchingInspection && matchingInspection.content) ||
    (matchingChunk && matchingChunk.content) ||
    null;

  // Format line numbers cleanly
  const lineDisplay =
    evidence.lines && evidence.lines !== "Unknown"
      ? evidence.lines
      : evidence.start_line != null && evidence.end_line != null
      ? `${evidence.start_line}-${evidence.end_line}`
      : evidence.start_line != null
      ? String(evidence.start_line)
      : null;

  return (
    <article className="evidence-card">
      <div className="evidence-header">
        <div className="citation-title">
          <code className="citation-code">{evidence.citation || evidence.file}</code>
          {evidence.symbol && <span className="symbol-badge">{evidence.symbol}</span>}
          {lineDisplay && lineDisplay !== "graph" && (
            <span className="line-badge">Lines {lineDisplay}</span>
          )}
          {evidence.score != null && (
            <span className="score-badge">Similarity: {(evidence.score * 100).toFixed(1)}%</span>
          )}
        </div>
        {codeContent && (
          <button
            type="button"
            className="secondary excerpt-toggle-btn"
            onClick={() => setExpanded((prev) => !prev)}
            aria-expanded={expanded}
          >
            {expanded ? "Hide source excerpt" : "View source excerpt"}
          </button>
        )}
      </div>

      {expanded && codeContent && (
        <div className="source-excerpt-container">
          <div className="source-excerpt-meta">
            <span>{evidence.file}</span>
            {lineDisplay && <span>Lines: {lineDisplay}</span>}
          </div>
          <pre className="source-excerpt-code">
            <code>{codeContent}</code>
          </pre>
        </div>
      )}
    </article>
  );
}
