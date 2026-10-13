import React, { useState } from "react";

export default function InvestigationPath({
  toolsUsed = [],
  graphInvestigations = [],
  sourceInspections = [],
  investigationStatus = "",
}) {
  const [expanded, setExpanded] = useState(true);

  const hasGraph = graphInvestigations && graphInvestigations.length > 0;
  const hasInspections = sourceInspections && sourceInspections.length > 0;

  if (!hasGraph && !hasInspections && toolsUsed.length === 0) {
    return null;
  }

  return (
    <div className="investigation-path-section">
      <div className="investigation-path-header">
        <div className="investigation-path-title">
          <span className="investigation-path-icon">🧭</span>
          <strong>Agent Investigation Trace</strong>
          {investigationStatus && (
            <span className="status-badge">Status: {investigationStatus}</span>
          )}
        </div>
        <button
          type="button"
          className="secondary trace-toggle-btn"
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
        >
          {expanded ? "Collapse trace" : "Expand trace"}
        </button>
      </div>

      {expanded && (
        <div className="investigation-path-body">
          {/* Tool execution timeline */}
          {toolsUsed.length > 0 && (
            <div className="tools-execution-row">
              <span className="tools-label">Tool Workflow:</span>
              <div className="tool-badges-list">
                {toolsUsed.map((tool, idx) => {
                  let icon = "⚙️";
                  let label = tool;
                  if (tool === "semantic_search") {
                    icon = "🔍";
                    label = "Semantic Search (BGE-M3 + Qdrant)";
                  } else if (tool === "graph_investigation") {
                    icon = "🔗";
                    label = "Repository Graph Tool";
                  } else if (tool === "source_inspection") {
                    icon = "📄";
                    label = "Source Inspection Tool";
                  }

                  return (
                    <span key={idx} className="tool-step-badge">
                      <span className="step-number">{idx + 1}</span>
                      <span>{icon} {label}</span>
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* Graph Investigations */}
          {hasGraph && (
            <div className="graph-results-group">
              <span className="group-heading">Discovered Code Topology:</span>
              <div className="investigation-path">
                {graphInvestigations.map((inv, idx) => (
                  <div key={idx} className="graph-result-card">
                    {/* Dependencies */}
                    {inv.file && inv.dependencies && inv.dependencies.length > 0 && (
                      <div className="dep-chain">
                        <span className="chain-label">File Imports:</span>
                        <code className="chain-node">{inv.file}</code>
                        {inv.dependencies.map((dep, di) => (
                          <div key={di} className="chain-arrow">
                            <span className="arrow-line">↓ imports</span>
                            <code className="chain-node">{dep}</code>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Dependents */}
                    {inv.file && inv.dependents && inv.dependents.length > 0 && (
                      <div className="dep-chain">
                        <span className="chain-label">Imported By:</span>
                        {inv.dependents.map((dep, di) => (
                          <div key={di} className="chain-arrow">
                            <code className="chain-node">{dep}</code>
                            <span className="arrow-line">↓ imports</span>
                          </div>
                        ))}
                        <code className="chain-node">{inv.file}</code>
                      </div>
                    )}

                    {/* Related Symbols */}
                    {inv.related && inv.related.length > 0 && (
                      <div className="related-symbols">
                        <span className="related-label">Symbols related to "{inv.query}":</span>
                        <div className="related-symbols-list">
                          {inv.related.map((r, ri) => (
                            <div key={ri} className="related-item">
                              <code>{r.name}</code>
                              <span className="symbol-badge">{r.type}</span>
                              {r.file && (
                                <span className="related-file-path"> in {r.file}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Inspected Source Files */}
          {hasInspections && (
            <div className="inspections-group">
              <span className="group-heading">Targeted Source Inspections:</span>
              <div className="inspections-list">
                {sourceInspections.map((insp, idx) => (
                  <div key={idx} className="inspection-item-card">
                    <div className="inspection-item-header">
                      <code>{insp.file}</code>
                      <span className="line-badge">Lines {insp.lines || `${insp.start_line}-${insp.end_line}`}</span>
                      <span className={`status-pill ${insp.found ? "found" : "missing"}`}>
                        {insp.found ? "✓ Loaded" : "✗ Not found"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
