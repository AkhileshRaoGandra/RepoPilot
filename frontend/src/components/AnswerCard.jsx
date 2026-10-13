import React from "react";
import EvidenceList from "./EvidenceList";
import InvestigationPath from "./InvestigationPath";

export default function AnswerCard({ queryResult, queryMode }) {
  if (!queryResult) return null;

  const isAgent = queryMode === "agent";
  const question = queryResult.question || "";
  const answer = queryResult.answer || "No answer generated.";
  const evidence = queryResult.evidence || [];
  const retrievedChunks = queryResult.retrieved_chunks || [];
  const sourceInspections = queryResult.source_inspections || [];
  const graphInvestigations = queryResult.graph_investigations || [];
  const toolsUsed = queryResult.tools_used || [];
  const status = queryResult.investigation_status || "";

  return (
    <section className="results" aria-live="polite">
      <div className="results-heading">
        <div>
          <p className="eyebrow">
            {isAgent
              ? "LANGGRAPH AGENT INVESTIGATION (/agent/query)"
              : "DETERMINISTIC RAG ANSWER (/query)"}
          </p>
          <h2>Question: &ldquo;{question}&rdquo;</h2>
        </div>
        <div className="mode-indicator-tag">
          <span className={`mode-pill ${isAgent ? "agent" : "rag"}`}>
            {isAgent ? "Agent Mode" : "RAG Mode"}
          </span>
        </div>
      </div>

      {/* Answer Box */}
      <div className="answer-box">
        <h3>Grounded Answer</h3>
        <div className="answer-content">{answer}</div>
      </div>

      {/* Agent Investigation Details */}
      {isAgent && (
        <InvestigationPath
          toolsUsed={toolsUsed}
          graphInvestigations={graphInvestigations}
          sourceInspections={sourceInspections}
          investigationStatus={status}
        />
      )}

      {/* Source Evidence Citations */}
      <EvidenceList
        evidence={evidence}
        retrievedChunks={retrievedChunks}
        sourceInspections={sourceInspections}
      />
    </section>
  );
}
