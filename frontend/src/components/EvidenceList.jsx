import React from "react";
import EvidenceCard from "./EvidenceCard";

export default function EvidenceList({ evidence = [], retrievedChunks = [], sourceInspections = [] }) {
  if (!evidence || evidence.length === 0) {
    return (
      <div className="evidence-section empty-evidence">
        <p className="evidence-heading">SOURCE EVIDENCE</p>
        <div className="empty-evidence-message">
          <p>No source evidence citations were returned for this response.</p>
        </div>
      </div>
    );
  }

  return (
    <section className="evidence-section" aria-label="Source Evidence">
      <p className="evidence-heading">
        SOURCE EVIDENCE CITATIONS ({evidence.length})
      </p>
      <div className="evidence-list">
        {evidence.map((item, index) => {
          // Find matching semantic search chunk
          const matchingChunk = retrievedChunks.find((chunk) => {
            const meta = chunk.metadata || {};
            if (meta.file !== item.file) return false;
            if (item.start_line != null && meta.start_line != null) {
              return meta.start_line === item.start_line;
            }
            return true;
          });

          // Find matching inspected source file
          const matchingInspection = sourceInspections.find(
            (insp) => insp.file === item.file && insp.found && insp.content
          );

          return (
            <EvidenceCard
              key={`${item.citation || item.file}-${item.start_line || index}-${index}`}
              evidence={item}
              matchingChunk={matchingChunk}
              matchingInspection={matchingInspection}
            />
          );
        })}
      </div>
    </section>
  );
}
