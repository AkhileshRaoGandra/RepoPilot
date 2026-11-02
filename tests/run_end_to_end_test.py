import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ingestion.repository import (
    analyze_repository,
    clone_or_open_repository,
    get_active_repository,
    get_dependency_graph,
    get_vector_store,
    is_repository_indexed,
    scan_repository,
)
from app.code.parser import parse_source
from app.code.chunker import create_chunks
from app.rag import get_rag_service, get_retriever
from app.agent import get_agent_service

REPOSITORIES = [
    {
        "url": "https://github.com/pallets/markupsafe",
        "lang": "Python",
        "name": "markupsafe",
        "questions": {
            "general": "What is the primary purpose of MarkupSafe?",
            "function": "What does escape do?",
            "file": "Explain the responsibility of _native.py",
            "relationship": "How does _native.py interact with other modules?",
            "unanswerable": "Where is the cryptocurrency payment gateway implemented?",
        },
    },
    {
        "url": "https://github.com/facebook/prop-types",
        "lang": "JavaScript",
        "name": "prop-types",
        "questions": {
            "general": "What does this prop-types library do?",
            "function": "What does checkPropTypes do?",
            "file": "What is the responsibility of index.js?",
            "relationship": "What modules does index.js import or depend on?",
            "unanswerable": "Where is the SQL database migration script located?",
        },
    },
    {
        "url": "https://github.com/square/javapoet",
        "lang": "Java",
        "name": "javapoet",
        "questions": {
            "general": "What is the architectural purpose of JavaPoet?",
            "function": "What does MethodSpec do?",
            "file": "What is the responsibility of TypeSpec.java?",
            "relationship": "How does TypeSpec relate to MethodSpec and FieldSpec?",
            "unanswerable": "Where is the stripe billing webhook listener configured?",
        },
    },
]


def _parse_files_benchmark(files):
    parsed = 0
    for idx, f in enumerate(files):
        lang = str(f["language"])
        if lang in {"Python", "JavaScript", "JavaScript (React JSX)", "Java"}:
            res = parse_source(str(f["content"]), lang)
            parsed += 1
    return parsed


def main():
    print("=" * 80)
    print("REPOPILOT: END-TO-END VERIFICATION ACROSS 3 PUBLIC GITHUB REPOSITORIES")
    print("=" * 80)

    for target in REPOSITORIES:
        url = target["url"]
        lang = target["lang"]
        name = target["name"]
        print(f"\n=======================================================")
        print(f"REPOSITORY: {name} ({lang}) -> {url}")
        print(f"=======================================================")

        # 1. URL validation
        assert "https://github.com/" in url

        # 2. Clone / Open
        t0 = time.perf_counter()
        ref, repo_path = clone_or_open_repository(url)
        t_clone = time.perf_counter() - t0
        print(f"[1/8] Clone/Open: {t_clone:.2f}s (path: {repo_path.name})")

        # 3. File discovery & filtering
        t0 = time.perf_counter()
        files, unreadable, has_files = scan_repository(repo_path)
        t_scan = time.perf_counter() - t0
        print(f"[2/8] Scan & Filter: {t_scan:.3f}s (discovered: {len(files)} files, unreadable: {unreadable})")

        # 4. Parsing verification
        t0 = time.perf_counter()
        parsed_count = _parse_files_benchmark(files)
        t_parse = time.perf_counter() - t0
        print(f"[3/8] AST Parsing: {t_parse:.3f}s (parsed {parsed_count} source files)")


        # 5. Full analyze_repository (Tree-sitter, Chunking, BGE-M3 Embeddings, Qdrant Upsert, Graph)
        t0 = time.perf_counter()
        res = analyze_repository(url)
        t_analyze = time.perf_counter() - t0
        print(f"[4/8] Ingestion & Indexing Pipeline: {t_analyze:.2f}s")
        print(f"      • Chunks created: {res['chunks_created']}")
        print(f"      • Chunks stored in Qdrant: {res['chunks_stored']}")
        print(f"      • Dependency graph nodes: {res['graph_nodes']}")
        print(f"      • Dependency graph edges: {res['graph_edges']}")
        assert res["chunks_stored"] > 0, "No chunks were stored!"
        assert is_repository_indexed() is True, "Repository status reported as unindexed!"
        assert get_active_repository() == ref.name, "Active repository mismatch!"

        # 6. Semantic Retrieval
        retriever = get_retriever()
        t0 = time.perf_counter()
        chunks = retriever.retrieve(target["questions"]["general"], top_k=5)
        t_retrieval = time.perf_counter() - t0
        print(f"[5/8] Semantic Retrieval: {t_retrieval:.3f}s ({len(chunks)} chunks retrieved)")
        assert len(chunks) > 0, "Semantic retrieval returned 0 chunks!"
        first_chunk = chunks[0]
        assert "file" in first_chunk["metadata"]
        assert "start_line" in first_chunk["metadata"]
        assert "end_line" in first_chunk["metadata"]
        print(f"      Top match: {first_chunk['metadata']['file']} ({first_chunk['metadata']['start_line']}-{first_chunk['metadata']['end_line']})")

        # 7. RAG Question Answering Across Categories
        print(f"[6/8] RAG Response Generation & Answer Grounding:")
        rag_service = get_rag_service()
        for cat, question in target["questions"].items():
            t0 = time.perf_counter()
            rag_out = rag_service.answer_question(question, top_k=5)
            t_rag = time.perf_counter() - t0
            ans = rag_out["answer"]
            ev_count = len(rag_out["evidence"])
            citations = [e["citation"] for e in rag_out["evidence"][:2]]
            print(f"      • [{cat.upper()}] '{question}' ({t_rag:.2f}s)")
            print(f"        Answer: {ans[:95]}...")
            print(f"        Evidence: {ev_count} citations {citations}")

            # Verify unanswerable question grounding
            if cat == "unanswerable":
                ans_lower = ans.lower()
                assert (
                    "insufficient" in ans_lower
                    or "not implemented" in ans_lower
                    or "not present" in ans_lower
                    or "not found" in ans_lower
                    or "does not" in ans_lower
                ), f"Model hallucinated non-existent feature in answer: {ans}"

        # 8. LangGraph Agent Multi-Scenario Investigation
        print(f"[7/8] LangGraph Agent Investigation Scenarios:")
        agent_service = get_agent_service()

        # Scenario 1: Simple question
        t0 = time.perf_counter()
        agent_s1 = agent_service.run_investigation(target["questions"]["function"], top_k=3)
        t_s1 = time.perf_counter() - t0
        print(f"      • Scenario 1 (Simple): {t_s1:.2f}s | Tools: {agent_s1['tools_used']}")
        assert "semantic_search" in agent_s1["tools_used"]
        assert len(agent_s1["evidence"]) > 0

        # Scenario 3: Multi-file / relationship investigation
        t0 = time.perf_counter()
        agent_s3 = agent_service.run_investigation(target["questions"]["relationship"], top_k=3)
        t_s3 = time.perf_counter() - t0
        print(f"      • Scenario 3 (Multi-file / Graph): {t_s3:.2f}s | Tools: {agent_s3['tools_used']} | Evidence: {len(agent_s3['evidence'])}")
        assert "graph_investigation" in agent_s3["tools_used"] or len(agent_s3["evidence"]) >= 1

        # Scenario 4: Unanswerable question
        t0 = time.perf_counter()
        agent_s4 = agent_service.run_investigation(target["questions"]["unanswerable"], top_k=3)
        t_s4 = time.perf_counter() - t0
        print(f"      • Scenario 4 (Unanswerable): {t_s4:.2f}s | Status: {agent_s4['investigation_status']}")
        ans_lower = agent_s4["answer"].lower()
        assert (
            "insufficient" in ans_lower
            or "not implemented" in ans_lower
            or "not present" in ans_lower
            or "not found" in ans_lower
            or "does not" in ans_lower
        ), f"Agent hallucinated unanswerable question: {agent_s4['answer']}"

        # 9. Re-indexing Isolation & Idempotence
        print(f"[8/8] Re-indexing Idempotence & Vector Store Isolation:")
        reindex_res = analyze_repository(url)
        assert reindex_res["chunks_stored"] == res["chunks_stored"]
        assert get_vector_store().count() == res["chunks_stored"]
        print(f"      • Verified exact count: {get_vector_store().count()} chunks (no duplicate points)")

    print("\n" + "=" * 80)
    print("ALL 3 REPOSITORIES PASSED FULL END-TO-END VERIFICATION!")
    print("=" * 80)


if __name__ == "__main__":
    main()
