# RepoPilot

RepoPilot is an agentic repository-onboarding project. This repository currently implements **Day 1 only**: a FastAPI service that clones a public GitHub repository and extracts safe, supported source and documentation files with metadata.

## Day 1 scope

The API validates a GitHub URL, reuses an existing local clone when present, recursively scans the repository, excludes generated directories and sensitive files, and returns UTF-8 file contents with metadata. It does not include RAG, embeddings, vector databases, LLMs, agents, parsing, frontend, databases, containers, authentication, or deployment.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
```

## Run the API

From the project root:

```bash
uvicorn app.main:app --app-dir backend --reload
```

The server starts at `http://127.0.0.1:8000` and interactive documentation is at `/docs`.

## Frontend

The React frontend is a small dashboard for the Day 1 API. In a second terminal, run:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://127.0.0.1:5173`). Keep the FastAPI server running while using the frontend.

## Endpoint

`POST /repository/analyze`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/repository/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"github_url\": \"https://github.com/owner/repository\"}"
```

Example response:

```json
{
  "repository": "todo-app",
  "files_found": 42,
  "code_files": 38,
  "documentation_files": 4,
  "skipped_unreadable_files": 0,
  "files": [
    {
      "path": "backend/auth.py",
      "language": "Python",
      "extension": ".py",
      "size": 1840,
      "lines": 84,
      "content": "..."
    }
  ]
}
```

Clones are stored under `data/repositories/<owner>--<repository>/`. The service never reads sensitive `.env`, `.env.*`, `.pem`, `.key`, or `.crt` files.

## Tests

```bash
pytest
```
