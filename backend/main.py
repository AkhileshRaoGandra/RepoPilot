"""Root FastAPI entrypoint for Vercel service deployment."""

import os

# Suppress GitPython from crashing if git binary is missing in serverless environments (e.g. Vercel)
os.environ["GIT_PYTHON_REFRESH"] = "0"

import sys
from pathlib import Path

# Ensure backend root is on sys.path
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.main import app

__all__ = ["app"]
