"""Background task runner and state management for repository ingestion."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger("repopilot.tasks")


class IngestionTask:
    """Represents the lifecycle and progress of an asynchronous ingestion job."""

    def __init__(self, task_id: str, github_url: str) -> None:
        self.task_id = task_id
        self.github_url = github_url
        self.status: str = "queued"  # "queued" | "running" | "completed" | "failed"
        self.progress: int = 0
        self.step: str = "Queued for processing"
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.completed_at: float | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "github_url": self.github_url,
            "status": self.status,
            "progress": self.progress,
            "step": self.step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class TaskManager:
    """Thread-safe in-memory store for ingestion tasks."""

    def __init__(self, max_history: int = 100) -> None:
        self._tasks: dict[str, IngestionTask] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def create_task(self, github_url: str) -> IngestionTask:
        task_id = str(uuid.uuid4())
        task = IngestionTask(task_id=task_id, github_url=github_url)
        with self._lock:
            # Evict oldest tasks if limit reached
            if len(self._tasks) >= self._max_history:
                oldest_key = min(self._tasks.keys(), key=lambda k: self._tasks[k].created_at)
                self._tasks.pop(oldest_key, None)
            self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> IngestionTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: int | None = None,
        step: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task.updated_at = time.time()
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = progress
            if step is not None:
                task.step = step
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if status in {"completed", "failed"}:
                task.completed_at = time.time()
            return task


# Global singleton task manager
task_manager = TaskManager()


def run_ingestion_task(task_id: str, github_url: str) -> None:
    """Background worker function executed by FastAPI BackgroundTasks."""
    from app.ingestion.repository import analyze_repository

    logger.info(f"Starting ingestion task {task_id} for {github_url}")
    task_manager.update_task(
        task_id,
        status="running",
        progress=5,
        step="Starting repository clone and analysis...",
    )

    def on_progress(percent: int, step_description: str) -> None:
        task_manager.update_task(task_id, progress=percent, step=step_description)

    try:
        result = analyze_repository(github_url, progress_callback=on_progress)
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            step="Repository analysis complete.",
            result=result,
        )
        logger.info(f"Ingestion task {task_id} completed successfully for {github_url}")
    except Exception as exc:
        logger.error(f"Ingestion task {task_id} failed for {github_url}: {exc}", exc_info=True)
        task_manager.update_task(
            task_id,
            status="failed",
            progress=100,
            step=f"Analysis failed: {exc}",
            error=str(exc),
        )
