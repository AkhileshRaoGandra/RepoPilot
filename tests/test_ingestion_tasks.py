"""Unit tests for asynchronous repository ingestion tasks and polling endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ingestion.tasks import TaskManager, IngestionTask
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_task_manager_lifecycle() -> None:
    tm = TaskManager(max_history=5)
    task = tm.create_task("https://github.com/octocat/Hello-World")
    assert task.task_id is not None
    assert task.status == "queued"
    assert task.progress == 0

    # Retrieve
    found = tm.get_task(task.task_id)
    assert found is not None
    assert found.github_url == "https://github.com/octocat/Hello-World"

    # Update progress
    tm.update_task(task.task_id, status="running", progress=50, step="Parsing AST...")
    updated = tm.get_task(task.task_id)
    assert updated is not None
    assert updated.status == "running"
    assert updated.progress == 50
    assert updated.step == "Parsing AST..."

    # Complete
    tm.update_task(task.task_id, status="completed", progress=100, result={"files_found": 10})
    completed = tm.get_task(task.task_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.result == {"files_found": 10}
    assert completed.completed_at is not None


def test_task_manager_evicts_oldest_when_limit_reached() -> None:
    tm = TaskManager(max_history=3)
    t1 = tm.create_task("https://github.com/a/1")
    t2 = tm.create_task("https://github.com/a/2")
    t3 = tm.create_task("https://github.com/a/3")
    t4 = tm.create_task("https://github.com/a/4")

    # t1 should have been evicted
    assert tm.get_task(t1.task_id) is None
    assert tm.get_task(t2.task_id) is not None
    assert tm.get_task(t3.task_id) is not None
    assert tm.get_task(t4.task_id) is not None


def test_analyze_endpoint_queues_task(client: TestClient) -> None:
    resp = client.post(
        "/repository/analyze",
        json={"github_url": "https://github.com/octocat/Hello-World"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert data["progress"] == 0
    assert "message" in data

    # Poll status
    task_id = data["task_id"]
    status_resp = client.get(f"/repository/tasks/{task_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["task_id"] == task_id
    assert status_data["github_url"] == "https://github.com/octocat/Hello-World"


def test_task_status_not_found(client: TestClient) -> None:
    resp = client.get("/repository/tasks/non-existent-uuid-12345")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_analyze_endpoint_validates_url_before_queuing(client: TestClient) -> None:
    resp = client.post(
        "/repository/analyze",
        json={"github_url": "https://not-github.com/invalid"},
    )
    assert resp.status_code == 422
    assert "detail" in resp.json()
