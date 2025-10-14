import pytest
from app import create_app
from app.db import mongo

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_create_and_get_comment(client):
    # Create a task
    task_res = client.post("/api/tasks/", json={"title": "My Task"})
    assert task_res.status_code == 201
    task_id = task_res.get_json()["_id"]

    # Create comment
    res = client.post(f"/api/tasks/{task_id}/comments", json={"author": "A", "text": "Hello"})
    assert res.status_code == 201

    # List comments
    list_res = client.get(f"/api/tasks/{task_id}/comments")
    assert list_res.status_code == 200
    data = list_res.get_json()
    assert len(data) == 1
    assert data[0]["text"] == "Hello"
