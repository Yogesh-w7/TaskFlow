from datetime import datetime

def task_to_dict(task):
    return {
        "_id": str(task["_id"]),
        "title": task.get("title", ""),
        "description": task.get("description", ""),
        "priority": task.get("priority", "Medium"),  # default Medium
        "completed": task.get("completed", False),
        "createdAt": task.get("created_at") if task.get("created_at") else datetime.utcnow().isoformat()
    }

def comment_to_dict(comment):
    return {
        "_id": str(comment["_id"]),
        "taskId": str(comment.get("task_id")),  # ensure string for frontend
        "author": comment.get("author", "Anonymous"),
        "text": comment.get("text", ""),
        "createdAt": comment.get("created_at") if comment.get("created_at") else datetime.utcnow().isoformat(),
        "updatedAt": comment.get("updated_at") if comment.get("updated_at") else datetime.utcnow().isoformat()
    }
