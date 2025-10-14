from flask import Blueprint, request, jsonify
from bson import ObjectId
from datetime import datetime
from ..db import mongo
from ..models import task_to_dict

task_bp = Blueprint("tasks", __name__)

@task_bp.route("/", methods=["GET"])
def list_tasks():
    tasks = mongo.db.tasks.find()
    return jsonify([task_to_dict(t) for t in tasks]), 200

@task_bp.route("/", methods=["POST"])
def add_task():
    data = request.get_json()
    if not data.get("title"):
        return jsonify({"error": "Title required"}), 400

    task = {
        "title": data["title"],
        "description": data.get("description", ""),
        "priority": data.get("priority", "Medium"),  # default Medium
        "completed": False,  # default not completed
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = mongo.db.tasks.insert_one(task)
    task["_id"] = result.inserted_id
    return jsonify(task_to_dict(task)), 201

@task_bp.route("/<id>", methods=["PUT"])
def edit_task(id):
    data = request.get_json()
    if not data.get("title"):
        return jsonify({"error": "Title required"}), 400

    update_fields = {
        "title": data["title"],
        "description": data.get("description", ""),
        "updated_at": datetime.utcnow()
    }

    # Optional: update priority and completed if provided
    if "priority" in data:
        update_fields["priority"] = data["priority"]
    if "completed" in data:
        update_fields["completed"] = data["completed"]

    updated = mongo.db.tasks.update_one(
        {"_id": ObjectId(id)},
        {"$set": update_fields}
    )

    if updated.matched_count == 0:
        return jsonify({"error": "Task not found"}), 404

    task = mongo.db.tasks.find_one({"_id": ObjectId(id)})
    return jsonify(task_to_dict(task)), 200

@task_bp.route("/<id>", methods=["DELETE"])
def delete_task(id):
    result = mongo.db.tasks.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        return jsonify({"error": "Task not found"}), 404

    # Delete associated comments
    mongo.db.comments.delete_many({"task_id": id})
    return "", 204
