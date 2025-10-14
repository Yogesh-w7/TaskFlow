from flask import Blueprint, request, jsonify
from bson import ObjectId
from datetime import datetime
from ..db import mongo
from ..models import comment_to_dict

comment_bp = Blueprint("comments", __name__)

# GET all comments for a task
@comment_bp.route("/tasks/<task_id>/comments", methods=["GET"])
def get_comments(task_id):
    comments = mongo.db.comments.find({"task_id": task_id})
    return jsonify([comment_to_dict(c) for c in comments]), 200

# POST add comment
@comment_bp.route("/tasks/<task_id>/comments", methods=["POST"])
def add_comment(task_id):
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    comment = {
        "task_id": task_id,
        "author": data.get("author", "Anonymous"),
        "text": text,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = mongo.db.comments.insert_one(comment)
    comment["_id"] = result.inserted_id
    return jsonify(comment_to_dict(comment)), 201

# PUT/PATCH update comment
@comment_bp.route("/tasks/<task_id>/comments/<id>", methods=["PUT", "PATCH"])
def edit_comment(task_id, id):
    data = request.get_json()
    if not data.get("text"):
        return jsonify({"error": "Text required"}), 400
    updated = mongo.db.comments.update_one(
        {"_id": ObjectId(id), "task_id": task_id},
        {"$set": {"text": data["text"], "author": data.get("author"), "updated_at": datetime.utcnow()}}
    )
    if updated.matched_count == 0:
        return jsonify({"error": "Comment not found"}), 404
    c = mongo.db.comments.find_one({"_id": ObjectId(id)})
    return jsonify(comment_to_dict(c)), 200

# DELETE comment
@comment_bp.route("/tasks/<task_id>/comments/<id>", methods=["DELETE"])
def delete_comment(task_id, id):
    result = mongo.db.comments.delete_one({"_id": ObjectId(id), "task_id": task_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Comment not found"}), 404
    return "", 204
