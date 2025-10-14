import React, { useState } from "react";
import { updateTask, deleteTask, addComment } from "../api/task";
import Comment from "./Comment";

function Task({ task, refresh }) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(task.title);
  const [commentText, setCommentText] = useState("");

  const saveEdit = async () => {
    await updateTask(task._id, title);
    setEditing(false);
    refresh();
  };

  const removeTask = async () => {
    await deleteTask(task._id);
    refresh();
  };

  const addNewComment = async () => {
    if (commentText.trim() === "") return;
    await addComment(task._id, commentText);
    setCommentText("");
    refresh();
  };

  return (
    <div style={{ border: "1px solid gray", padding: "10px", marginBottom: "10px" }}>
      {editing ? (
        <>
          <input value={title} onChange={(e) => setTitle(e.target.value)} />
          <button onClick={saveEdit}>Save</button>
          <button onClick={() => setEditing(false)}>Cancel</button>
        </>
      ) : (
        <>
          <h4>{task.title}</h4>
          <button onClick={() => setEditing(true)}>Edit</button>
          <button onClick={removeTask}>Delete</button>
        </>
      )}

      <div style={{ marginTop: "10px" }}>
        <input
          placeholder="Add comment"
          value={commentText}
          onChange={(e) => setCommentText(e.target.value)}
        />
        <button onClick={addNewComment}>Add Comment</button>
      </div>

      <div>
        {task.comments?.map((c) => (
          <Comment key={c._id} comment={c} taskId={task._id} refresh={refresh} />
        ))}
      </div>
    </div>
  );
}

export default Task;
