import React, { useState } from "react";
import { updateComment, deleteComment } from "../api/task";

function Comment({ comment, taskId, refresh }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(comment.text);

  const saveEdit = async () => {
    await updateComment(taskId, comment._id, text);
    setEditing(false);
    refresh();
  };

  const removeComment = async () => {
    await deleteComment(taskId, comment._id);
    refresh();
  };

  return (
    <div style={{ marginLeft: "20px" }}>
      {editing ? (
        <>
          <input value={text} onChange={(e) => setText(e.target.value)} />
          <button onClick={saveEdit}>Save</button>
          <button onClick={() => setEditing(false)}>Cancel</button>
        </>
      ) : (
        <>
          <span>{comment.text}</span>
          <button onClick={() => setEditing(true)}>Edit</button>
          <button onClick={removeComment}>Delete</button>
        </>
      )}
    </div>
  );
}

export default Comment;
