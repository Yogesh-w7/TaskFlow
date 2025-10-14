import React, { useEffect, useState } from "react";
import axios from "axios";
import GlitchText from "./component/GlitchText";
import "./App.css";

function App() {
  const [tasks, setTasks] = useState([]);
  const [title, setTitle] = useState("");
  const [priority, setPriority] = useState("Medium");
  const [comment, setComment] = useState("");
  const [selectedTask, setSelectedTask] = useState(null);
  const [comments, setComments] = useState([]);
  const [editingTask, setEditingTask] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [search, setSearch] = useState("");
  const [showDeletePopup, setShowDeletePopup] = useState(false);
  const [deleteTaskId, setDeleteTaskId] = useState(null);

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api";



  useEffect(() => {
    loadTasks();
  }, []);

  const loadTasks = async () => {
    try {
      const res = await axios.get(`${BASE_URL}/tasks/`);
      const safeTasks = res.data.map((t) => ({
        ...t,
        title: t.title || "",
        priority: t.priority || "Medium",
        completed: t.completed || false,
      }));
      setTasks(safeTasks);
    } catch (err) {
      console.error("Error loading tasks:", err);
    }
  };

  const addTask = async () => {
    if (!title.trim()) return alert("Task title required");
    try {
      await axios.post(`${BASE_URL}/tasks/`, { title: title.trim(), priority });
      setTitle("");
      setPriority("Medium");
      loadTasks();
    } catch (err) {
      console.error("Error adding task:", err);
    }
  };

  const confirmDeleteTask = (id) => {
    setDeleteTaskId(id);
    setShowDeletePopup(true);
  };

  const deleteTask = async () => {
    try {
      await axios.delete(`${BASE_URL}/tasks/${deleteTaskId}`);
      if (selectedTask === deleteTaskId) {
        setSelectedTask(null);
        setComments([]);
      }
      setShowDeletePopup(false);
      setDeleteTaskId(null);
      loadTasks();
    } catch (err) {
      console.error("Error deleting task:", err);
    }
  };

  const editTask = async (id) => {
    if (!editTitle.trim()) return alert("New title required");
    try {
      await axios.put(`${BASE_URL}/tasks/${id}`, { title: editTitle.trim() });
      setEditingTask(null);
      setEditTitle("");
      loadTasks();
    } catch (err) {
      console.error("Error editing task:", err);
    }
  };

  const toggleComplete = async (id, completed) => {
    try {
      await axios.put(`${BASE_URL}/tasks/${id}`, { completed: !completed });
      loadTasks();
    } catch (err) {
      console.error("Error toggling completion:", err);
    }
  };

  const loadComments = async (taskId) => {
    try {
      const res = await axios.get(`${BASE_URL}/tasks/${taskId}/comments`);
      setComments(res.data || []);
      setSelectedTask(taskId);
    } catch (err) {
      console.error("Error loading comments:", err);
    }
  };

  const addComment = async () => {
    if (!selectedTask || !comment.trim()) return;
    try {
      await axios.post(`${BASE_URL}/tasks/${selectedTask}/comments`, { text: comment.trim() });
      setComment("");
      loadComments(selectedTask);
    } catch (err) {
      console.error("Error adding comment:", err);
    }
  };

  const filteredTasks = tasks.filter((t) => (t.title || "").toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="container futuristic">
      <GlitchText text="Task Manager Pro" />

      <div className="input-box">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Enter task title"
        />
        <select value={priority} onChange={(e) => setPriority(e.target.value)}>
          <option value="Low">Low</option>
          <option value="Medium">Medium</option>
          <option value="High">High</option>
        </select>
        <button onClick={addTask}>Add Task</button>
      </div>

      <input
        className="search-bar"
        placeholder="🔍 Search tasks..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <ul className="task-list">
        {filteredTasks.length === 0 ? (
          <p className="empty">No tasks found. Add your first task 🚀</p>
        ) : (
          filteredTasks.map((t) => (
            <li key={t._id} className={`task-item ${t.completed ? "done" : ""}`}>
              {editingTask === t._id ? (
                <div className="edit-task">
                  <input
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    className="edit-input"
                  />
                  <select
                    value={t.priority}
                    onChange={(e) => {
                      t.priority = e.target.value;
                      setTasks([...tasks]);
                    }}
                    className="edit-select"
                  >
                    <option value="Low">Low</option>
                    <option value="Medium">Medium</option>
                    <option value="High">High</option>
                  </select>
                  <button onClick={() => editTask(t._id)}>Save</button>
                  <button onClick={() => setEditingTask(null)}>Cancel</button>
                </div>
              ) : (
                <>
                  <div className="task-info">
                    <span
                      onClick={() => toggleComplete(t._id, t.completed)}
                      className={t.completed ? "completed" : ""}
                    >
                      {t.title}
                    </span>
                    <div className={`priority ${t.priority.toLowerCase()}`}>
                      {t.priority}
                    </div>
                  </div>
                  <div className="meta">
                    <small>📅 {t.createdAt ? new Date(t.createdAt).toLocaleString() : ""}</small>
                  </div>
                  <div className="btn-group">
                    <button onClick={() => loadComments(t._id)}>💬 Comments</button>
                    <button
                      onClick={() => {
                        setEditingTask(t._id);
                        setEditTitle(t.title);
                      }}
                    >
                      ✏️ Edit
                    </button>
                    <button onClick={() => confirmDeleteTask(t._id)}>🗑️ Delete</button>
                  </div>
                </>
              )}
            </li>
          ))
        )}
      </ul>

      {selectedTask && (
        <div className="comment-section">
          <h3>💭 Comments</h3>
          <ul>
            {comments.length === 0 ? (
              <p>No comments yet.</p>
            ) : (
              comments.map((c, i) => <li key={i}>{c.text}</li>)
            )}
          </ul>
          <div className="input-box">
            <input
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Add a comment"
            />
            <button onClick={addComment}>Add</button>
          </div>
        </div>
      )}

      {showDeletePopup && (
        <div className="delete-popup">
          <div className="popup-content">
            <p>Are you sure you want to delete this task?</p>
            <div className="popup-buttons">
              <button onClick={deleteTask}>Yes, Delete</button>
              <button onClick={() => setShowDeletePopup(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
