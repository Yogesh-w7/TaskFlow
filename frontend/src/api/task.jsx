import axios from "axios";

const API_URL = "http://127.0.0.1:5000/api";

export const getTasks = async () => {
  const res = await axios.get(`${API_URL}/tasks/`);
  return res.data;
};

export const addTask = async (title) => {
  const res = await axios.post(`${API_URL}/tasks/`, { title });
  return res.data;
};

export const updateTask = async (taskId, title) => {
  const res = await axios.put(`${API_URL}/tasks/${taskId}`, { title });
  return res.data;
};

export const deleteTask = async (taskId) => {
  const res = await axios.delete(`${API_URL}/tasks/${taskId}`);
  return res.data;
};

export const addComment = async (taskId, text) => {
  const res = await axios.post(`${API_URL}/tasks/${taskId}/comments`, { text });
  return res.data;
};

export const updateComment = async (taskId, commentId, text) => {
  const res = await axios.put(`${API_URL}/tasks/${taskId}/comments/${commentId}`, { text });
  return res.data;
};

export const deleteComment = async (taskId, commentId) => {
  const res = await axios.delete(`${API_URL}/tasks/${taskId}/comments/${commentId}`);
  return res.data;
};
