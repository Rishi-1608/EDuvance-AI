// ═══════════════════════════════════════════════════════
//  AcademIQ — API Service Layer
//  Base URL configurable via env or defaults to localhost
// ═══════════════════════════════════════════════════════

const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const token = localStorage.getItem('eduvance_ai_token');
  if (token) {
    options.headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    };
  }

  const res = await fetch(`${BASE_URL}${path}`, options);

  if (res.status === 401) {
    localStorage.removeItem('eduvance_ai_token');
    window.location.reload();
  }

  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try { msg = JSON.parse(text)?.detail || text; } catch { }
    throw new ApiError(res.status, msg);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  if (ct.includes('application/pdf')) return res.blob();
  return res.text();
}

export async function login(username, password) {
  const form = new URLSearchParams();
  form.append('username', username);
  form.append('password', password);
  return request('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form
  });
}

export async function register(username, email, password) {
  return request('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password })
  });
}

export async function getMe() {
  return request('/auth/me');
}

// ── Upload Endpoints ───────────────────────────────────

export async function uploadVideo(files) {
  const form = new FormData();
  const keys = ['file1', 'file2', 'file3'];
  files.forEach((f, i) => form.append(keys[i], f));
  return request('/upload/video', { method: 'POST', body: form });
}

export async function uploadImage(files) {
  const form = new FormData();
  const keys = ['file1', 'file2', 'file3'];
  files.forEach((f, i) => form.append(keys[i], f));
  return request('/upload/image', { method: 'POST', body: form });
}

export async function uploadAudio(files) {
  const form = new FormData();
  const keys = ['file1', 'file2', 'file3'];
  files.forEach((f, i) => form.append(keys[i], f));
  return request('/upload/audio', { method: 'POST', body: form });
}

export async function uploadDocument(files) {
  const form = new FormData();
  const keys = ['file1', 'file2', 'file3'];
  files.forEach((f, i) => form.append(keys[i], f));
  return request('/upload/document', { method: 'POST', body: form });
}

// ── Pipeline Control ───────────────────────────────────

export async function getStatus() {
  return request('/status');
}

export async function stopPipeline() {
  return request('/stop', { method: 'POST' });
}

export async function clearResults() {
  return request('/results', { method: 'DELETE' });
}

export async function deleteLecture(stem) {
  return request(`/api/v1/lecture/${stem}`, { method: 'DELETE' });
}

// ── Results ────────────────────────────────────────────

export async function getVideoResults() {
  return request('/results/video');
}

export async function getImageResults() {
  return request('/results/image');
}

export async function getAudioResults() {
  return request('/results/audio');
}

export async function getStudyNotes(stem) {
  return request(`/results/notes/${stem}`);
}

export async function getPdfUrl(stem) {
  return `${BASE_URL}/results/pdf/${stem}`;
}

export async function getFlashcards(stem) {
  return request(`/results/flashcards/${stem}`);
}

export async function getQuiz(stem) {
  return request(`/results/quiz/${stem}`);
}

export async function generateFlashcardsAndQuiz(stem) {
  return request(`/generate/flashcards/${stem}`, { method: 'POST' });
}

export async function getFrames(stem) {
  return request(`/results/frames/${stem}`);
}

export function getVideoUrl(stem) {
  const token = localStorage.getItem('eduvance_ai_token');
  return `${BASE_URL}/video/${stem}${token ? `?token=${token}` : ''}`;
}

export async function getDashboardStats() {
  return request('/dashboard/stats');
}

export async function getLatestFrames(n = 10) {
  return request(`/results/latest?n=${n}`);
}

export async function generateStudyPlan(stem) {
  return request(`/dashboard/generate-plan/${stem}`, { method: 'POST' });
}

export async function generateMindMap(stem) {
  return request(`/dashboard/generate-mindmap/${stem}`, { method: 'POST' });
}

export async function getDiagnostics() {
  return request('/diagnostics');
}

export async function submitQuizSession(stem, totalQuestions, correctAnswers) {
  return request(`/quiz/session/${stem}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      total_questions: totalQuestions,
      correct_answers: correctAnswers
    })
  });
}

export async function downloadPdf(stem) {
  const blob = await request(`/results/pdf/${stem}`);
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${stem}.pdf`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
}

export async function sendChatMessage(messages) {
  return request('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  });
}

// ── Live Lecture ───────────────────────────────────────

export async function startLiveSession(title) {
  return request('/live/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  });
}

export async function endLiveSession(sessionId) {
  return request(`/live/end/${sessionId}`, { method: 'POST' });
}

export async function cancelLiveSession(sessionId) {
  return request(`/live/cancel/${sessionId}`, { method: 'POST' });
}

export async function getLiveStatus(sessionId) {
  return request(`/live/status/${sessionId}`);
}

export async function getLiveTranscript(sessionId) {
  return request(`/live/transcript/${sessionId}`);
}

export async function getLiveSessions() {
  return request('/live/sessions');
}

export { BASE_URL, request };
