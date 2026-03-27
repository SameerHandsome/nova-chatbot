/**
 * api.js — Central API client.
 * Every fetch() call in the app lives here.
 * All methods auto-attach Authorization header and handle 401 redirects.
 */

const BASE_URL = 'http://localhost:8000';

function _getToken() {
  return localStorage.getItem('nova_token');
}

function _headers(extra = {}) {
  const token = _getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...extra,
  };
}

async function _request(method, path, body = null) {
  const opts = {
    method,
    headers: _headers(),
  };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(`${BASE_URL}${path}`, opts);

  // Auto-redirect on auth failure
  if (res.status === 401) {
    localStorage.removeItem('nova_token');
    window.location.href = '/login.html';
    return;
  }

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || data.detail || 'Request failed');
  }

  return data;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export const apiClient = {

  async register({ email, username, password }) {
    return _request('POST', '/auth/register', { email, username, password });
  },

  async login({ email, password }) {
    return _request('POST', '/auth/login', { email, password });
  },

  githubLoginUrl() {
    return `${BASE_URL}/auth/github`;
  },

  async me() {
    return _request('GET', '/auth/me');
  },

  // ── Chat ────────────────────────────────────────────────────────────────────

  async chat({ session_id, text, image_b64, image_media_type, audio_b64 }) {
    return _request('POST', '/chat', {
      session_id,
      text:             text       || null,
      image_b64:        image_b64  || null,
      image_media_type: image_media_type || null,
      audio_b64:        audio_b64  || null,
    });
  },

  // ── Sessions ────────────────────────────────────────────────────────────────

  async endSession(session_id) {
    return _request('POST', '/sessions/end', { session_id });
  },

  async getSessions(limit = 20, offset = 0) {
    return _request('GET', `/sessions?limit=${limit}&offset=${offset}`);
  },

  async getSessionMessages(session_id) {
    return _request('GET', `/sessions/${session_id}/messages`);
  },

  // ── Preferences ─────────────────────────────────────────────────────────────

  async getPreferences() {
    return _request('GET', '/preferences');
  },

};