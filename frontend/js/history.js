/**
 * history.js — Session history with two-panel layout like Claude.
 * Left: sessions list. Right: messages for selected session.
 */

import { apiClient }               from './api.js';
import { renderNavbar, showToast } from './components.js';

let allSessions = [];

document.addEventListener('DOMContentLoaded', async () => {
  renderNavbar('history');
  await loadSessions();
});

// ── Load sessions list ────────────────────────────────────────────────────────

async function loadSessions() {
  const list = document.getElementById('sessionsGrid');
  list.innerHTML = `<div style="padding:16px;color:var(--muted);font-size:13px">Loading…</div>`;

  try {
    allSessions = await apiClient.getSessions();

    if (!allSessions.length) {
      list.innerHTML = `
        <div style="padding:24px 16px;text-align:center;color:var(--muted);font-size:13px">
          <div style="font-size:32px;margin-bottom:8px">📋</div>
          No sessions yet.<br>Start chatting first.
        </div>`;
      return;
    }

    list.innerHTML = '';
    allSessions.forEach((s, i) => {
      const item = buildSessionItem(s);
      list.appendChild(item);
      if (i === 0) {
        item.classList.add('selected');
        loadMessages(s.id);
      }
    });

  } catch (err) {
    showToast(err.message, 'error');
    list.innerHTML = `<div style="padding:16px;color:var(--danger);font-size:13px">Failed to load.</div>`;
  }
}

// ── Build session list item ───────────────────────────────────────────────────

function buildSessionItem(s) {
  const item  = document.createElement('div');
  item.className = 'session-item';
  item.dataset.id = s.id;

  const date   = new Date(s.created_at).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric',
  });
  const title  = s.title ? s.title.substring(0, 50) : 'Untitled session';
  const active = s.is_active;

  item.innerHTML = `
    <div class="session-item-title">${esc(title)}</div>
    <div class="session-item-meta">
      <span class="session-dot ${active ? 'active' : 'ended'}"></span>
      <span>${date}</span>
      <span>${active ? 'Active' : 'Ended'}</span>
    </div>`;

  item.addEventListener('click', () => {
    // Deselect all
    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('selected'));
    item.classList.add('selected');
    loadMessages(s.id);
  });

  return item;
}

// ── Load messages for a session ───────────────────────────────────────────────

async function loadMessages(sessionId) {
  const panel = document.getElementById('messagesPanel');
  const title = document.getElementById('messagesPanelTitle');
  const s     = allSessions.find(x => x.id === sessionId);

  if (title) title.textContent = s?.title || 'Session Messages';

  panel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:center;height:120px;color:var(--muted);font-size:13px">
      Loading messages…
    </div>`;

  try {
    const messages = await apiClient.getSessionMessages(sessionId);

    if (!messages.length) {
      panel.innerHTML = `
        <div style="text-align:center;padding:48px 20px;color:var(--muted);font-size:13px">
          No messages in this session.
        </div>`;
      return;
    }

    panel.innerHTML = '';
    messages.forEach(msg => {
      const userEl = buildUserBubble(msg);
      if (userEl) panel.appendChild(userEl);
      if (msg.response_text) panel.appendChild(buildBotBubble(msg));
    });

    panel.scrollTop = panel.scrollHeight;

  } catch (err) {
    panel.innerHTML = `
      <div style="color:var(--danger);padding:20px;font-size:13px">
        Error: ${esc(err.message)}
      </div>`;
  }
}

// ── Build message bubbles ─────────────────────────────────────────────────────

function buildUserBubble(msg) {
  const parts = [];
  if (msg.has_audio && msg.audio_transcript) {
    parts.push(`<div style="font-size:13px"><span style="font-family:var(--font-mono);font-size:10px;color:var(--accent2)">🎙 Voice:</span> ${esc(msg.audio_transcript)}</div>`);
  }
  if (msg.has_text && msg.raw_text) {
    parts.push(`<div style="font-size:14px;line-height:1.6">${esc(msg.raw_text)}</div>`);
  }
  if (msg.has_image) {
    parts.push(`<div style="font-family:var(--font-mono);font-size:10px;color:var(--warn)">🖼 Image attached</div>`);
  }
  if (!parts.length) return null;

  const el = document.createElement('div');
  el.className = 'hist-message hist-user';
  el.innerHTML = `
    <div class="hist-avatar" style="background:linear-gradient(135deg,var(--accent),var(--accent2))">👤</div>
    <div class="hist-bubble hist-bubble-user">${parts.join('')}</div>`;
  return el;
}

function buildBotBubble(msg) {
  const tools = (msg.tools_called || [])
    .map(t => `<span style="font-family:var(--font-mono);font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid rgba(108,255,206,0.4);color:var(--accent3);background:rgba(108,255,206,0.07);margin-right:4px">🔧 ${t}</span>`)
    .join('');

  const el = document.createElement('div');
  el.className = 'hist-message hist-bot';
  el.innerHTML = `
    <div class="hist-avatar" style="background:var(--surface2);border:1px solid var(--border)">✦</div>
    <div class="hist-bubble hist-bubble-bot">
      <div style="font-size:14px;line-height:1.65">${fmt(msg.response_text)}</div>
      ${tools ? `<div style="margin-top:8px">${tools}</div>` : ''}
    </div>`;
  return el;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmt(text) {
  return esc(text || '')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em>$1</em>')
    .replace(/`(.*?)`/g,       '<code style="font-family:var(--font-mono);background:var(--surface2);padding:1px 5px;border-radius:3px">$1</code>')
    .replace(/\n/g,            '<br>');
}