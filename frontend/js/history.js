/**
 * history.js — Session history with two-panel layout.
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
    list.innerHTML = `<div style="padding:16px;color:var(--danger)">Failed to load sessions.</div>`;
    showToast('Error loading history', 'error');
  }
}

function buildSessionItem(session) {
  const el = document.createElement('div');
  el.className = 'session-item';
  
  const d = new Date(session.created_at);
  const dateStr = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  el.innerHTML = `
    <div class="session-title">${esc(session.title || 'New Conversation')}</div>
    <div class="session-meta">${dateStr} • ${timeStr}</div>
  `;

  el.addEventListener('click', () => {
    document.querySelectorAll('.session-item').forEach(n => n.classList.remove('selected'));
    el.classList.add('selected');
    loadMessages(session.id);
  });

  return el;
}

// ── Load selected session messages ────────────────────────────────────────────

async function loadMessages(sessionId) {
  const panel = document.getElementById('messagesPanel');
  panel.innerHTML = `
    <div class="panel-placeholder">
      <div class="spinner" style="width:24px;height:24px"></div>
    </div>`;

  try {
    const messages = await apiClient.getSessionMessages(sessionId);
    if (!messages || !messages.length) {
      panel.innerHTML = `<div class="panel-placeholder">No messages in this session.</div>`;
      return;
    }

    panel.innerHTML = '';
    
    messages.forEach(msg => {
      // If the backend happens to send a pure assistant row, render it
      if (msg.role === 'assistant') {
        panel.appendChild(buildBotBubble(msg));
        return;
      }

      // 1. Render the User's input (Fix: mapped to raw_text)
      panel.appendChild(buildUserBubble(msg));
      
      // 2. Render the AI's response from the exact same DB row
      if (msg.response_text || msg.final_response || msg.response) {
        panel.appendChild(buildBotBubble(msg));
      }
    });
    
    // Auto scroll to bottom of history
    requestAnimationFrame(() => {
      panel.scrollTop = panel.scrollHeight;
    });

  } catch (err) {
    panel.innerHTML = `<div class="panel-placeholder" style="color:var(--danger)">Failed to load messages.</div>`;
  }
}

function buildUserBubble(msg) {
  const parts = [];
  
  // FIX: Checks for backend booleans if the full base64 string isn't sent back
  if (msg.has_image || msg.image_b64) {
    parts.push(`<div style="font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:4px">🖼️ Image Attached</div>`);
  }
  if (msg.has_audio || msg.audio_b64) {
    parts.push(`<div style="font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:4px">🎤 Voice Message</div>`);
  }
  
  // FIX: Safely reads raw_text (your actual DB column) instead of message_text
  const textContent = msg.raw_text || msg.message_text || '';
  if (textContent) {
    parts.push(`<div>${fmt(textContent)}</div>`);
  }

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

  // FIX: Safely falls back through all possible backend response key names
  const textContent = msg.response_text || msg.final_response || msg.response || 'No text content available.';

  const el = document.createElement('div');
  el.className = 'hist-message hist-bot';
  el.innerHTML = `
    <div class="hist-avatar" style="background:var(--surface2);border:1px solid var(--border)">✦</div>
    <div class="hist-bubble hist-bubble-bot">
      <div style="font-size:14px;line-height:1.65">${fmt(textContent)}</div>
      ${tools ? `<div style="margin-top:8px">${tools}</div>` : ''}
    </div>`;
  return el;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmt(s) {
  return esc(s).replace(/\n/g, '<br>');
}