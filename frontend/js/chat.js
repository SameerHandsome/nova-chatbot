/**
 * chat.js — Full chat page logic.
 */

import { apiClient }               from './api.js';
import { renderNavbar, showToast } from './components.js';

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  renderNavbar('chat');
  bindEvents();
  textInput.focus();
});

window.addEventListener('beforeunload', () => {
  if (sessionId) {
    navigator.sendBeacon(
      '/sessions/end',
      new Blob([JSON.stringify({ session_id: sessionId })], { type: 'application/json' })
    );
  }
});

// ── State ─────────────────────────────────────────────────────────────────────

let sessionId    = null;
let pendingImage = null;
let pendingAudio = null;
let mediaRecorder= null;
let audioChunks  = [];
let isRecording  = false;
let isSending    = false;

// ── DOM ───────────────────────────────────────────────────────────────────────

const chatArea   = document.getElementById('chatArea');
const textInput  = document.getElementById('textInput');
const sendBtn    = document.getElementById('sendBtn');
const audioBtn   = document.getElementById('audioBtn');
const imageInput = document.getElementById('imageInput');
const previewRow = document.getElementById('previewRow');
const welcome    = document.getElementById('welcomeScreen');
const endBtn     = document.getElementById('endSessionBtn');

// ── Events ────────────────────────────────────────────────────────────────────

function bindEvents() {
  sendBtn.addEventListener('click', sendMessage);
  audioBtn.addEventListener('click', toggleRecording);
  document.getElementById('imgBtn').addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', handleImageSelect);
  textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  textInput.addEventListener('input', () => autoResize(textInput));
  endBtn?.addEventListener('click', endSession);
}

// ── Image ─────────────────────────────────────────────────────────────────────

function handleImageSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { showToast('Image too large (max 5MB)', 'error'); return; }
  const reader = new FileReader();
  reader.onload = ev => {
    const url    = ev.target.result;
    pendingImage = { b64: url.split(',')[1], mediaType: file.type, url };
    renderPreview();
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

// ── Audio ─────────────────────────────────────────────────────────────────────

async function toggleRecording() {
  if (isRecording) { stopRecording(); return; }
  try {
    const stream  = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks   = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob   = new Blob(audioChunks, { type: 'audio/webm' });
      pendingAudio = { b64: await blobToBase64(blob) };
      stream.getTracks().forEach(t => t.stop());
      renderPreview();
    };
    mediaRecorder.start();
    isRecording = true;
    audioBtn.classList.add('recording');
    audioBtn.textContent = '⏹';
    audioBtn.title = 'Click to stop';
  } catch { showToast('Microphone access denied', 'error'); }
}

function stopRecording() {
  mediaRecorder?.stop();
  isRecording = false;
  audioBtn.classList.remove('recording');
  audioBtn.textContent = '🎤';
  audioBtn.title = 'Record voice';
}

function blobToBase64(blob) {
  return new Promise(r => {
    const fr = new FileReader();
    fr.onload = () => r(fr.result.split(',')[1]);
    fr.readAsDataURL(blob);
  });
}

// ── Preview ───────────────────────────────────────────────────────────────────

function renderPreview() {
  previewRow.innerHTML = '';
  let any = false;
  if (pendingImage) {
    any = true;
    previewRow.insertAdjacentHTML('beforeend', `
      <div class="preview-item img-item">
        <img src="${pendingImage.url}" class="preview-thumb" />
        <button class="preview-remove" onclick="window._rmImg()">✕</button>
      </div>`);
  }
  if (pendingAudio) {
    any = true;
    previewRow.insertAdjacentHTML('beforeend', `
      <div class="preview-item">
        🎙 Voice recorded
        <button class="preview-remove" onclick="window._rmAud()">✕</button>
      </div>`);
  }
  previewRow.style.display = any ? 'flex' : 'none';
}

window._rmImg = () => { pendingImage = null; renderPreview(); };
window._rmAud = () => { pendingAudio = null; renderPreview(); };

// ── Send ──────────────────────────────────────────────────────────────────────

async function sendMessage() {
  if (isSending) return;
  if (isRecording) { stopRecording(); await new Promise(r => setTimeout(r, 200)); }

  const text = textInput.value.trim();
  if (!text && !pendingImage && !pendingAudio) {
    showToast('Add a message, image, or voice note', 'error');
    return;
  }

  // Hide welcome screen completely and remove it from flow
  if (welcome) {
    welcome.style.display = 'none';
    welcome.style.height  = '0';
  }

  isSending        = true;
  sendBtn.disabled = true;

  const snap = {
    text:  text || null,
    image: pendingImage ? { ...pendingImage } : null,
    audio: pendingAudio ? { ...pendingAudio } : null,
  };

  textInput.value = '';
  autoResize(textInput);
  pendingImage = null;
  pendingAudio = null;
  renderPreview();

  appendUserMessage(snap);
  const typingId = appendTyping();

  try {
    const data = await apiClient.chat({
      session_id:       sessionId,
      text:             snap.text,
      image_b64:        snap.image?.b64    || null,
      image_media_type: snap.image?.mediaType || null,
      audio_b64:        snap.audio?.b64    || null,
    });

    sessionId = data.session_id;
    removeTyping(typingId);
    appendBotMessage(data);

  } catch (err) {
    removeTyping(typingId);
    appendErrorMessage(err.message);
    showToast(err.message, 'error');
  }

  isSending        = false;
  sendBtn.disabled = false;
  textInput.focus();
}

// ── Session End ───────────────────────────────────────────────────────────────

async function endSession() {
  if (!sessionId) { showToast('No active session', 'error'); return; }
  try {
    const res = await apiClient.endSession(sessionId);
    sessionId = null;
    showToast('Session ended — preferences saved!', 'success');
    appendSystemMessage(`📝 Session ended. Summary: ${res.summary}`);
  } catch (err) { showToast(err.message, 'error'); }
}

// ── Message Rendering ─────────────────────────────────────────────────────────

function appendUserMessage(snap) {
  const chips = [];
  if (snap.audio) chips.push(`<span class="att-chip audio">🎙 Voice</span>`);
  if (snap.text)  chips.push(`<span class="att-chip" style="color:var(--accent3);border-color:rgba(108,255,206,0.4);background:rgba(108,255,206,0.07);font-family:var(--font-mono);font-size:10px;padding:3px 9px;border-radius:4px;border:1px solid">⌨ ${esc(snap.text)}</span>`);

  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `
    <div class="avatar">👤</div>
    <div class="bubble-wrap">
      ${snap.image ? `<img src="${snap.image.url}" class="att-img-preview" alt="attached" />` : ''}
      ${chips.length ? `<div class="att-row" style="margin-bottom:4px">${chips.join('')}</div>` : ''}
      <span class="timestamp">${now()}</span>
    </div>`;
  chatArea.appendChild(el);
  scrollBottom();
}

function appendBotMessage(data) {
  const tools = (data.tools_called || [])
    .map(t => `<span class="att-chip tool">🔧 ${t}</span>`).join('');

  const meta = data.transcribed_text
    ? `<div class="meta-block" style="margin-top:6px;font-family:var(--font-mono);font-size:11px;color:var(--muted);background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:6px 10px">🎙 Heard: ${esc(data.transcribed_text)}</div>` : '';

  const cacheTag = data.cache_hit
    ? `<span style="font-family:var(--font-mono);font-size:9px;color:var(--accent3);background:rgba(108,255,206,0.07);border:1px solid rgba(108,255,206,0.3);padding:2px 6px;border-radius:3px;margin-left:6px">CACHED</span>` : '';

  const el = document.createElement('div');
  el.className = 'message bot';
  el.innerHTML = `
    <div class="avatar">✦</div>
    <div class="bubble-wrap">
      <div class="bubble">${fmt(data.response)}${cacheTag}</div>
      ${tools ? `<div class="att-row" style="margin-top:6px">${tools}</div>` : ''}
      ${meta}
      <span class="timestamp">${now()} · ${data.latency_ms || 0}ms</span>
    </div>`;
  chatArea.appendChild(el);
  scrollBottom();
}

function appendTyping() {
  const id = `t${Date.now()}`;
  const el = document.createElement('div');
  el.className = 'message bot';
  el.id = id;
  el.innerHTML = `
    <div class="avatar">✦</div>
    <div class="bubble-wrap">
      <div class="bubble">
        <div class="dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  chatArea.appendChild(el);
  scrollBottom();
  return id;
}

function removeTyping(id) { document.getElementById(id)?.remove(); }

function appendErrorMessage(msg) {
  const el = document.createElement('div');
  el.className = 'message bot';
  el.innerHTML = `
    <div class="avatar">✦</div>
    <div class="bubble-wrap">
      <div class="bubble" style="color:var(--danger);border-color:rgba(255,77,109,0.3)">⚠ ${esc(msg)}</div>
    </div>`;
  chatArea.appendChild(el);
  scrollBottom();
}

function appendSystemMessage(text) {
  const el = document.createElement('div');
  el.style.cssText = 'text-align:center;font-size:12px;color:var(--muted);font-family:var(--font-mono);padding:6px 0';
  el.textContent = text;
  chatArea.appendChild(el);
  scrollBottom();
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function scrollBottom() {
  requestAnimationFrame(() => { chatArea.scrollTop = chatArea.scrollHeight; });
}

function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmt(text) {
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em>$1</em>')
    .replace(/`(.*?)`/g,       '<code style="font-family:var(--font-mono);background:var(--surface2);padding:1px 5px;border-radius:3px">$1</code>')
    .replace(/\n/g,            '<br>');
}