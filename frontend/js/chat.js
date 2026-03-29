/**
 * chat.js — Full chat page logic.
 */

import { apiClient, BASE_URL } from './api.js';
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
      `${BASE_URL}/sessions/end`,
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
  sendBtn.addEventListener('click', handleSend);

  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  textInput.addEventListener('input', () => autoResize(textInput));

  document.getElementById('imgBtn').addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', handleImageSelect);

  audioBtn.addEventListener('click', toggleRecording);

  endBtn.addEventListener('click', async () => {
    if (!sessionId) return;
    try {
      await apiClient.endSession(sessionId);
      sessionId = null;
      endBtn.style.display = 'none';
      appendSystemMessage("Session ended. Start typing to begin a new one.");
    } catch (err) {
      showToast("Failed to end session", "error");
    }
  });
}

// ── Core Actions ──────────────────────────────────────────────────────────────

async function handleSend() {
  const text = textInput.value.trim();
  if (!text && !pendingImage && !pendingAudio) return;
  if (isSending) return;

  isSending = true;
  sendBtn.disabled = true;

  if (welcome) welcome.style.display = 'none';

  // Build payloads
  const textPayload  = text;
  const imgPayload   = pendingImage ? pendingImage.b64 : null;
  const imgMime      = pendingImage ? pendingImage.mime : null;
  const audioPayload = pendingAudio ? pendingAudio.b64 : null;

  // Append user UI
  appendUserMessage(text, imgPayload, audioPayload);

  // Reset inputs
  textInput.value = '';
  autoResize(textInput);
  clearPreviews();

  // Show loading
  const loadingId = appendLoading();

  try {
    const data = await apiClient.chat({
      session_id: sessionId,
      text: textPayload,
      image_b64: imgPayload,
      image_media_type: imgMime,
      audio_b64: audioPayload,
    });

    if (data.session_id) {
      sessionId = data.session_id;
      endBtn.style.display = 'block';
    }

    removeMessage(loadingId);

    const botText = data.final_response || data.response_text || data.response || 'No response generated.';
    const toolsUsed = data.tools_called || [];

    appendBotMessage(botText, toolsUsed);

  } catch (err) {
    removeMessage(loadingId);
    appendError(err.message || 'Something went wrong.');
    showToast(err.message, 'error');
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    textInput.focus();
  }
}

// ── Media Handling & Compression ──────────────────────────────────────────────

// FIX: Helper function to compress images before uploading
async function compressImage(dataUrl, maxWidth = 1024) {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = dataUrl;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, maxWidth / img.width);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // Compress to 70% quality JPEG (Solves 10MB payload issue)
      resolve(canvas.toDataURL('image/jpeg', 0.7)); 
    };
  });
}

function handleImageSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (ev) => {
    const dataUrl = ev.target.result;
    
    // FIX: Await compression before setting the state
    const compressedDataUrl = await compressImage(dataUrl);
    const [header, b64] = compressedDataUrl.split(',');
    const mime = header.split(':')[1].split(';')[0];

    pendingImage = { b64, mime, dataUrl: compressedDataUrl };
    renderPreviews();
  };
  reader.readAsDataURL(file);
  e.target.value = '';
}

async function toggleRecording() {
  if (isRecording) {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result.split(',')[1];
        pendingAudio = { b64, url: URL.createObjectURL(blob) };
        renderPreviews();
      };
      reader.readAsDataURL(blob);
      stream.getTracks().forEach(t => t.stop());
      isRecording = false;
      audioBtn.classList.remove('recording');
    };

    mediaRecorder.start();
    isRecording = true;
    audioBtn.classList.add('recording');
  } catch (err) {
    showToast('Microphone access denied', 'error');
  }
}

// ── UI Rendering ──────────────────────────────────────────────────────────────

function renderPreviews() {
  previewRow.innerHTML = '';
  previewRow.style.display = 'none';

  if (pendingImage) {
    previewRow.style.display = 'flex';
    previewRow.innerHTML += `
      <div class="preview-item">
        <img src="${pendingImage.dataUrl}" />
        <button class="remove-btn" onclick="removeImage()">×</button>
      </div>`;
  }

  if (pendingAudio) {
    previewRow.style.display = 'flex';
    previewRow.innerHTML += `
      <div class="preview-item audio-preview">
        🎤 Audio attached
        <button class="remove-btn" onclick="removeAudio()">×</button>
      </div>`;
  }
}

window.removeImage = () => { pendingImage = null; renderPreviews(); };
window.removeAudio = () => { pendingAudio = null; renderPreviews(); };

function clearPreviews() {
  pendingImage = null;
  pendingAudio = null;
  renderPreviews();
}

function appendUserMessage(text, imgB64, audioB64) {
  const el = document.createElement('div');
  el.className = 'message user';
  
  let mediaHtml = '';
  if (imgB64) mediaHtml += `<img src="data:image/jpeg;base64,${imgB64}" style="max-width:200px;border-radius:8px;margin-bottom:8px;display:block"/>`;
  if (audioB64) mediaHtml += `<div style="font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:4px">🎤 Voice Message</div>`;

  el.innerHTML = `
    <div class="bubble-wrap">
      <div class="bubble user-bubble">
        ${mediaHtml}
        ${esc(text).replace(/\n/g, '<br>')}
      </div>
      <div class="time">${now()}</div>
    </div>
    <div class="avatar" style="background:linear-gradient(135deg,var(--accent),var(--accent2))">👤</div>
  `;
  chatArea.appendChild(el);
  scrollBottom();
}

function appendBotMessage(text, tools) {
  const el = document.createElement('div');
  el.className = 'message bot';
  
  let toolHtml = '';
  if (tools && tools.length) {
    const badges = tools.map(t => `<span style="font-family:var(--font-mono);font-size:10px;padding:2px 8px;border-radius:4px;border:1px solid rgba(108,255,206,0.4);color:var(--accent3);background:rgba(108,255,206,0.07);margin-right:4px">🔧 ${t}</span>`).join('');
    toolHtml = `<div style="margin-top:8px">${badges}</div>`;
  }

  const formattedText = esc(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');

  el.innerHTML = `
    <div class="avatar" style="background:var(--surface2);border:1px solid var(--border)">✦</div>
    <div class="bubble-wrap">
      <div class="bubble bot-bubble" style="line-height:1.65">
        ${formattedText}
        ${toolHtml}
      </div>
      <div class="time">${now()}</div>
    </div>
  `;
  chatArea.appendChild(el);
  scrollBottom();
}

function appendLoading() {
  const id = 'load-' + Date.now();
  const el = document.createElement('div');
  el.className = 'message bot';
  el.id = id;
  el.innerHTML = `
    <div class="avatar" style="background:var(--surface2);border:1px solid var(--border)">✦</div>
    <div class="bubble-wrap">
      <div class="bubble bot-bubble" style="display:flex;align-items:center;gap:6px">
        <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
      </div>
    </div>
  `;
  chatArea.appendChild(el);
  scrollBottom();
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendError(msg) {
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
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}