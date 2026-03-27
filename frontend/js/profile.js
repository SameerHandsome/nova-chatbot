/**
 * profile.js — Profile & user preferences page logic.
 * Fetches user info and analyzed preferences in parallel.
 */

import { apiClient }               from './api.js';
import { renderNavbar, showToast } from './components.js';

document.addEventListener('DOMContentLoaded', async () => {
  renderNavbar('profile');
  await Promise.all([loadProfile(), loadPreferences()]);
});

async function loadProfile() {
  try {
    const user = await apiClient.me();

    document.getElementById('profileInitial').textContent = user.username[0].toUpperCase();
    document.getElementById('profileName').textContent    = user.username;
    document.getElementById('profileEmail').textContent   = user.email;

    const tierEl       = document.getElementById('profileTier');
    tierEl.textContent = user.tier.toUpperCase();
    tierEl.className   = `profile-tier ${user.tier}`;

    const since = new Date(user.created_at).toLocaleDateString(undefined, {
      month: 'long', year: 'numeric',
    });
    document.getElementById('profileSince').textContent = `Member since ${since}`;

  } catch (err) {
    showToast('Could not load profile', 'error');
  }
}

async function loadPreferences() {
  const panel = document.getElementById('prefPanel');

  try {
    const pref = await apiClient.getPreferences();
    panel.innerHTML = buildPreferences(pref);

  } catch (err) {
    // 404 = no preferences yet (normal for new users)
    panel.innerHTML = `
      <div class="no-prefs">
        <span class="no-prefs-icon">🧠</span>
        <p>No preferences analyzed yet.<br>Complete and end a chat session to generate your preference profile.</p>
        <a href="/chat.html" class="btn btn-primary btn-sm" style="margin-top:8px">Start chatting →</a>
      </div>
    `;
  }
}

function buildPreferences(p) {
  const topics = (p.topics_of_interest || [])
    .map(t => `<span class="pref-tag">${esc(t)}</span>`)
    .join('') || '<span style="color:var(--muted);font-size:13px">None detected</span>';

  const voiceFlag = `<span class="flag-chip ${p.uses_voice  ? 'on' : 'off'}">🎤 Voice ${p.uses_voice  ? 'Yes' : 'No'}</span>`;
  const imgFlag   = `<span class="flag-chip ${p.uses_images ? 'on' : 'off'}">🖼 Images ${p.uses_images ? 'Yes' : 'No'}</span>`;

  const updated = p.last_analyzed_at
    ? new Date(p.last_analyzed_at).toLocaleString()
    : 'Never';

  return `
    <div class="pref-panel-title">🧠 AI-Analyzed Preferences</div>

    <div class="pref-row">
      <span class="pref-label">Communication Style</span>
      <span class="pref-value">${cap(p.communication_style) || '—'}</span>
    </div>

    <div class="pref-row">
      <span class="pref-label">Topics of Interest</span>
      <div class="pref-tags">${topics}</div>
    </div>

    <div class="pref-row">
      <span class="pref-label">Preferred Response Length</span>
      <span class="pref-value">${cap(p.preferred_response_length) || '—'}</span>
    </div>

    <div class="pref-row">
      <span class="pref-label">Language</span>
      <span class="pref-value">${(p.language || 'en').toUpperCase()}</span>
    </div>

    <div class="pref-row">
      <span class="pref-label">Modality Usage</span>
      <div class="modality-flags">${voiceFlag}${imgFlag}</div>
    </div>

    <div class="pref-updated">Last analyzed: ${updated}</div>
  `;
}

function cap(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function esc(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}