/**
 * components.js — Shared UI components injected into every page.
 * Call renderNavbar() after DOMContentLoaded on every protected page.
 */

import { getCurrentUserInfo, logout } from './auth.js';

export function renderNavbar(activePage = '') {
  const user  = getCurrentUserInfo();
  const tier  = user?.tier || 'free';
  const pages = [
    { href: '/chat.html',    label: '💬 Chat',     key: 'chat' },
    { href: '/history.html', label: '📋 History',  key: 'history' },
    { href: '/profile.html', label: '👤 Profile',  key: 'profile' },
  ];

  const links = pages.map(p => `
    <a href="${p.href}" class="nav-link ${activePage === p.key ? 'active' : ''}">
      ${p.label}
    </a>
  `).join('');

  const nav = document.createElement('nav');
  nav.className = 'navbar';
  nav.innerHTML = `
    <a href="/chat.html" class="navbar-logo">
      <div class="logo-dot"></div>
      NOVA
    </a>
    <div class="navbar-links">${links}</div>
    <div class="navbar-right">
      <span class="tier-badge ${tier}">${tier.toUpperCase()}</span>
      <button class="btn btn-ghost btn-sm" id="logoutBtn">Sign out</button>
    </div>
  `;

  document.body.prepend(nav);
  document.getElementById('logoutBtn').addEventListener('click', logout);
}

export function showToast(message, type = 'success') {
  let toast = document.getElementById('__nova_toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = '__nova_toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent  = message;
  toast.className    = `toast ${type} show`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), 3000);
}