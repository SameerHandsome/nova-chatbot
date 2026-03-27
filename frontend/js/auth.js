/**
 * auth.js — JWT token lifecycle.
 * Storage, retrieval, expiry check, and logout.
 * Imported by auth-guard.js and all pages that need user info.
 */

const TOKEN_KEY = 'nova_token';

export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

/**
 * Decode JWT payload (base64url) without verifying signature.
 * Verification happens server-side on every request.
 */
export function decodeToken(token) {
  try {
    const payload = token.split('.')[1];
    const decoded = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decoded);
  } catch {
    return null;
  }
}

/**
 * Returns true if token is missing or past its exp timestamp.
 */
export function isTokenExpired() {
  const token = getToken();
  if (!token) return true;

  const payload = decodeToken(token);
  if (!payload || !payload.exp) return true;

  // exp is in seconds, Date.now() is milliseconds
  return Date.now() >= payload.exp * 1000;
}

/**
 * Returns the current user's basic info from the token payload.
 * { sub, email, tier }
 */
export function getCurrentUserInfo() {
  const token = getToken();
  if (!token) return null;
  return decodeToken(token);
}

/**
 * Logout — clear token and redirect to login.
 */
export function logout() {
  clearToken();
  window.location.href = '/login.html';
}

/**
 * Store token from URL query param (used after Google OAuth callback).
 * Call this on pages that may receive ?token=xxx
 */
export function captureTokenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const token  = params.get('token');
  if (token) {
    saveToken(token);
    // Remove token from URL bar (security hygiene)
    window.history.replaceState({}, '', window.location.pathname);
    return true;
  }
  return false;
}