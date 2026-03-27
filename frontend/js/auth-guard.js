/**
 * auth-guard.js — Auth protection for every protected page.
 * Include as the FIRST script tag on chat.html, history.html, profile.html.
 * Redirects to login if JWT is missing or expired.
 */

import { isTokenExpired, captureTokenFromUrl } from './auth.js';

// Check for OAuth callback token first
captureTokenFromUrl();

// Then guard
if (isTokenExpired()) {
  window.location.replace('/login.html');
}