/**
 * SIH26166 — Frontend API service.
 *
 * Centralised HTTP client for communicating with the FastAPI backend.
 * All fetch logic lives here so components stay free of transport concerns.
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

/**
 * Check backend health.
 * @returns {Promise<object>} Health status payload.
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Upload a reference and target image pair.
 *
 * Uses multipart/form-data — the browser sets the Content-Type boundary
 * automatically when given a FormData body, so we must NOT set it manually.
 *
 * @param {File} referenceFile - The reference lunar image.
 * @param {File} targetFile    - The target lunar image.
 * @returns {Promise<object>}  Parsed JSON response with metadata for both images.
 */
export async function uploadImages(referenceFile, targetFile) {
  const form = new FormData();
  form.append('reference', referenceFile);
  form.append('target', targetFile);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: form,
    // Do NOT set Content-Type — the browser must generate the boundary.
  });

  if (!res.ok) {
    // Try to extract a structured error detail from FastAPI's error response.
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) detail = err.detail;
    } catch {
      // Response body was not JSON; use the status text.
    }
    throw new Error(detail);
  }

  return res.json();
}
