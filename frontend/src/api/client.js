/**
 * Centralized API Client for Attendance System FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(message, status, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Accept': 'application/json',
    ...options.headers,
  };

  if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  try {
    const response = await fetch(url, { ...options, headers });

    if (!response.ok) {
      let detail = null;
      try {
        const errorData = await response.json();
        detail = errorData.detail || errorData.message || null;
      } catch {
        detail = response.statusText;
      }
      throw new ApiError(detail || `HTTP ${response.status}: ${response.statusText}`, response.status, detail);
    }

    // Return JSON if present
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    }
    return response;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      'Network error: Unable to reach FastAPI backend. Ensure server is running at ' + API_BASE_URL,
      0,
      err.message
    );
  }
}

export const api = {
  get: (endpoint, options) => request(endpoint, { ...options, method: 'GET' }),
  post: (endpoint, body, options) => request(endpoint, { ...options, method: 'POST', body }),
  delete: (endpoint, options) => request(endpoint, { ...options, method: 'DELETE' }),
  getBaseUrl: () => API_BASE_URL,
};
