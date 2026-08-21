// Thin fetch wrapper: every route on this app returns either a real payload
// or {error: "..."} at various HTTP statuses (some legacy routes still
// return errors at 200) -- this normalizes both into a thrown Error so
// every caller can just try/catch instead of re-checking response.ok and
// body.error separately every time.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch (networkError) {
    throw new ApiError(`Network error reaching ${path}: ${networkError.message}`, 0);
  }

  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      throw new ApiError(`Invalid JSON response from ${path}`, response.status);
    }
  }

  if (!response.ok) {
    const message = body?.error || `${path} returned HTTP ${response.status}`;
    throw new ApiError(message, response.status);
  }
  if (body?.error) {
    throw new ApiError(body.error, response.status);
  }
  return body;
}

export function apiGet(path) {
  return request(path, { method: 'GET' });
}

export function apiPost(path, payload) {
  return request(path, { method: 'POST', body: JSON.stringify(payload) });
}
