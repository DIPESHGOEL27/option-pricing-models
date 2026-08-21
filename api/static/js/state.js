// Minimal app state -- no framework, just a plain object plus a small
// pub/sub so modules can react without being wired together directly.

const state = {
  symbol: 'NIFTY',
  expiry: '', // '' = nearest
  chain: null, // last successful /api/india/option_chain response
  lastFetchedAt: null,
  isStale: false,
};

const listeners = new Set();

export function getState() {
  return state;
}

export function setState(patch) {
  Object.assign(state, patch);
  listeners.forEach((fn) => fn(state));
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
