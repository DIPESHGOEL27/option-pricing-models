// One notification system. The previous frontend had two competing ones
// (showAlert / showNotification) that both rendered at the same screen
// position and, worse, showAlert's auto-dismiss used an unscoped
// `$(".alert")` selector that faded out every alert on the page --
// including static empty-state panels -- five seconds after ANY toast
// fired, even on a success path. Each call here creates and dismisses only
// its own element.

const STACK_ID = 'notificationStack';

function stack() {
  return document.getElementById(STACK_ID);
}

export function notify(message, type = 'info', durationMs = 5000) {
  const container = stack();
  if (!container) return;

  const el = document.createElement('div');
  el.className = `notification ${type}`;
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');

  const text = document.createElement('span');
  text.textContent = message;
  el.appendChild(text);

  const closeBtn = document.createElement('button');
  closeBtn.className = 'notification-close';
  closeBtn.setAttribute('aria-label', 'Dismiss');
  closeBtn.textContent = '×';
  el.appendChild(closeBtn);

  const remove = () => el.remove();
  closeBtn.addEventListener('click', remove);

  container.appendChild(el);
  if (durationMs > 0) {
    setTimeout(remove, durationMs);
  }
  return remove;
}

export const notifyError = (message) => notify(message, 'error', 7000);
export const notifySuccess = (message) => notify(message, 'success', 4000);
export const notifyWarning = (message) => notify(message, 'warning', 6000);
