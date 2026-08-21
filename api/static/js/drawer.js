// Position drawer: opens when a strike is clicked, showing a single-leg
// payoff diagram and Greeks computed at that leg's solved IV. This is a
// quick-look view (one leg, quantity 1, long) rather than a full multi-leg
// portfolio builder -- a deliberate v1 scope choice, not an oversight.

import { apiPost } from './api.js';
import { renderServerPlot, clearChart } from './charts.js';
import { notifyError } from './notifications.js';
import { formatNumber, formatPercent } from './format.js';

const GREEK_UNITS = {
  delta: '∂price / ∂spot',
  gamma: '∂delta / ∂spot',
  theta: 'per day',
  vega: 'per 1% vol',
  rho: 'per 1% rate',
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function closeDrawer() {
  const drawer = document.getElementById('positionDrawer');
  drawer.classList.remove('open');
  drawer.setAttribute('aria-hidden', 'true');
  clearChart(document.getElementById('payoffChart'));
}

export async function openDrawerForStrike({ strike, legKey, leg, underlying, timeToExpiry, riskFreeRate, symbol }) {
  const drawer = document.getElementById('positionDrawer');
  const title = document.getElementById('drawerTitle');
  const greeksList = document.getElementById('greeksList');
  const payoffChart = document.getElementById('payoffChart');

  const optionType = legKey === 'ce' ? 'call' : 'put';
  title.textContent = `${symbol} ${strike} ${optionType.toUpperCase()}`;

  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  greeksList.replaceChildren(el('div', 'state-block', 'Loading…'));

  const premium = leg?.market_price;
  if (premium == null) {
    greeksList.replaceChildren(el('div', 'state-block', 'No market price available for this leg.'));
    return;
  }

  try {
    const [payoff, priced] = await Promise.all([
      apiPost('/api/plot_payoff', {
        positions: [
          {
            symbol,
            option_type: optionType,
            strike,
            quantity: 1,
            premium_paid: premium,
            underlying_price: underlying,
          },
        ],
      }),
      leg.solved_iv != null && riskFreeRate != null
        ? apiPost('/api/calculate_black_scholes', {
            S: underlying,
            K: strike,
            T: timeToExpiry,
            r: riskFreeRate,
            sigma: leg.solved_iv,
            optionType: optionType,
          })
        : Promise.resolve(null),
    ]);

    if (payoff.plot) {
      renderServerPlot(payoffChart, payoff.plot);
    }

    greeksList.replaceChildren();
    if (priced) {
      for (const key of ['delta', 'gamma', 'theta', 'vega', 'rho']) {
        const row = el('div', 'greek-row');
        const label = el('span', 'label', key[0].toUpperCase() + key.slice(1));
        const unit = el('span', 'unit', GREEK_UNITS[key]);
        label.appendChild(unit);
        const value = el('span', 'num', formatNumber(priced[key], 4));
        row.appendChild(label);
        row.appendChild(value);
        greeksList.appendChild(row);
      }
      const ivRow = el('div', 'greek-row');
      ivRow.appendChild(el('span', 'label', 'Solved IV used'));
      ivRow.appendChild(el('span', 'num', formatPercent(leg.solved_iv)));
      greeksList.appendChild(ivRow);
    } else {
      greeksList.appendChild(el('div', 'state-block', 'Greeks unavailable: no solved IV for this leg.'));
    }

    if (payoff.analysis) {
      const be = payoff.analysis.break_even_points;
      const summary = el('div', 'stat-sub');
      summary.style.marginTop = 'var(--space-3)';
      summary.textContent = be && be.length
        ? `Break-even: ${be.map((v) => formatNumber(v, 0)).join(', ')}`
        : 'No break-even within the plotted range.';
      greeksList.appendChild(summary);
    }
  } catch (err) {
    notifyError(`Could not load position details: ${err.message}`);
    greeksList.replaceChildren(el('div', 'state-block', 'Failed to load.'));
  }
}
