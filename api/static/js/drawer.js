// Position drawer: opens when a strike is clicked, showing model comparison
// (Black-Scholes, Binomial, Monte Carlo), a single-leg payoff diagram, and
// Greeks computed at that leg's solved IV.

import { apiPost } from './api.js';
import { renderServerPlot, clearChart } from './charts.js';
import { notifyError } from './notifications.js';
import { formatNumber, formatPercent, formatSignedNumber } from './format.js';

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

function renderModelComparison(container, { marketPrice, bsPriced, binomPriced, mcPriced }) {
  container.replaceChildren();
  if (!bsPriced && !binomPriced && !mcPriced) {
    container.appendChild(el('div', 'state-block', 'Model pricing unavailable: no solved IV for this leg.'));
    return;
  }

  const table = el('table', 'model-table');
  const thead = el('thead');
  const trHead = el('tr');
  trHead.appendChild(el('th', '', 'Model'));
  trHead.appendChild(el('th', 'num', 'Price'));
  trHead.appendChild(el('th', 'num', 'Δ vs Market'));
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = el('tbody');

  const models = [
    {
      name: 'Market Price (Actual)',
      price: marketPrice,
      delta: 0,
      isMarket: true,
    },
    {
      name: 'Black-Scholes (Analytic)',
      price: bsPriced?.option_price,
      delta: bsPriced?.option_price != null ? bsPriced.option_price - marketPrice : null,
    },
    {
      name: 'Binomial Tree (100 steps)',
      price: binomPriced?.option_price,
      delta: binomPriced?.option_price != null ? binomPriced.option_price - marketPrice : null,
    },
    {
      name: 'Monte Carlo (GBM, 50k sim)',
      price: mcPriced?.option_price,
      delta: mcPriced?.option_price != null ? mcPriced.option_price - marketPrice : null,
    },
  ];

  for (const m of models) {
    const row = el('tr');
    row.appendChild(el('td', '', m.name));
    row.appendChild(el('td', 'num', m.price != null ? formatNumber(m.price, 2) : '–'));

    const deltaCell = el('td', 'num');
    if (m.isMarket) {
      deltaCell.textContent = '–';
      deltaCell.style.color = 'var(--text-tertiary)';
    } else if (m.delta != null) {
      deltaCell.textContent = formatSignedNumber(m.delta, 2);
      if (Math.abs(m.delta) > 0.01) {
        deltaCell.style.color = m.delta > 0 ? 'var(--positive)' : 'var(--negative)';
      }
    } else {
      deltaCell.textContent = '–';
    }
    row.appendChild(deltaCell);
    tbody.appendChild(row);
  }

  table.appendChild(tbody);
  container.appendChild(table);
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
  const modelComparison = document.getElementById('modelComparison');
  const greeksList = document.getElementById('greeksList');
  const payoffChart = document.getElementById('payoffChart');

  const optionType = legKey === 'ce' ? 'call' : 'put';
  title.textContent = `${symbol} ${strike} ${optionType.toUpperCase()}`;

  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  modelComparison?.replaceChildren(el('div', 'state-block', 'Loading models…'));
  greeksList.replaceChildren(el('div', 'state-block', 'Loading…'));

  const premium = leg?.market_price;
  if (premium == null) {
    modelComparison?.replaceChildren(el('div', 'state-block', 'No market price available for this leg.'));
    greeksList.replaceChildren(el('div', 'state-block', 'No market price available for this leg.'));
    return;
  }

  try {
    const canPrice = leg.solved_iv != null && riskFreeRate != null && timeToExpiry > 0;
    const pricingPayload = canPrice ? {
      S: underlying,
      K: strike,
      T: timeToExpiry,
      r: riskFreeRate,
      sigma: leg.solved_iv,
      optionType: optionType,
    } : null;

    const [payoff, bsPriced, binomPriced, mcPriced] = await Promise.all([
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
      pricingPayload
        ? apiPost('/api/calculate_black_scholes', pricingPayload).catch(() => null)
        : Promise.resolve(null),
      pricingPayload
        ? apiPost('/api/calculate_binomial', { ...pricingPayload, steps: 100 }).catch(() => null)
        : Promise.resolve(null),
      pricingPayload
        ? apiPost('/api/monte_carlo', { ...pricingPayload, model: 'gbm', simulations: 50000 }).catch(() => null)
        : Promise.resolve(null),
    ]);

    if (modelComparison) {
      renderModelComparison(modelComparison, {
        marketPrice: premium,
        bsPriced,
        binomPriced,
        mcPriced,
      });
    }

    if (payoff.plot) {
      renderServerPlot(payoffChart, payoff.plot);
    }

    greeksList.replaceChildren();
    if (bsPriced) {
      for (const key of ['delta', 'gamma', 'theta', 'vega', 'rho']) {
        const row = el('div', 'greek-row');
        const label = el('span', 'label', key[0].toUpperCase() + key.slice(1));
        const unit = el('span', 'unit', GREEK_UNITS[key]);
        label.appendChild(unit);
        const value = el('span', 'num', formatNumber(bsPriced[key], 4));
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
    modelComparison?.replaceChildren(el('div', 'state-block', 'Failed to load model comparison.'));
    greeksList.replaceChildren(el('div', 'state-block', 'Failed to load.'));
  }
}
