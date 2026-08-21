import { apiGet } from './api.js';
import { getState, setState } from './state.js';
import { renderChainTable, renderChainLoading, renderChainError } from './chain.js';
import { renderServerPlot, renderSkewChart } from './charts.js';
import { openDrawerForStrike, closeDrawer } from './drawer.js';
import { notify, notifyError, notifyWarning } from './notifications.js';
import { formatNumber, formatPercent, formatTime } from './format.js';

const POLL_INTERVAL_MS = 45_000;
const MAX_EXPIRIES_IN_DROPDOWN = 6;

let pollTimer = null;

const els = {
  instrumentSelect: document.getElementById('instrumentSelect'),
  expirySelect: document.getElementById('expirySelect'),
  refreshBtn: document.getElementById('refreshBtn'),
  chainContainer: document.getElementById('chainContainer'),
  chainSubtitle: document.getElementById('chainSubtitle'),
  spotPrice: document.getElementById('spotPrice'),
  spotChange: document.getElementById('spotChange'),
  marketStatusDot: document.getElementById('marketStatusDot'),
  marketStatusText: document.getElementById('marketStatusText'),
  asOf: document.getElementById('asOf'),
  statPcr: document.getElementById('statPcr'),
  statMaxPain: document.getElementById('statMaxPain'),
  statAtmIv: document.getElementById('statAtmIv'),
  statRate: document.getElementById('statRate'),
  statRateSource: document.getElementById('statRateSource'),
  skewChart: document.getElementById('skewChart'),
  oiChart: document.getElementById('oiChart'),
  staleBanner: document.getElementById('staleBanner'),
  staleBannerText: document.getElementById('staleBannerText'),
  onboardingStrip: document.getElementById('onboardingStrip'),
  dismissOnboarding: document.getElementById('dismissOnboarding'),
  howToReadBtn: document.getElementById('howToReadBtn'),
  howToReadModal: document.getElementById('howToReadModal'),
  howToReadCloseBtn: document.getElementById('howToReadCloseBtn'),
  howToReadContent: document.getElementById('howToReadContent'),
  drawerCloseBtn: document.getElementById('drawerCloseBtn'),
};

function updateAsOf(date, stale) {
  els.asOf.textContent = `as of ${formatTime(date)}${stale ? ' · stale' : ''}`;
  els.asOf.classList.toggle('stale', stale);
}

function updateMarketStatusFromChain() {
  // Option-chain data itself is the freshest signal we have of whether the
  // market is live; a dedicated status call is a nice-to-have, not fetched
  // separately here to avoid an extra round trip on every refresh.
  els.marketStatusDot.className = 'status-dot open';
  els.marketStatusText.textContent = 'Live chain data';
}

function populateExpiryOptions(expiryDates) {
  const current = els.expirySelect.value;
  els.expirySelect.replaceChildren();
  const nearestOpt = document.createElement('option');
  nearestOpt.value = '';
  nearestOpt.textContent = 'Nearest expiry';
  els.expirySelect.appendChild(nearestOpt);

  for (const date of expiryDates.slice(0, MAX_EXPIRIES_IN_DROPDOWN)) {
    const opt = document.createElement('option');
    opt.value = date;
    opt.textContent = date;
    els.expirySelect.appendChild(opt);
  }
  if ([...els.expirySelect.options].some((o) => o.value === current)) {
    els.expirySelect.value = current;
  }
}

function renderStats(chainData) {
  els.statPcr.textContent = chainData.pcr?.oi_pcr != null ? formatNumber(chainData.pcr.oi_pcr, 2) : '–';
  els.statMaxPain.textContent = chainData.max_pain?.strike != null ? formatNumber(chainData.max_pain.strike, 0) : '–';

  const atmIv = chainData.skew?.atm_iv;
  els.statAtmIv.textContent = atmIv != null ? formatPercent(atmIv) : '–';

  const rate = chainData.iv_analysis?.risk_free_rate;
  els.statRate.textContent = rate != null ? formatPercent(rate) : '–';
  const source = chainData.iv_analysis?.rate_source;
  els.statRateSource.textContent = source === 'put_call_parity'
    ? 'from put-call parity'
    : source === 'rbi_fallback'
      ? 'from RBI (fallback)'
      : '';
}

function findAtmEntry(chainData) {
  let best = null;
  let bestDiff = Infinity;
  for (const entry of chainData.strikes) {
    const diff = Math.abs(entry.strike - chainData.underlying_value);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = entry;
    }
  }
  return best;
}

async function loadChain({ silent = false } = {}) {
  const { symbol, expiry } = getState();

  if (!silent) {
    renderChainLoading(els.chainContainer);
  }

  const path = `/api/india/option_chain/${encodeURIComponent(symbol)}` + (expiry ? `?expiry=${encodeURIComponent(expiry)}` : '');

  try {
    const chainData = await apiGet(path);

    setState({ chain: chainData, lastFetchedAt: new Date(), isStale: false });
    els.staleBanner.hidden = true;

    els.spotPrice.textContent = formatNumber(chainData.underlying_value, 2);
    els.chainSubtitle.textContent = `Expiry ${chainData.expiry} · ${chainData.strikes.length} strikes`;
    updateMarketStatusFromChain();
    updateAsOf(new Date(), false);

    populateExpiryOptions(chainData.expiry_dates || []);
    renderChainTable(els.chainContainer, chainData, {
      onStrikeClick: (entry) => onStrikeClick(entry, chainData),
    });
    renderStats(chainData);
    renderSkewChart(els.skewChart, chainData.skew);
    if (chainData.plot) {
      renderServerPlot(els.oiChart, chainData.plot);
    }

    if (chainData.iv_analysis?.summary) {
      const { legs_skipped, legs_priced } = chainData.iv_analysis.summary;
      if (legs_skipped > 0 && !silent) {
        notify(`${legs_priced} legs priced, ${legs_skipped} skipped (illiquid/no quote) — see per-cell tooltips.`, 'info', 4000);
      }
    }
  } catch (err) {
    if (getState().chain) {
      // We have something to show -- keep it visible, mark it stale rather
      // than replacing a working view with an error screen.
      setState({ isStale: true });
      els.staleBanner.hidden = false;
      els.staleBannerText.textContent = `Live refresh failed (${err.message}). Showing the last successful load.`;
      updateAsOf(getState().lastFetchedAt, true);
      if (!silent) notifyWarning(`Refresh failed: ${err.message}`);
    } else {
      renderChainError(els.chainContainer, err.message);
      notifyError(`Could not load the option chain: ${err.message}`);
    }
  }
}

function onStrikeClick(entry, chainData) {
  const isCall = entry.strike >= chainData.underlying_value;
  const legKey = isCall ? 'ce' : 'pe';
  const leg = entry[legKey];
  if (!leg) {
    notifyWarning('No quote available for this leg.');
    return;
  }
  openDrawerForStrike({
    strike: entry.strike,
    legKey,
    leg,
    underlying: chainData.underlying_value,
    timeToExpiry: chainData.iv_analysis?.time_to_expiry?.calendar_years,
    riskFreeRate: chainData.iv_analysis?.risk_free_rate,
    symbol: chainData.symbol,
  });
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (document.visibilityState === 'visible') {
      await loadChain({ silent: true });
    }
    schedulePoll();
  }, POLL_INTERVAL_MS);
}

function wireControls() {
  els.instrumentSelect.addEventListener('change', () => {
    setState({ symbol: els.instrumentSelect.value, expiry: '' });
    loadChain();
  });
  els.expirySelect.addEventListener('change', () => {
    setState({ expiry: els.expirySelect.value });
    loadChain();
  });
  els.refreshBtn.addEventListener('click', () => loadChain());

  els.dismissOnboarding.addEventListener('click', () => {
    els.onboardingStrip.hidden = true;
  });

  const howToReadCopy = [
    ['Put/Call Ratio (PCR)', 'Total put open interest divided by total call open interest. Above ~1 is read as bearish positioning (more put buying), below ~0.7 as bullish -- a positioning indicator, not a price forecast.'],
    ['Max Pain', 'The strike at which option writers as a group would owe the least at expiry. A widely used heuristic for where price may gravitate near expiry, not a guarantee.'],
    ['Solved IV vs NSE IV', 'This terminal solves implied volatility independently from real bid/ask quotes (Newton-Raphson, falling back to Brent) and shows it alongside NSE’s own published figure. They can differ -- that gap is itself informative, not a bug.'],
    ['Skew flag (rich/cheap)', 'A strike’s solved IV compared to a smooth curve fitted through the rest of the chain. "Rich" means priced high relative to its neighbors today; "cheap" means priced low. Relative value, not a prediction.'],
    ['Change in OI', 'Open interest added or removed since the previous session at that strike -- shows where fresh positioning is building, distinct from the total OI level.'],
  ];
  els.howToReadContent.replaceChildren();
  for (const [term, body] of howToReadCopy) {
    const block = document.createElement('div');
    block.style.marginBottom = 'var(--space-4)';
    const h = document.createElement('h4');
    h.textContent = term;
    h.style.marginBottom = 'var(--space-1)';
    const p = document.createElement('p');
    p.textContent = body;
    p.style.color = 'var(--text-secondary)';
    p.style.fontSize = 'var(--text-sm)';
    block.appendChild(h);
    block.appendChild(p);
    els.howToReadContent.appendChild(block);
  }

  els.howToReadBtn.addEventListener('click', () => {
    els.howToReadModal.classList.add('open');
    els.howToReadModal.setAttribute('aria-hidden', 'false');
  });
  els.howToReadCloseBtn.addEventListener('click', () => {
    els.howToReadModal.classList.remove('open');
    els.howToReadModal.setAttribute('aria-hidden', 'true');
  });
  els.drawerCloseBtn.addEventListener('click', closeDrawer);

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      const { lastFetchedAt } = getState();
      if (!lastFetchedAt || Date.now() - lastFetchedAt.getTime() > POLL_INTERVAL_MS) {
        loadChain({ silent: true });
      }
    }
  });
}

wireControls();
loadChain();
schedulePoll();
