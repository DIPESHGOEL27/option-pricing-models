// Renders the option chain table: builds every cell via DOM APIs (not
// innerHTML string interpolation) so nothing from the NSE feed -- which is
// external, scraped data -- can be interpreted as markup.

import { formatNumber, formatCompact } from './format.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function oiChangeCell(value) {
  const cell = el('td', 'num');
  if (value === null || value === undefined) {
    cell.textContent = '–';
    return cell;
  }
  const span = el('span', `oi-change ${value >= 0 ? 'positive' : 'negative'}`);
  span.textContent = `${value >= 0 ? '+' : ''}${formatCompact(value)}`;
  cell.appendChild(span);
  return cell;
}

function ivCell(leg) {
  const cell = el('td', 'num');
  if (!leg) {
    cell.textContent = '–';
    return cell;
  }
  if (leg.solved_iv === undefined || leg.solved_iv === null) {
    if (leg.skip_reason) {
      const marker = el('span', 'iv-marker skipped', '–');
      marker.title = `Not solved: ${leg.skip_reason.replace(/_/g, ' ')}`;
      cell.appendChild(marker);
    } else {
      cell.textContent = '–';
    }
    return cell;
  }
  const valSpan = el('span', '', `${(leg.solved_iv * 100).toFixed(1)}%`);
  cell.appendChild(valSpan);

  if (leg.low_confidence) {
    const warn = el('span', 'iv-marker low-confidence', ' ⚠');
    warn.title = 'Low confidence: option has very low vega at this price/time to expiry';
    cell.appendChild(warn);
  }
  if (leg.skew_flag && leg.skew_flag !== 'normal') {
    const badge = el('span', `skew-badge ${leg.skew_flag}`, leg.skew_flag);
    badge.style.marginLeft = '6px';
    cell.appendChild(badge);
  }
  return cell;
}

function priceCell(leg) {
  const cell = el('td', 'num');
  cell.textContent = leg?.market_price != null ? formatNumber(leg.market_price, 2) : '–';
  return cell;
}

function oiCell(leg) {
  const cell = el('td', 'num');
  cell.textContent = leg?.open_interest != null ? formatCompact(leg.open_interest) : '–';
  return cell;
}

export function renderChainTable(container, chainData, { onStrikeClick } = {}) {
  container.replaceChildren();

  if (!chainData || !chainData.strikes || chainData.strikes.length === 0) {
    const empty = el('div', 'state-block');
    empty.appendChild(el('div', 'state-title', 'No chain data'));
    empty.appendChild(el('div', '', 'This instrument/expiry has no option chain data right now.'));
    container.appendChild(empty);
    return;
  }

  const underlying = chainData.underlying_value;
  const maxPainStrike = chainData.max_pain?.strike;
  let atmStrike = null;
  let atmDiff = Infinity;
  for (const entry of chainData.strikes) {
    const diff = Math.abs(entry.strike - underlying);
    if (diff < atmDiff) {
      atmDiff = diff;
      atmStrike = entry.strike;
    }
  }

  const table = el('table', 'chain-table');
  const thead = el('thead');
  const headRow1 = el('tr');
  const thCalls = el('th', 'group-call', 'Calls');
  thCalls.colSpan = 4;
  headRow1.appendChild(thCalls);
  const thStrike = el('th', '', '');
  headRow1.appendChild(thStrike);
  const thPuts = el('th', 'group-put', 'Puts');
  thPuts.colSpan = 4;
  headRow1.appendChild(thPuts);

  const headRow2 = el('tr');
  [
    ['group-call', 'OI'], ['group-call', 'Chg'], ['group-call', 'LTP'], ['group-call', 'IV'],
    ['', 'Strike'],
    ['group-put', 'IV'], ['group-put', 'LTP'], ['group-put', 'Chg'], ['group-put', 'OI'],
  ].forEach(([cls, label]) => headRow2.appendChild(el('th', cls, label)));
  thead.appendChild(headRow1);
  thead.appendChild(headRow2);
  table.appendChild(thead);

  const tbody = el('tbody');
  for (const entry of chainData.strikes) {
    const row = el('tr');
    if (entry.strike === atmStrike) row.classList.add('atm-row');
    if (entry.strike === maxPainStrike) row.classList.add('max-pain-row');

    const ce = entry.ce;
    const pe = entry.pe;

    const oiCe = oiCell(ce); oiCe.classList.add('call-cell'); row.appendChild(oiCe);
    const chgCe = oiChangeCell(ce?.oi_change); chgCe.classList.add('call-cell'); row.appendChild(chgCe);
    const priceCe = priceCell(ce); priceCe.classList.add('call-cell'); row.appendChild(priceCe);
    const ivCe = ivCell(ce); ivCe.classList.add('call-cell'); row.appendChild(ivCe);

    const strikeCell = el('td', 'strike-cell num');
    const strikeBtn = el('button', '', formatNumber(entry.strike, 0));
    strikeBtn.type = 'button';
    strikeBtn.addEventListener('click', () => onStrikeClick?.(entry));
    strikeCell.appendChild(strikeBtn);
    row.appendChild(strikeCell);

    const ivPe = ivCell(pe); ivPe.classList.add('put-cell'); row.appendChild(ivPe);
    const pricePe = priceCell(pe); pricePe.classList.add('put-cell'); row.appendChild(pricePe);
    const chgPe = oiChangeCell(pe?.oi_change); chgPe.classList.add('put-cell'); row.appendChild(chgPe);
    const oiPe = oiCell(pe); oiPe.classList.add('put-cell'); row.appendChild(oiPe);

    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

export function renderChainLoading(container) {
  container.replaceChildren();
  const block = el('div', 'state-block');
  block.appendChild(el('div', 'spinner'));
  block.appendChild(el('span', '', 'Loading live chain…'));
  container.appendChild(block);
}

export function renderChainError(container, message) {
  container.replaceChildren();
  const block = el('div', 'state-block');
  block.appendChild(el('div', 'state-title', 'Could not load the option chain'));
  block.appendChild(el('div', '', message));
  container.appendChild(block);
}
