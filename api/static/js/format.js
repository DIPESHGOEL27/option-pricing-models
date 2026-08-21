// Formatting helpers. The old app rendered negative currency as "$-10.13"
// (string-concatenated the symbol before Number.toFixed, which puts the
// minus sign after the symbol) -- Intl.NumberFormat handles sign placement
// correctly by construction, so that bug class isn't reachable here.

const inrFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 2,
});

const inrFormatterNoDecimals = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
});

export function formatINR(value, decimals = true) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return decimals ? inrFormatter.format(value) : inrFormatterNoDecimals.format(value);
}

export function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return Number(value).toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatPercent(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

export function formatTime(date) {
  return date.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatSignedNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  const sign = value > 0 ? '+' : '';
  return `${sign}${formatNumber(value, decimals)}`;
}
