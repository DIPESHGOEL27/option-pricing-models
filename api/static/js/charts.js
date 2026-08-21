// Plotly chart rendering. Two flavors: charts the backend already builds as
// full Plotly figure JSON (OI buildup, payoff diagram -- rendered as-is,
// just with the background made transparent so the surrounding panel's own
// background shows through instead of Plotly's default), and the skew
// chart, which is built client-side from the raw solved-IV points the
// option-chain response carries.

const PLOTLY_FONT = { family: 'Inter, sans-serif', color: '#9aa4b8', size: 11 };

const BASE_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: PLOTLY_FONT,
  margin: { l: 48, r: 16, t: 16, b: 36 },
  legend: { orientation: 'h', y: -0.2 },
};

const CONFIG = { displayModeBar: false, responsive: true };

export function renderServerPlot(container, plotJsonString) {
  if (!window.Plotly) return;
  const fig = JSON.parse(plotJsonString);
  const layout = { ...fig.layout, ...BASE_LAYOUT, title: undefined };
  window.Plotly.newPlot(container, fig.data, layout, CONFIG);
}

export function renderSkewChart(container, skew) {
  if (!window.Plotly) return;
  if (!skew || skew.error || !skew.points || skew.points.length === 0) {
    container.replaceChildren();
    const msg = document.createElement('div');
    msg.className = 'state-block';
    msg.textContent = skew?.error || 'Not enough liquid quotes to fit a skew curve.';
    container.appendChild(msg);
    return;
  }

  const calls = skew.points.filter((p) => p.option_type === 'call');
  const puts = skew.points.filter((p) => p.option_type === 'put');

  const traces = [
    {
      x: calls.map((p) => p.strike),
      y: calls.map((p) => p.solved_iv * 100),
      mode: 'markers',
      name: 'Calls (solved IV)',
      marker: { color: '#3fb8af', size: 7 },
      type: 'scatter',
    },
    {
      x: puts.map((p) => p.strike),
      y: puts.map((p) => p.solved_iv * 100),
      mode: 'markers',
      name: 'Puts (solved IV)',
      marker: { color: '#e08a3c', size: 7 },
      type: 'scatter',
    },
  ];

  const allSorted = [...skew.points].sort((a, b) => a.strike - b.strike);
  traces.push({
    x: allSorted.map((p) => p.strike),
    y: allSorted.map((p) => p.fitted_iv * 100),
    mode: 'lines',
    name: 'Fitted curve',
    line: { color: '#4f8cff', width: 2, dash: 'dot' },
    type: 'scatter',
  });

  const layout = {
    ...BASE_LAYOUT,
    xaxis: { title: 'Strike', gridcolor: '#262c3b' },
    yaxis: { title: 'IV (%)', gridcolor: '#262c3b' },
  };

  window.Plotly.newPlot(container, traces, layout, CONFIG);
}

export function clearChart(container) {
  if (window.Plotly && container) {
    window.Plotly.purge(container);
  }
}
