const state = {
  options: null,
  data: null,
  frameIdx: -1,
  timer: null,
  map: null,
  staticLayer: null,
  frameLayer: null,
};

const paramSchema = {
  SEQSCAN: [
    { key: "dist_thresh", label: "dist_thresh (m)", min: 3, max: 120, step: 1 },
    { key: "time_thresh", label: "time_thresh (min)", min: 5, max: 720, step: 1 },
    { key: "min_pts", label: "min_pts", min: 2, max: 8, step: 1 },
    { key: "dur_min", label: "dur_min (min)", min: 1, max: 90, step: 1 },
  ],
  "TA-DBSCAN": [
    { key: "dist_thresh", label: "dist_thresh (m)", min: 3, max: 120, step: 1 },
    { key: "time_thresh", label: "time_thresh (min)", min: 5, max: 720, step: 1 },
    { key: "min_pts", label: "min_pts", min: 2, max: 8, step: 1 },
    { key: "dur_min", label: "dur_min (min)", min: 1, max: 90, step: 1 },
  ],
  Lachesis: [
    { key: "delta_roam", label: "delta_roam (m)", min: 3, max: 120, step: 1 },
    { key: "dt_max", label: "dt_max (min)", min: 5, max: 720, step: 1 },
    { key: "dur_min", label: "dur_min (min)", min: 1, max: 90, step: 1 },
  ],
  HDBSCAN: [
    { key: "time_thresh", label: "time_thresh (min)", min: 5, max: 720, step: 1 },
    { key: "min_pts", label: "min_pts", min: 2, max: 8, step: 1 },
    { key: "min_cluster_size", label: "min_cluster_size", min: 1, max: 8, step: 1 },
    { key: "dur_min", label: "dur_min (min)", min: 1, max: 90, step: 1 },
  ],
  "Grid-Based": [
    { key: "time_thresh", label: "time_thresh (min)", min: 5, max: 720, step: 1 },
    { key: "min_cluster_size", label: "min_cluster_size", min: 1, max: 8, step: 1 },
    { key: "dur_min", label: "dur_min (min)", min: 1, max: 90, step: 1 },
    { key: "h3_res", label: "h3_res", min: 7, max: 12, step: 1 },
  ],
};

function setStatus(text, isError = false) {
  const el = document.getElementById("statusText");
  el.textContent = text;
  el.style.color = isError ? "#a21d23" : "#24334f";
}

function toRgba(rgb, alpha = 1.0) {
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function initMap() {
  if (state.map) return;
  state.map = L.map("map", { zoomControl: true }).setView([38.32, -36.66], 14);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap",
  }).addTo(state.map);
  state.staticLayer = L.layerGroup().addTo(state.map);
  state.frameLayer = L.layerGroup().addTo(state.map);
}

function renderMetricCards(meta) {
  const metrics = [
    { label: "Input pings", value: meta.input_pings.toLocaleString() },
    { label: "Sparse/noisy pings", value: meta.sparse_pings.toLocaleString() },
    { label: "Stop runs", value: meta.stop_runs.toLocaleString() },
    { label: "Noise points", value: meta.noise_points.toLocaleString() },
    { label: "Selected user", value: meta.user_id },
  ];

  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = "";
  for (const item of metrics) {
    const card = document.createElement("div");
    card.className = "metric";
    card.innerHTML = `
      <div class="metric-label">${item.label}</div>
      <div class="metric-value">${item.value}</div>
    `;
    grid.appendChild(card);
  }
}

function renderParamInputs(algo) {
  const container = document.getElementById("algoParamsContainer");
  const defs = state.options.default_params[algo] || {};
  const schema = paramSchema[algo] || [];

  container.innerHTML = "";
  for (const field of schema) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <label for="param-${field.key}">${field.label}</label>
      <input
        id="param-${field.key}"
        data-param="${field.key}"
        type="number"
        min="${field.min}"
        max="${field.max}"
        step="${field.step}"
        value="${defs[field.key]}"
      />
    `;
    container.appendChild(wrap);
  }
}

function readParams() {
  const inputs = document.querySelectorAll("#algoParamsContainer input[data-param]");
  const params = {};
  for (const el of inputs) {
    params[el.dataset.param] = Number(el.value);
  }
  return params;
}

function populateControls(options) {
  const userSelect = document.getElementById("userSelect");
  userSelect.innerHTML = "";
  for (const u of options.users) {
    const opt = document.createElement("option");
    opt.value = u.user_id;
    opt.textContent = `${u.user_id} (${u.count})`;
    userSelect.appendChild(opt);
  }

  const algoSelect = document.getElementById("algorithmSelect");
  algoSelect.innerHTML = "";
  for (const algo of options.algorithms) {
    const opt = document.createElement("option");
    opt.value = algo;
    opt.textContent = algo;
    algoSelect.appendChild(opt);
  }

  algoSelect.addEventListener("change", () => {
    renderParamInputs(algoSelect.value);
  });
  renderParamInputs(algoSelect.value);
}

function setFrame(idx) {
  if (!state.data) return;
  state.frameIdx = Math.max(-1, Math.min(idx, state.data.sparse.length - 1));
  document.getElementById("frameSlider").value = String(state.frameIdx);
  const shown = state.frameIdx < 0 ? 0 : state.frameIdx + 1;
  document.getElementById("frameLabel").textContent = `${shown} / ${state.data.sparse.length}`;
  renderFrame();
}

function renderStaticMapLayers() {
  state.staticLayer.clearLayers();
  const data = state.data;
  const denseLatLng = data.dense.map((p) => [p.latitude, p.longitude]);
  L.polyline(denseLatLng, {
    color: "rgba(62,95,146,0.7)",
    weight: 2,
  }).addTo(state.staticLayer);

  for (const run of data.stop_runs) {
    const c = data.cluster_colors[String(run.cluster)] || [130, 138, 150];
    L.circleMarker([run.latitude, run.longitude], {
      radius: 7,
      color: toRgba(c, 0.9),
      fillColor: toRgba(c, 0.25),
      fillOpacity: 0.6,
      weight: 1,
    }).addTo(state.staticLayer);
  }
}

function renderFrameMapLayers() {
  state.frameLayer.clearLayers();
  const data = state.data;
  if (state.frameIdx < 0) return;

  const visible = data.sparse.slice(0, state.frameIdx + 1);
  const currentTs = data.sparse[state.frameIdx].timestamp;

  const pathLatLng = visible.map((p) => [p.latitude, p.longitude]);
  if (pathLatLng.length > 1) {
    L.polyline(pathLatLng, {
      color: "rgba(62,95,146,0.7)",
      weight: 2,
    }).addTo(state.frameLayer);
  }

  for (const run of data.stop_runs) {
    if (run.start_timestamp > currentTs) continue;
    const c = data.cluster_colors[String(run.cluster)] || [130, 138, 150];
    L.circleMarker([run.latitude, run.longitude], {
      radius: 7,
      color: toRgba(c, 0.9),
      fillColor: toRgba(c, 0.25),
      fillOpacity: 0.6,
      weight: 1,
    }).addTo(state.frameLayer);
  }

  for (const p of visible) {
    const c = data.cluster_colors[String(p.cluster)] || [130, 138, 150];
    L.circleMarker([p.latitude, p.longitude], {
      radius: 4,
      color: toRgba(c, 1.0),
      fillColor: toRgba(c, 0.7),
      fillOpacity: 0.9,
      weight: 1,
    }).addTo(state.frameLayer);
  }

  const current = data.sparse[state.frameIdx];
  L.circleMarker([current.latitude, current.longitude], {
    radius: 8,
    color: "rgb(25,35,55)",
    fillColor: "rgb(255,191,72)",
    fillOpacity: 1,
    weight: 1.5,
  }).addTo(state.frameLayer);
}

function renderBarcode() {
  const data = state.data;
  const sparse = data.sparse;
  const visible = state.frameIdx < 0 ? [] : sparse.slice(0, state.frameIdx + 1);
  const padMs = 20 * 60 * 1000;
  const minTs = sparse[0].timestamp * 1000 - padMs;
  const maxTs = sparse[sparse.length - 1].timestamp * 1000 + padMs;
  const currentTs = state.frameIdx < 0 ? null : sparse[state.frameIdx].timestamp;

  const shapes = [];

  if (currentTs !== null) {
    for (const run of data.stop_runs) {
      if (run.start_timestamp > currentTs) continue;
      const endTs = Math.min(run.end_timestamp, currentTs);
      const c = data.cluster_colors[String(run.cluster)] || [130, 138, 150];
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "y",
        x0: new Date(run.start_timestamp * 1000),
        x1: new Date(endTs * 1000),
        y0: 0,
        y1: 1,
        fillcolor: toRgba(c, 0.18),
        line: { width: 0 },
        layer: "below",
      });
    }
  }

  for (const p of visible) {
    const ts = new Date(p.timestamp * 1000);
    shapes.push({
      type: "line",
      xref: "x",
      yref: "y",
      x0: ts,
      x1: ts,
      y0: 0.2,
      y1: 0.8,
      line: { color: "rgba(20,20,20,0.6)", width: 1 },
    });
    if (p.cluster >= 0) {
      const c = data.cluster_colors[String(p.cluster)] || [130, 138, 150];
      shapes.push({
        type: "line",
        xref: "x",
        yref: "y",
        x0: ts,
        x1: ts,
        y0: 0.2,
        y1: 0.8,
        line: { color: toRgba(c, 0.95), width: 1.6 },
      });
    }
  }

  if (currentTs !== null) {
    shapes.push({
      type: "line",
      xref: "x",
      yref: "y",
      x0: new Date(currentTs * 1000),
      x1: new Date(currentTs * 1000),
      y0: 0,
      y1: 1,
      line: { color: "rgb(210,35,45)", width: 2 },
    });
  }

  const trace = {
    x: [new Date(minTs), new Date(maxTs)],
    y: [0, 1],
    mode: "lines",
    line: { color: "rgba(0,0,0,0)" },
    hoverinfo: "skip",
    showlegend: false,
  };

  const layout = {
    margin: { t: 10, r: 10, b: 40, l: 8 },
    paper_bgcolor: "#f8f9fc",
    plot_bgcolor: "#f8f9fc",
    xaxis: {
      type: "date",
      range: [new Date(minTs), new Date(maxTs)],
      showgrid: false,
      tickformat: "%H:%M\n%b %d",
    },
    yaxis: {
      range: [0, 1],
      showticklabels: false,
      showgrid: false,
      zeroline: false,
    },
    shapes,
  };

  Plotly.react("barcode", [trace], layout, { displayModeBar: false, responsive: true });
}

function renderFrame() {
  if (!state.data) return;
  renderFrameMapLayers();
  renderBarcode();
}

function clearTimer() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function play() {
  if (!state.data) return;
  clearTimer();
  if (state.frameIdx >= state.data.sparse.length - 1) {
    setFrame(-1);
  }
  const speed = Number(document.getElementById("speedInput").value || 110);
  state.timer = setInterval(() => {
    if (!state.data) {
      clearTimer();
      return;
    }
    if (state.frameIdx >= state.data.sparse.length - 1) {
      clearTimer();
      return;
    }
    setFrame(state.frameIdx + 1);
  }, speed);
}

async function runDashboard() {
  clearTimer();
  if (!state.options) return;

  const payload = {
    user_id: document.getElementById("userSelect").value,
    algorithm: document.getElementById("algorithmSelect").value,
    beta_ping_proxy: Number(document.getElementById("betaPingInput").value),
    noise_m: Number(document.getElementById("noiseInput").value),
    max_pings: Number(document.getElementById("maxPingsInput").value),
    seed: Number(document.getElementById("seedInput").value),
    params: readParams(),
  };

  setStatus("Running...");
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Failed to run dashboard");
    }

    state.data = data;
    renderMetricCards(data.meta);
    initMap();

    const b = data.meta.bounds;
    state.map.fitBounds([[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]]);

    const slider = document.getElementById("frameSlider");
    slider.min = "-1";
    slider.max = String(Math.max(0, data.sparse.length - 1));
    setFrame(-1);
    play();
    setStatus("Ready");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function loadOptions() {
  setStatus("Loading options...");
  try {
    const res = await fetch("/api/options");
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Failed to load options");
    }
    state.options = data;
    populateControls(data);
    setStatus("Ready");
  } catch (err) {
    setStatus(err.message, true);
  }
}

function attachEvents() {
  document.getElementById("runButton").addEventListener("click", runDashboard);
  document.getElementById("playBtn").addEventListener("click", play);
  document.getElementById("pauseBtn").addEventListener("click", clearTimer);
  document.getElementById("resetBtn").addEventListener("click", () => setFrame(-1));
  document.getElementById("frameSlider").addEventListener("input", (e) => {
    clearTimer();
    setFrame(Number(e.target.value));
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  attachEvents();
  initMap();
  await loadOptions();
});
