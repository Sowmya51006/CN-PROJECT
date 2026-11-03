/**
 * frontend/js/app.js — NETPULSE AI
 * Neon Network Studio Client-Side Application
 * Handles API interactions, input form management, predictions, and offline charting.
 */

const API_BASE = "";

// ═══════════════════════════════════════════════════════════════
// 1. NETWORK ANALYZER (PAGE 1)
// ═══════════════════════════════════════════════════════════════
let currentMode = "manual";
let datasetPresets = [];
let activeRecordData = null;

function initAnalyzer() {
  const modeManualBtn = document.getElementById("modeManual");
  const modeDatasetBtn = document.getElementById("modeDataset");
  const manualContainer = document.getElementById("manualInputContainer");
  const datasetContainer = document.getElementById("datasetInputContainer");
  const analyzeBtn = document.getElementById("btnAnalyze");

  if (!analyzeBtn) return; // Not on analyzer page

  // Mode Switchers
  if (modeManualBtn && modeDatasetBtn) {
    modeManualBtn.addEventListener("click", () => {
      currentMode = "manual";
      modeManualBtn.classList.add("active");
      modeDatasetBtn.classList.remove("active");
      manualContainer.style.display = "block";
      datasetContainer.style.display = "none";
    });

    modeDatasetBtn.addEventListener("click", () => {
      currentMode = "dataset";
      modeDatasetBtn.classList.add("active");
      modeManualBtn.classList.remove("active");
      manualContainer.style.display = "none";
      datasetContainer.style.display = "block";
      if (datasetPresets.length === 0) {
        loadDatasetPresets();
      }
    });
  }

  // Preset Buttons for Option A
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const p = btn.dataset;
      setManualValues({
        rtt: p.rtt, loss: p.loss, jitter: p.jitter,
        down: p.down, up: p.up, gw: p.gw,
        dns: p.dns, http: p.http, wifi: p.wifi
      });
    });
  });

  // Analyze Button Click Handler
  analyzeBtn.addEventListener("click", performAnalysis);

  // Preload presets in background
  loadDatasetPresets();
}

function setManualValues(v) {
  if (v.rtt !== undefined) document.getElementById("inputRtt").value = v.rtt;
  if (v.loss !== undefined) document.getElementById("inputLoss").value = v.loss;
  if (v.jitter !== undefined) document.getElementById("inputJitter").value = v.jitter;
  if (v.down !== undefined) document.getElementById("inputDown").value = v.down;
  if (v.up !== undefined) document.getElementById("inputUp").value = v.up;
  if (v.gw !== undefined) document.getElementById("inputGw").value = v.gw;
  if (v.dns !== undefined) document.getElementById("inputDns").value = v.dns;
  if (v.http !== undefined) document.getElementById("inputHttp").value = v.http;
  if (v.wifi !== undefined) document.getElementById("inputWifi").value = v.wifi;
}

async function loadDatasetPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/dataset/presets`);
    const data = await res.json();
    if (data.presets && data.presets.length > 0) {
      datasetPresets = data.presets;
      const select = document.getElementById("datasetRecordSelect");
      if (select) {
        select.innerHTML = "";
        datasetPresets.forEach((p, idx) => {
          const opt = document.createElement("option");
          opt.value = idx;
          opt.textContent = p.label;
          select.appendChild(opt);
        });

        select.addEventListener("change", (e) => {
          selectPresetRecord(parseInt(e.target.value));
        });

        // Select first preset by default
        selectPresetRecord(0);
      }
    }
  } catch (err) {
    console.error("Failed to load dataset presets:", err);
  }
}

function selectPresetRecord(idx) {
  if (!datasetPresets[idx]) return;
  const p = datasetPresets[idx];
  activeRecordData = p.full_features || p.measurements;

  const m = p.measurements;
  const strip = document.getElementById("presetPreviewStrip");
  if (strip) {
    strip.innerHTML = `
      <div class="preview-item">Record: <b>#${p.record_id}</b></div>
      <div class="preview-item">Condition: <b>${p.category}</b></div>
      <div class="preview-item">Latency: <b>${m.rtt_avg_ms} ms</b></div>
      <div class="preview-item">Packet Loss: <b>${m.packet_loss_pct}%</b></div>
      <div class="preview-item">Jitter: <b>${m.jitter_ms} ms</b></div>
      <div class="preview-item">Download: <b>${m.download_mbps} Mbps</b></div>
      <div class="preview-item">Gateway RTT: <b>${m.gateway_rtt_avg_ms} ms</b></div>
      <div class="preview-item">Wi-Fi: <b>${m.wifi_signal_pct}%</b></div>
    `;
  }
}

async function performAnalysis() {
  const analyzeBtn = document.getElementById("btnAnalyze");
  const errBox = document.getElementById("errorBanner");
  const outSection = document.getElementById("outputSection");

  if (errBox) errBox.style.display = "none";

  let payload = {};

  if (currentMode === "manual") {
    const rtt = parseFloat(document.getElementById("inputRtt").value);
    const loss = parseFloat(document.getElementById("inputLoss").value);
    const jitter = parseFloat(document.getElementById("inputJitter").value);
    const down = parseFloat(document.getElementById("inputDown").value);
    const up = parseFloat(document.getElementById("inputUp").value);
    const gw = parseFloat(document.getElementById("inputGw").value);
    const dns = parseFloat(document.getElementById("inputDns").value);
    const http = parseFloat(document.getElementById("inputHttp").value);
    const wifi = parseFloat(document.getElementById("inputWifi").value);

    // Validation
    if (isNaN(rtt) || isNaN(loss) || isNaN(jitter) || isNaN(down) || isNaN(up)) {
      showError("Please enter valid numeric values for all network measurements.");
      return;
    }
    if (rtt < 0 || loss < 0 || jitter < 0 || down < 0 || up < 0 || gw < 0) {
      showError("Measurements cannot be negative numbers.");
      return;
    }
    if (loss > 100 || wifi > 100) {
      showError("Packet loss and Wi-Fi percentages cannot exceed 100%.");
      return;
    }

    payload = {
      rtt_avg_ms: rtt,
      packet_loss_pct: loss,
      jitter_ms: jitter,
      download_mbps: down,
      upload_mbps: up,
      gateway_rtt_avg_ms: gw,
      dns_resolve_ms: dns,
      http_ms: http,
      wifi_signal_pct: wifi,
    };
  } else {
    if (!activeRecordData) {
      showError("Please select a benchmark dataset record.");
      return;
    }
    payload = activeRecordData;
  }

  // Display Spinner
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span class="spinner"></span> ANALYZING TELEMETRY...`;

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok || result.error) {
      showError(result.error || "Inference server error.");
      return;
    }

    renderPredictionResult(result);
  } catch (err) {
    showError("Could not reach backend API server. Please verify Flask is running.");
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `⚡ ANALYZE NETWORK`;
  }
}

function showError(msg) {
  const errBox = document.getElementById("errorBanner");
  if (errBox) {
    errBox.textContent = `⚠ ${msg}`;
    errBox.style.display = "block";
    errBox.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    alert(msg);
  }
}

function renderPredictionResult(res) {
  const outSection = document.getElementById("outputSection");
  const card = document.getElementById("resultCard");
  const classEl = document.getElementById("resClass");
  const confEl = document.getElementById("resConf");
  const summaryEl = document.getElementById("resSummary");
  const pillsEl = document.getElementById("resPills");
  const recsContainer = document.getElementById("recsContainer");

  // Overall Performance Status (Good Performance vs Poor Performance)
  const perfBadgeEl = document.getElementById("resPerfBadge");
  const perfIconEl = document.getElementById("resPerfIcon");
  const perfRatingEl = document.getElementById("resPerfRating");
  const perfSubEl = document.getElementById("resPerfSub");

  const isGood = res.performance_status ? (res.performance_status === "good") : (res.predicted_class === "Normal");
  const ratingText = res.performance_rating || (isGood ? "GOOD PERFORMANCE" : "POOR PERFORMANCE");
  const ratingIcon = res.performance_icon || (isGood ? "🟢" : "🔴");
  const ratingSub = res.performance_sub || (isGood
    ? "Optimal network health. Telemetry demonstrates low latency, zero packet loss, and robust transmission capacity."
    : "Elevated latency, packet loss, or delay variance detected exceeding nominal operational standards.");

  if (perfRatingEl) perfRatingEl.textContent = ratingText;
  if (perfIconEl) perfIconEl.textContent = ratingIcon;
  if (perfSubEl) perfSubEl.textContent = ratingSub;

  if (perfBadgeEl) {
    if (isGood) {
      perfBadgeEl.className = "perf-status-badge badge-good";
    } else if (res.level === "warning") {
      perfBadgeEl.className = "perf-status-badge badge-poor-warning";
    } else {
      perfBadgeEl.className = "perf-status-badge badge-poor";
    }
  }

  // Style class card
  card.className = `output-card res-${res.level}`;
  classEl.textContent = `${res.icon} ${res.class_title}`;
  classEl.style.color = res.color;
  confEl.innerHTML = `Model Confidence: <b>${res.confidence}%</b>`;
  summaryEl.innerHTML = `<b>Diagnosis Summary:</b> ${res.summary}`;

  // Recap pills
  const inps = res.inputs_recap;
  pillsEl.innerHTML = `
    <span class="output-recap-pill">Latency (RTT): <b>${inps.rtt_avg_ms} ms</b></span>
    <span class="output-recap-pill">Packet Loss: <b>${inps.packet_loss_pct}%</b></span>
    <span class="output-recap-pill">Jitter: <b>${inps.jitter_ms} ms</b></span>
    <span class="output-recap-pill">Download: <b>${inps.download_mbps} Mbps</b></span>
    <span class="output-recap-pill">Gateway RTT: <b>${inps.gateway_rtt_avg_ms} ms</b></span>
    <span class="output-recap-pill">Wi-Fi: <b>${inps.wifi_signal_pct}%</b></span>
  `;

  // Render Recommendations (Max 3)
  recsContainer.innerHTML = "";
  (res.recommendations || []).forEach(r => {
    const rCard = document.createElement("div");
    rCard.className = `rec-card rec-${res.level}`;
    rCard.innerHTML = `
      <div class="rec-title">🛠️ ${r.problem}</div>
      <div class="rec-body"><b>1. Identified Issue:</b> ${r.problem}</div>
      <div class="rec-body"><b>2. Why It Matters:</b> ${r.why}</div>
      <div class="rec-body"><b>3. What to Investigate:</b> ${r.investigate}</div>
    `;
    recsContainer.appendChild(rCard);
  });

  // Render Congestion Control & Mitigation Techniques
  const congSection = document.getElementById("congestionSection");
  const congContainer = document.getElementById("congestionContainer");
  if (congContainer && res.congestion_techniques && res.congestion_techniques.length > 0) {
    congContainer.innerHTML = "";
    res.congestion_techniques.forEach(tech => {
      const card = document.createElement("div");
      card.className = "congestion-card";
      card.innerHTML = `
        <div class="congestion-header-row">
          <div class="congestion-name">${tech.icon} ${tech.name}</div>
          <span class="congestion-badge">${tech.category}</span>
        </div>
        <div>
          <div class="congestion-section-title">How It Works (Mechanism):</div>
          <div class="congestion-text">${tech.mechanism}</div>
        </div>
        <div>
          <div class="congestion-section-title">Practical Implementation:</div>
          <div class="congestion-cmd">${tech.implementation}</div>
        </div>
      `;
      congContainer.appendChild(card);
    });
    if (congSection) congSection.style.display = "block";
  }

  outSection.style.display = "block";
  outSection.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ═══════════════════════════════════════════════════════════════
// 2. DATASET EXPLORER (PAGE 2)
// ═══════════════════════════════════════════════════════════════
async function initDatasetPage() {
  const tableBody = document.getElementById("datasetTableBody");
  if (!tableBody) return;

  const catSelect = document.getElementById("filterCategory");
  const limitSelect = document.getElementById("pageSizeSelect");
  const uploadInput = document.getElementById("csvUploadInput");
  const uploadBtn = document.getElementById("btnUploadFile");

  async function loadTable() {
    const cat = catSelect ? catSelect.value : "All";
    const limit = limitSelect ? limitSelect.value : 50;

    tableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem">Loading telemetry records...</td></tr>`;

    try {
      const res = await fetch(`${API_BASE}/api/dataset/records?limit=${limit}&category=${cat}`);
      const data = await res.json();

      tableBody.innerHTML = "";
      if (!data.records || data.records.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem">No matching records found.</td></tr>`;
        return;
      }

      data.records.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>#${r._index}</td>
          <td>${r.timestamp_local || "-"}</td>
          <td><b style="color:${getCatColor(r.event_category)}">${r.event_category}</b></td>
          <td>${r.rtt_avg_ms} ms</td>
          <td>${r.packet_loss_pct}%</td>
          <td>${r.jitter_ms} ms</td>
          <td>${r.download_mbps} Mbps</td>
          <td>${r.gateway_rtt_avg_ms} ms</td>
        `;
        tableBody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
      tableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:#FB7185;padding:2rem">Failed to load dataset records.</td></tr>`;
    }
  }

  if (catSelect) catSelect.addEventListener("change", loadTable);
  if (limitSelect) limitSelect.addEventListener("change", loadTable);

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener("click", async () => {
      if (!uploadInput.files || uploadInput.files.length === 0) {
        alert("Please select a CSV file first.");
        return;
      }
      const formData = new FormData();
      formData.append("file", uploadInput.files[0]);

      uploadBtn.disabled = true;
      uploadBtn.textContent = "Uploading...";

      try {
        const res = await fetch(`${API_BASE}/api/dataset/upload`, {
          method: "POST",
          body: formData,
        });
        const d = await res.json();
        if (res.ok) {
          alert(`Success! Loaded ${d.rows_loaded} rows.`);
          loadTable();
        } else {
          alert(d.error || "Upload failed.");
        }
      } catch (e) {
        alert("Error connecting to upload endpoint.");
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload & Clean Dataset";
      }
    });
  }

  loadTable();
}

function getCatColor(cat) {
  const map = {
    "Normal": "#35D399",
    "HighLatency": "#FBBF24",
    "PacketLoss": "#FB7185",
    "HighJitter": "#9673FF",
    "Outage": "#FB7185"
  };
  return map[cat] || "#00D9FF";
}


// ═══════════════════════════════════════════════════════════════
// 3. ANALYTICS (PAGE 3) — ZERO-DEPENDENCY OFFLINE CHARTS
// ═══════════════════════════════════════════════════════════════
let cachedAnalytics = null;

async function initAnalyticsPage() {
  const trendCanvas = document.getElementById("trendCanvas");
  const distCanvas = document.getElementById("distCanvas");

  if (!trendCanvas || !distCanvas) return;

  if (cachedAnalytics) {
    renderAnalytics(cachedAnalytics);
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/analytics/data`);
    const data = await res.json();
    cachedAnalytics = data;
    renderAnalytics(data);
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

function renderAnalytics(data) {
  const trendCanvas = document.getElementById("trendCanvas");
  const distCanvas = document.getElementById("distCanvas");

  if (data.metrics) {
    const elLat = document.getElementById("metricLatency");
    const elLoss = document.getElementById("metricLoss");
    const elDown = document.getElementById("metricDown");
    const elUp = document.getElementById("metricUptime");
    if (elLat) elLat.innerHTML = `${data.metrics.avg_latency}<span class="metric-unit">ms</span>`;
    if (elLoss) elLoss.innerHTML = `${data.metrics.avg_loss}<span class="metric-unit">%</span>`;
    if (elDown) elDown.innerHTML = `${data.metrics.avg_down}<span class="metric-unit">Mbps</span>`;
    if (elUp) elUp.innerHTML = `${data.metrics.normal_pct}<span class="metric-unit">%</span>`;
  }

  // Draw charts in next frame after DOM geometry is ready
  requestAnimationFrame(() => {
    if (trendCanvas && data.trend && data.trend.length > 0) {
      drawTrendLineChart(trendCanvas, data.trend);
    }
    if (distCanvas && data.distribution) {
      drawDistributionBarChart(distCanvas, data.distribution);
    }
  });
}

function getCanvasSize(canvas) {
  const wrap = canvas.parentElement;
  let w = wrap ? wrap.getBoundingClientRect().width : canvas.getBoundingClientRect().width;
  let h = wrap ? wrap.getBoundingClientRect().height : canvas.getBoundingClientRect().height;

  // Resilient fallback if container was temporarily hidden
  if (!w || w < 200) w = 950;
  if (!h || h < 100) h = 440;

  return { w: Math.round(w), h: Math.round(h) };
}

// Interactive Pure Canvas Line Chart for Latency Trend
function drawTrendLineChart(canvas, points) {
  if (!canvas || !points || points.length === 0) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const { w, h } = getCanvasSize(canvas);

  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  const padL = 65, padR = 30, padT = 30, padB = 45;
  ctx.clearRect(0, 0, w, h);

  const rtts = points.map(p => p.rtt);
  const minVal = 0;
  const maxVal = Math.max(100, Math.ceil(Math.max(...rtts) / 50) * 50);

  // Background Gridlines
  ctx.strokeStyle = "#223354";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#CBD5E1";
  ctx.font = "14px 'Inter', sans-serif";
  ctx.textAlign = "right";

  const gridSteps = 5;
  for (let i = 0; i <= gridSteps; i++) {
    const yVal = minVal + (maxVal - minVal) * (i / gridSteps);
    const yPos = h - padB - ((yVal - minVal) / (maxVal - minVal)) * (h - padT - padB);

    ctx.beginPath();
    ctx.moveTo(padL, yPos);
    ctx.lineTo(w - padR, yPos);
    ctx.stroke();

    ctx.fillText(`${Math.round(yVal)} ms`, padL - 10, yPos + 5);
  }

  // Draw Area Fill & Line
  const chartW = Math.max(100, w - padL - padR);
  const chartH = Math.max(50, h - padT - padB);

  ctx.beginPath();
  points.forEach((p, i) => {
    const x = padL + (i / (points.length - 1)) * chartW;
    const y = h - padB - ((p.rtt - minVal) / (maxVal - minVal)) * chartH;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  // Stroke Line
  ctx.strokeStyle = "#00D9FF";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Gradient Area under curve
  ctx.lineTo(padL + chartW, h - padB);
  ctx.lineTo(padL, h - padB);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padT, 0, h - padB);
  grad.addColorStop(0, "rgba(0, 217, 255, 0.28)");
  grad.addColorStop(1, "rgba(0, 217, 255, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();

  // Axis Labels
  ctx.fillStyle = "#CBD5E1";
  ctx.font = "15px 'Inter', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Longitudinal Measurement Timeline (19 Days Telemetry)", padL + chartW / 2, h - 12);
}

// Pure Canvas Bar Chart for Condition Distribution
function drawDistributionBarChart(canvas, dist) {
  if (!canvas || !dist) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const { w, h } = getCanvasSize(canvas);

  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  const padL = 70, padR = 30, padT = 35, padB = 60;
  ctx.clearRect(0, 0, w, h);

  const keys = Object.keys(dist);
  const values = Object.values(dist);
  const maxVal = Math.max(...values, 1000);

  // Background Gridlines
  ctx.strokeStyle = "#223354";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#CBD5E1";
  ctx.font = "14px 'Inter', sans-serif";
  ctx.textAlign = "right";

  const gridSteps = 4;
  for (let i = 0; i <= gridSteps; i++) {
    const val = (maxVal / gridSteps) * i;
    const yPos = h - padB - (val / maxVal) * (h - padT - padB);

    ctx.beginPath();
    ctx.moveTo(padL, yPos);
    ctx.lineTo(w - padR, yPos);
    ctx.stroke();

    ctx.fillText(`${(val / 1000).toFixed(0)}k`, padL - 10, yPos + 5);
  }

  // Draw Bars
  const chartW = Math.max(100, w - padL - padR);
  const chartH = Math.max(50, h - padT - padB);
  const slotW = chartW / keys.length;
  const barW = Math.min(85, slotW * 0.65);

  keys.forEach((cat, i) => {
    const val = dist[cat];
    const barH = Math.max(4, (val / maxVal) * chartH);
    const x = padL + i * slotW + (slotW - barW) / 2;
    const y = h - padB - barH;

    ctx.fillStyle = getCatColor(cat);
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, barW, barH, [6, 6, 0, 0]);
    } else {
      ctx.rect(x, y, barW, barH);
    }
    ctx.fill();

    // Value on top of bar
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 15px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(val.toLocaleString(), x + barW / 2, y - 8);

    // Label below bar
    ctx.fillStyle = "#CBD5E1";
    ctx.font = "14px 'Inter', sans-serif";
    ctx.fillText(cat, x + barW / 2, h - padB + 26);
  });
}

// Window resize handler for fluid charts
window.addEventListener("resize", () => {
  if (cachedAnalytics) {
    const trendCanvas = document.getElementById("trendCanvas");
    if (trendCanvas && trendCanvas.offsetParent !== null) {
      renderAnalytics(cachedAnalytics);
    }
  }
});


// ═══════════════════════════════════════════════════════════════
// 4. VIEW ROUTER & NAVIGATION (Strict Mutual Exclusivity)
// ═══════════════════════════════════════════════════════════════
function initNavigation() {
  const navLinks = document.querySelectorAll(".nav-link[data-page]");
  if (!navLinks || navLinks.length === 0) return;

  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      const targetPageId = link.getAttribute("data-page");
      const targetSection = document.getElementById(targetPageId);

      // If the target container exists on this page, perform strict tab switching
      if (targetSection) {
        e.preventDefault();

        // 1. Hide ALL page views strictly
        document.querySelectorAll(".page-view").forEach(p => {
          p.classList.remove("active");
          p.style.display = "none";
        });

        // 2. Remove active state from all nav links
        navLinks.forEach(l => l.classList.remove("active"));

        // 3. Show ONLY the chosen page view
        targetSection.classList.add("active");
        targetSection.style.display = "block";
        link.classList.add("active");

        // 4. Scroll smoothly to top
        window.scrollTo({ top: 0, behavior: "smooth" });

        // 5. Initialize target data if not already done
        if (targetPageId === "view-dataset") {
          initDatasetPage();
        } else if (targetPageId === "view-analytics") {
          requestAnimationFrame(() => {
            setTimeout(() => {
              initAnalyticsPage();
            }, 60);
          });
        }
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION ROUTER
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initAnalyzer();
  initDatasetPage();
  initAnalyticsPage();
});

