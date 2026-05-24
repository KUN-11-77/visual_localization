"""
Integrated Web UI for Visual Localization Pipeline.

Flask-powered interactive interface meeting task.md requirements:
- Accepts user input (config selection, parameters)
- Displays results (metrics, per-frame details, visualization)
- Saves to user-specified locations
- Single-page HTML application with tabbed navigation

Usage:
  python scripts/web_ui.py [--port 5000]
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
AR_DEMO_DIR = PROJECT_ROOT / "outputs" / "ar_demo"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# We'll try Flask, fall back gracefully
try:
    from flask import Flask, render_template_string, request, jsonify, Response
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Localization — Interactive Pipeline</title>
<style>
:root {
  --bg: #1a1a2e; --surface: #16213e; --surface2: #0f3460;
  --accent: #e94560; --accent2: #4CAF50; --text: #eee; --text2: #aaa;
  --border: #2a2a4a; --radius: 10px;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
.container { max-width:1400px; margin:0 auto; padding:20px; }
header { text-align:center; padding:30px 0 20px; border-bottom:2px solid var(--accent); margin-bottom:30px; }
header h1 { font-size:28px; letter-spacing:1px; }
header p { color:var(--text2); margin-top:6px; font-size:14px; }

/* Tabs */
.tabs { display:flex; gap:4px; margin-bottom:24px; flex-wrap:wrap; }
.tab-btn { padding:10px 24px; border:none; background:var(--surface); color:var(--text2);
  cursor:pointer; border-radius:var(--radius) var(--radius) 0 0; font-size:14px; transition:.2s; }
.tab-btn:hover { background:var(--surface2); color:var(--text); }
.tab-btn.active { background:var(--accent); color:#fff; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

/* Cards */
.card { background:var(--surface); border-radius:var(--radius); padding:24px; margin-bottom:20px;
  border:1px solid var(--border); }
.card h2 { font-size:18px; margin-bottom:16px; color:var(--accent2); border-bottom:1px solid var(--border); padding-bottom:8px; }
.card h3 { font-size:15px; margin:16px 0 8px; color:var(--text); }

/* Tables */
table { width:100%; border-collapse:collapse; font-size:13px; margin:10px 0; }
th { background:var(--surface2); color:var(--text); padding:10px 8px; text-align:left; font-weight:600; white-space:nowrap; }
td { padding:8px; border-bottom:1px solid var(--border); }
tr:hover td { background:rgba(233,69,96,.08); }

/* Buttons & Inputs */
.btn { padding:10px 20px; border:none; border-radius:6px; cursor:pointer; font-size:14px;
  font-weight:600; transition:.2s; display:inline-flex; align-items:center; gap:6px; }
.btn-primary { background:var(--accent); color:#fff; }
.btn-primary:hover { filter:brightness(1.2); }
.btn-green { background:var(--accent2); color:#fff; }
.btn-green:hover { filter:brightness(1.2); }
.btn-outline { background:transparent; border:1px solid var(--border); color:var(--text); }
.btn-outline:hover { background:var(--surface2); }
.btn-sm { padding:5px 12px; font-size:12px; }
.btn:disabled { opacity:.5; cursor:not-allowed; }

select, input[type=text], input[type=number] { padding:10px 14px; border:1px solid var(--border);
  border-radius:6px; background:var(--bg); color:var(--text); font-size:14px; width:100%; }
select:focus, input:focus { outline:none; border-color:var(--accent); }

.form-row { display:flex; gap:12px; align-items:end; flex-wrap:wrap; margin-bottom:12px; }
.form-group { flex:1; min-width:180px; }
.form-group label { display:block; margin-bottom:4px; font-size:12px; color:var(--text2); text-transform:uppercase; letter-spacing:.5px; }

/* Status badges */
.badge { display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; }
.badge-ok { background:rgba(76,175,80,.2); color:var(--accent2); }
.badge-fail { background:rgba(233,69,96,.2); color:var(--accent); }
.badge-warn { background:rgba(255,152,0,.2); color:#FF9800; }

/* Metric grid */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }
.metric-item { background:var(--bg); border-radius:8px; padding:16px; text-align:center; }
.metric-item .val { font-size:32px; font-weight:800; }
.metric-item .lbl { font-size:11px; color:var(--text2); margin-top:4px; text-transform:uppercase; }
.val-high { color:var(--accent2); }
.val-mid { color:#FF9800; }
.val-low { color:var(--accent); }

/* Experiment card in list */
.exp-card { display:flex; align-items:center; padding:14px 20px; background:var(--bg);
  border-radius:8px; margin-bottom:8px; cursor:pointer; transition:.2s; border:1px solid transparent; }
.exp-card:hover { border-color:var(--accent); }
.exp-card .exp-name { flex:1; font-weight:600; }
.exp-card .exp-meta { font-size:12px; color:var(--text2); margin:0 16px; text-align:right; }
.exp-card .exp-recall { font-size:18px; font-weight:700; }

/* Progress bar */
.progress-wrap { background:var(--bg); border-radius:8px; height:8px; margin:10px 0; overflow:hidden; }
.progress-bar { height:100%; background:var(--accent); width:0%; transition:width .3s; border-radius:8px; }

/* Log output */
.log-output { background:#0a0a1a; color:#0f0; font-family:'Consolas',monospace; font-size:12px;
  padding:16px; border-radius:8px; max-height:300px; overflow-y:auto; white-space:pre-wrap; }

/* Image grid */
.img-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }
.img-card { background:var(--bg); border-radius:8px; overflow:hidden; }
.img-card img { width:100%; display:block; }
.img-card .caption { padding:8px 12px; font-size:11px; color:var(--text2); }

/* Modal */
.modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7);
  z-index:1000; justify-content:center; align-items:center; }
.modal-overlay.show { display:flex; }
.modal { background:var(--surface); border-radius:var(--radius); padding:30px; max-width:700px;
  width:90%; max-height:85vh; overflow-y:auto; border:1px solid var(--border); }
.modal h3 { margin-bottom:16px; }

/* Toast */
.toast { position:fixed; bottom:24px; right:24px; padding:14px 24px; border-radius:8px;
  color:#fff; font-weight:600; z-index:2000; animation:fadeInUp .3s; }
.toast-success { background:var(--accent2); }
.toast-error { background:var(--accent); }
@keyframes fadeInUp { from{opacity:0;transform:translateY(20px);} to{opacity:1;transform:translateY(0);} }

/* Filter */
.filter-bar { display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap; align-items:center; }
.filter-bar input { max-width:300px; }
.filter-bar select { max-width:180px; }

/* Per-frame stats */
.stats-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
.stat-chip { background:var(--bg); padding:8px 16px; border-radius:20px; font-size:13px; }
.stat-chip strong { color:var(--accent); }

/* Spinner */
.spinner { display:inline-block; width:18px; height:18px; border:2px solid var(--text2);
  border-top-color:var(--accent); border-radius:50%; animation:spin .6s linear infinite; }
@keyframes spin { to{transform:rotate(360deg);} }

/* Responsive */
@media(max-width:768px) {
  .form-row { flex-direction:column; }
  .tabs { flex-direction:column; }
  .tab-btn { border-radius:var(--radius); }
  .metric-grid { grid-template-columns:repeat(2,1fr); }
}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>Visual Localization — Interactive Pipeline</h1>
  <p>OpenXRLab XRLocalization Extended | HLoc: Retrieval → Detection → Matching → PnP</p>
</header>

<!-- Tab Navigation -->
<div class="tabs">
  <button class="tab-btn active" data-tab="dashboard"> Dashboard</button>
  <button class="tab-btn" data-tab="run"> Run Experiment</button>
  <button class="tab-btn" data-tab="results"> Results Viewer</button>
  <button class="tab-btn" data-tab="ardemo"> AR Demo</button>
  <button class="tab-btn" data-tab="export"> Export</button>
</div>

<!-- ==================== DASHBOARD ==================== -->
<div id="tab-dashboard" class="tab-panel active">
  <div class="card">
    <h2>Experiment Overview</h2>
    <div id="dashboard-metric-grid" class="metric-grid"></div>
  </div>
  <div class="card">
    <h2>Recall Comparison (0.25m, 2&deg;)</h2>
    <div id="dashboard-recall-table"></div>
  </div>
  <div class="card">
    <h2>Pipeline Summary</h2>
    <table id="dashboard-summary-table">
      <thead><tr><th>Experiment</th><th>Retrieval</th><th>Detection</th><th>Matcher</th><th>Dataset</th><th>(0.25m,2&deg;)</th><th>(0.5m,5&deg;)</th><th>(5m,10&deg;)</th><th>Queries</th><th>Localized</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<!-- ==================== RUN EXPERIMENT ==================== -->
<div id="tab-run" class="tab-panel">
  <div class="card">
    <h2>Run Localization Experiment</h2>
    <div class="form-row">
      <div class="form-group">
        <label>Configuration</label>
        <select id="run-config"></select>
      </div>
      <div class="form-group">
        <label>Query Limit (0=all)</label>
        <input type="number" id="run-limit" value="5" min="0" max="10000">
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Build SP+SG SfM Model</label>
        <select id="run-build-sfm">
          <option value="0">No (use existing model)</option>
          <option value="1">Yes (rebuild SfM)</option>
        </select>
      </div>
    </div>
    <div style="display:flex;gap:12px;margin-top:8px;">
      <button class="btn btn-primary" id="btn-run" onclick="runExperiment()"> Run Experiment</button>
      <button class="btn btn-outline" id="btn-run-stop" onclick="stopRun()" style="display:none"> Stop</button>
    </div>
    <div id="run-progress" style="display:none;margin-top:16px;">
      <div class="progress-wrap"><div class="progress-bar" id="run-bar"></div></div>
      <p id="run-status" style="font-size:13px;color:var(--text2);margin-top:4px;"></p>
      <div class="log-output" id="run-log"></div>
    </div>
  </div>
</div>

<!-- ==================== RESULTS VIEWER ==================== -->
<div id="tab-results" class="tab-panel">
  <div class="card">
    <h2>Select Experiment</h2>
    <select id="results-select" onchange="loadResultDetail()" style="max-width:400px;"></select>
  </div>
  <div id="results-detail"></div>
</div>

<!-- ==================== AR DEMO ==================== -->
<div id="tab-ardemo" class="tab-panel">
  <div class="card">
    <h2>AR Demo Generator</h2>
    <div class="form-row">
      <div class="form-group">
        <label>Experiment</label>
        <select id="ar-experiment"></select>
      </div>
      <div class="form-group">
        <label>Frames (evenly sampled)</label>
        <input type="number" id="ar-limit" value="30" min="1" max="500">
      </div>
      <div class="form-group">
        <label>Cube Size (m, auto if empty)</label>
        <input type="text" id="ar-cube-size" placeholder="Auto-detect">
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Cube Distance (m in front of camera)</label>
        <input type="number" id="ar-distance" value="1.0" step="0.1" min="0.2" max="10">
      </div>
      <div class="form-group">
        <label>Face Alpha (0-1)</label>
        <input type="number" id="ar-alpha" value="0.55" step="0.05" min="0" max="1">
      </div>
      <div class="form-group">
        <label>Video FPS (0=skip)</label>
        <input type="number" id="ar-fps" value="5" min="0" max="30">
      </div>
    </div>
    <div style="display:flex;gap:12px;margin-top:8px;">
      <button class="btn btn-green" onclick="generateARDemo()"> Generate AR Demo</button>
      <button class="btn btn-outline" onclick="loadARDemos()"> Refresh Gallery</button>
    </div>
    <div id="ar-progress" style="display:none;margin-top:16px;">
      <div class="log-output" id="ar-log"></div>
    </div>
  </div>
  <div class="card">
    <h2>AR Demo Gallery</h2>
    <div class="filter-bar">
      <select id="ar-gallery-exp" onchange="loadARDemos()" style="max-width:300px;"></select>
    </div>
    <div class="img-grid" id="ar-gallery"></div>
  </div>
</div>

<!-- ==================== EXPORT ==================== -->
<div id="tab-export" class="tab-panel">
  <div class="card">
    <h2>Export Results</h2>
    <p style="color:var(--text2);margin-bottom:16px;">Save experiment results, reports, and AR demos to a user-specified location.</p>
    <div class="form-row">
      <div class="form-group">
        <label>Experiment</label>
        <select id="export-experiment"></select>
      </div>
      <div class="form-group">
        <label>Export Directory</label>
        <input type="text" id="export-dir" placeholder="e.g., D:/reports/">
      </div>
    </div>
    <button class="btn btn-primary" onclick="exportResults()"> Export</button>
    <button class="btn btn-outline" style="margin-left:8px;" onclick="exportAll()"> Export All Experiments</button>
    <div id="export-status" style="margin-top:12px;font-size:13px;"></div>
  </div>
  <div class="card">
    <h2>Package Final Deliverable</h2>
    <p style="color:var(--text2);margin-bottom:16px;">Generate the final submission package with all source code, results, and reports.</p>
    <div class="form-row">
      <div class="form-group">
        <label>Package Name</label>
        <input type="text" id="pkg-name" value="final-visual_localization-学号-姓名">
      </div>
      <div class="form-group">
        <label>Output Directory</label>
        <input type="text" id="pkg-dir" placeholder="e.g., D:/">
      </div>
    </div>
    <button class="btn btn-green" onclick="generatePackage()"> Generate Package</button>
    <div id="pkg-status" style="margin-top:12px;font-size:13px;"></div>
  </div>
</div>

</div><!-- .container -->

<script>
// ======================= GLOBAL STATE =======================
let runAbort = false;

// ======================= TAB SWITCHING =======================
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'dashboard') loadDashboard();
    if (btn.dataset.tab === 'results') loadResultList();
    if (btn.dataset.tab === 'ardemo') { loadARDemoExperiments(); loadARDemos(); }
    if (btn.dataset.tab === 'export') loadExportExperiments();
    if (btn.dataset.tab === 'run') loadConfigs();
  });
});

// ======================= API HELPERS =======================
async function api(url, opts={}) {
  try {
    const res = await fetch(url, opts);
    return await res.json();
  } catch(e) {
    console.error('API error:', e);
    return {error: e.message};
  }
}

// ======================= TOAST =======================
function toast(msg, type='success') {
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// ======================= DASHBOARD =======================
async function loadDashboard() {
  const data = await api('/api/results/all');
  if (data.error) return;

  let nExp = 0, bestRecall = 0, bestName = '', totalQueries = 0;
  if (data.experiments) {
    nExp = data.experiments.length;
    data.experiments.forEach(e => {
      const r = parseFloat(e.recall_025m_2deg || 0);
      if (r > bestRecall) { bestRecall = r; bestName = e.name; }
      totalQueries += parseInt(e.n_query || 0);
    });
  }
  document.getElementById('dashboard-metric-grid').innerHTML = `
    <div class="metric-item"><div class="val">${nExp}</div><div class="lbl">Experiments</div></div>
    <div class="metric-item"><div class="val val-high">${(bestRecall*100).toFixed(1)}%</div><div class="lbl">Best Recall (0.25m,2°)</div></div>
    <div class="metric-item"><div class="val">${bestName}</div><div class="lbl">Best Experiment</div></div>
    <div class="metric-item"><div class="val">${totalQueries}</div><div class="lbl">Total Queries</div></div>
  `;

  if (data.summary) {
    let rows = '';
    data.summary.forEach(r => {
      const cls = parseFloat(r['recall_0.25m_2deg']||0) > 0.6 ? 'val-high' :
                  parseFloat(r['recall_0.25m_2deg']||0) > 0.3 ? 'val-mid' : 'val-low';
      rows += `<tr>
        <td><strong>${r.experiment||''}</strong></td>
        <td>${r.retrieval||''}</td><td>${r.detector||''}</td><td>${r.matcher||''}</td>
        <td>${r.dataset||''} / ${r.scene||''}</td>
        <td style="color:${cls==='val-high'?'#4CAF50':cls==='val-mid'?'#FF9800':'#f44336'};font-weight:700">${(parseFloat(r['recall_0.25m_2deg']||0)*100).toFixed(1)}%</td>
        <td>${(parseFloat(r['recall_0.5m_5deg']||0)*100).toFixed(1)}%</td>
        <td>${(parseFloat(r['recall_5m_10deg']||0)*100).toFixed(1)}%</td>
        <td>${r.n_query||''}</td>
        <td>${r.n_localized||''}</td>
      </tr>`;
    });
    document.querySelector('#dashboard-summary-table tbody').innerHTML = rows;
  }

  // Recall comparison bar
  if (data.experiments) {
    let bars = data.experiments.map(e => {
      const r = parseFloat(e.recall_025m_2deg || 0) * 100;
      return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
        <span style="width:220px;font-size:12px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.name}</span>
        <div style="flex:1;background:var(--bg);border-radius:4px;height:22px;overflow:hidden">
          <div style="width:${r}%;height:100%;background:${r>60?'#4CAF50':r>30?'#FF9800':'#f44336'};border-radius:4px;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;font-size:11px;font-weight:700;min-width:${r>5?'auto':'50px'}">${r.toFixed(1)}%</div>
        </div>
      </div>`;
    }).join('');
    document.getElementById('dashboard-recall-table').innerHTML = bars;
  }
}

// ======================= RUN EXPERIMENT =======================
async function loadConfigs() {
  const data = await api('/api/configs');
  if (data.error) return;
  const sel = document.getElementById('run-config');
  sel.innerHTML = data.configs.map(c => `<option value="${c.path}">${c.name}</option>`).join('');
}

async function runExperiment() {
  const config = document.getElementById('run-config').value;
  const limit = document.getElementById('run-limit').value;
  const buildSfm = document.getElementById('run-build-sfm').value;

  runAbort = false;
  document.getElementById('btn-run').disabled = true;
  document.getElementById('btn-run-stop').style.display = 'inline-flex';
  document.getElementById('run-progress').style.display = 'block';
  document.getElementById('run-log').textContent = '';
  document.getElementById('run-bar').style.width = '0%';
  document.getElementById('run-status').textContent = 'Starting...';

  try {
    const res = await fetch('/api/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({config, limit:parseInt(limit), build_sfm: buildSfm==='1'})
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      if (runAbort) break;
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'progress') {
            document.getElementById('run-bar').style.width = msg.pct + '%';
            document.getElementById('run-status').textContent = msg.text;
          } else if (msg.type === 'log') {
            const log = document.getElementById('run-log');
            log.textContent += msg.text + '\n';
            log.scrollTop = log.scrollHeight;
          } else if (msg.type === 'done') {
            document.getElementById('run-status').textContent = 'Completed!';
            document.getElementById('run-bar').style.width = '100%';
            toast('Experiment completed successfully!', 'success');
          } else if (msg.type === 'error') {
            document.getElementById('run-status').textContent = 'Error: ' + msg.text;
            toast('Experiment failed: ' + msg.text, 'error');
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    toast('Connection error: ' + e.message, 'error');
  }

  document.getElementById('btn-run').disabled = false;
  document.getElementById('btn-run-stop').style.display = 'none';
  loadDashboard();
}

function stopRun() { runAbort = true; }

// ======================= RESULTS VIEWER =======================
async function loadResultList() {
  const data = await api('/api/results/list');
  if (data.error) return;
  const sel = document.getElementById('results-select');
  sel.innerHTML = '<option value="">-- Select --</option>' +
    data.results.map(r => `<option value="${r.path}">${r.name}</option>`).join('');
}

async function loadResultDetail() {
  const path = document.getElementById('results-select').value;
  if (!path) { document.getElementById('results-detail').innerHTML = ''; return; }

  const data = await api('/api/results/detail?path=' + encodeURIComponent(path));
  if (data.error) { document.getElementById('results-detail').innerHTML = '<p style="color:var(--accent)">Failed to load.</p>'; return; }

  // Metrics
  let metricHtml = '';
  if (data.results) {
    const r = data.results;
    const recallKeys = Object.keys(r).filter(k => k.startsWith('(')).sort();
    metricHtml += '<div class="metric-grid">';
    recallKeys.forEach(k => {
      const v = parseFloat(r[k]) * 100;
      metricHtml += `<div class="metric-item"><div class="val ${v>60?'val-high':v>30?'val-mid':'val-low'}">${v.toFixed(1)}%</div><div class="lbl">${k}</div></div>`;
    });
    metricHtml += `<div class="metric-item"><div class="val">${r.n_query||'?'}</div><div class="lbl">Total Queries</div></div>`;
    metricHtml += `<div class="metric-item"><div class="val val-high">${r.n_localized||'?'}</div><div class="lbl">Localized (0.25m,2°)</div></div>`;
    metricHtml += '</div>';
  }

  // Timing
  let timingHtml = '';
  if (data.timing && Object.keys(data.timing).length > 0) {
    timingHtml = '<h3>Timing (per query)</h3><table><tr><th>Stage</th><th>Mean (ms)</th><th>Std (ms)</th></tr>';
    for (const [key,val] of Object.entries(data.timing)) {
      if (typeof val === 'object' && val.mean_ms !== undefined) {
        timingHtml += `<tr><td>${key}</td><td>${val.mean_ms.toFixed(1)}</td><td>${val.std_ms.toFixed(1)}</td></tr>`;
      }
    }
    timingHtml += '</table>';
  }

  // Per-frame table with filtering
  let frameHtml = '';
  if (data.frames && data.frames.length > 0) {
    const nLoc = data.frames.filter(f => f.localized === 'True').length;
    frameHtml += `<div class="stats-row">
      <div class="stat-chip"><strong>${data.frames.length}</strong> frames</div>
      <div class="stat-chip"><strong>${nLoc}</strong> localized</div>
      <div class="stat-chip"><strong>${(nLoc/data.frames.length*100).toFixed(1)}%</strong> success rate</div>
    </div>`;
    frameHtml += `<div class="filter-bar">
      <input type="text" id="frame-search" placeholder="Search by image name..." oninput="filterFrames()">
      <select id="frame-status-filter" onchange="filterFrames()">
        <option value="all">All</option><option value="True">Localized</option><option value="False">Failed</option>
      </select>
    </div>`;
    frameHtml += `<div style="max-height:500px;overflow-y:auto;"><table id="frame-table">
      <thead><tr><th>#</th><th>Query Image</th><th>Kpts</th><th>Corr</th><th>Inliers</th><th>t_err (m)</th><th>r_err (°)</th><th>Top-1</th><th>Status</th></tr></thead>
      <tbody>`;
    data.frames.forEach((f,i) => {
      const loc = f.localized === 'True';
      const rowCls = loc ? '' : ' style="opacity:0.6"';
      frameHtml += `<tr data-status="${f.localized}" data-name="${f.query||''}"${rowCls}>
        <td>${i+1}</td><td style="font-size:12px;max-width:200px;overflow:hidden;text-overflow:ellipsis" title="${f.query||''}">${(f.query||'').split('/').pop()}</td>
        <td>${f.n_query_kpts||'?'}</td><td>${f.n_correspondences||'?'}</td><td>${f.n_inliers||'?'}</td>
        <td>${f.t_err||'?'}</td><td>${f.r_err||'?'}</td>
        <td style="font-size:11px;max-width:150px;overflow:hidden;text-overflow:ellipsis" title="${f.retrieved_top1||''}">${(f.retrieved_top1||'').split('/').pop()||'?'}</td>
        <td><span class="badge ${loc?'badge-ok':'badge-fail'}">${loc?'Localized':'Failed'}</span></td>
      </tr>`;
    });
    frameHtml += '</tbody></table></div>';
  }

  document.getElementById('results-detail').innerHTML = `
    <div class="card"><h2>Metrics</h2>${metricHtml}</div>
    <div class="card"><h2>Timing</h2>${timingHtml||'<p style="color:var(--text2)">No timing data available.</p>'}</div>
    <div class="card"><h2>Per-Frame Results</h2>${frameHtml||'<p style="color:var(--text2)">No per-frame data.</p>'}</div>
  `;
}

function filterFrames() {
  const search = (document.getElementById('frame-search')?.value || '').toLowerCase();
  const status = document.getElementById('frame-status-filter')?.value || 'all';
  document.querySelectorAll('#frame-table tbody tr').forEach(row => {
    const name = (row.dataset.name || '').toLowerCase();
    const st = row.dataset.status;
    const show = (status==='all' || st===status) && (!search || name.includes(search));
    row.style.display = show ? '' : 'none';
  });
}

// ======================= AR DEMO =======================
async function loadARDemoExperiments() {
  const data = await api('/api/results/list');
  if (data.error) return;
  const sel = document.getElementById('ar-experiment');
  sel.innerHTML = data.results.map(r => `<option value="${r.path}">${r.name}</option>`).join('');
}

async function loadARDemos() {
  const exp = document.getElementById('ar-gallery-exp').value;
  const data = await api('/api/ardemo/list' + (exp ? '?experiment=' + encodeURIComponent(exp) : ''));
  const gallery = document.getElementById('ar-gallery');

  // Populate filter dropdown
  if (data.experiments) {
    const sel = document.getElementById('ar-gallery-exp');
    sel.innerHTML = '<option value="">All</option>' +
      data.experiments.map(e => `<option value="${e}" ${e===exp?'selected':''}>${e}</option>`).join('');
  }

  if (data.images && data.images.length > 0) {
    gallery.innerHTML = data.images.map(img => `
      <div class="img-card">
        <a href="/api/ardemo/image?path=${encodeURIComponent(img.path)}" target="_blank">
          <img src="/api/ardemo/image?path=${encodeURIComponent(img.path)}" alt="${img.name}" loading="lazy">
        </a>
        <div class="caption">${img.name}</div>
      </div>`).join('');
  } else {
    gallery.innerHTML = '<p style="color:var(--text2)">No AR demo images yet. Generate one first!</p>';
  }
}

async function generateARDemo() {
  const experiment = document.getElementById('ar-experiment').value;
  const limit = document.getElementById('ar-limit').value;
  const cubeSize = document.getElementById('ar-cube-size').value;
  const distance = document.getElementById('ar-distance').value;
  const alpha = document.getElementById('ar-alpha').value;
  const fps = document.getElementById('ar-fps').value;

  document.getElementById('ar-progress').style.display = 'block';
  const log = document.getElementById('ar-log');
  log.textContent = 'Generating AR demo...\n';

  try {
    const res = await fetch('/api/ardemo/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({experiment, limit:parseInt(limit), cube_size:cubeSize||null,
                            distance:parseFloat(distance), alpha:parseFloat(alpha), fps:parseInt(fps)})
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'log') { log.textContent += msg.text + '\n'; log.scrollTop = log.scrollHeight; }
          else if (msg.type === 'done') { toast(msg.text, 'success'); loadARDemos(); }
          else if (msg.type === 'error') { toast(msg.text, 'error'); }
        } catch(e) {}
      }
    }
  } catch(e) { toast('Error: ' + e.message, 'error'); }

  document.getElementById('ar-progress').style.display = 'none';
}

// ======================= EXPORT =======================
async function loadExportExperiments() {
  const data = await api('/api/results/list');
  if (data.error) return;
  const sel = document.getElementById('export-experiment');
  sel.innerHTML = data.results.map(r => `<option value="${r.path}">${r.name}</option>`).join('');
}

async function exportResults() {
  const experiment = document.getElementById('export-experiment').value;
  const dir = document.getElementById('export-dir').value;
  if (!dir) { toast('Please specify an export directory.', 'error'); return; }
  const st = document.getElementById('export-status');
  st.innerHTML = '<span class="spinner"></span> Exporting...';
  const data = await api('/api/export', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({experiment, output_dir:dir})
  });
  if (data.error) { st.innerHTML = `<span style="color:var(--accent)">Error: ${data.error}</span>`; }
  else { st.innerHTML = `<span style="color:var(--accent2)">Exported to ${dir}</span>`; toast('Export complete!', 'success'); }
}

async function exportAll() {
  const dir = document.getElementById('export-dir').value;
  if (!dir) { toast('Please specify an export directory.', 'error'); return; }
  const st = document.getElementById('export-status');
  st.innerHTML = '<span class="spinner"></span> Exporting all experiments...';
  const data = await api('/api/export/all', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({output_dir:dir})
  });
  if (data.error) { st.innerHTML = `<span style="color:var(--accent)">Error: ${data.error}</span>`; }
  else { st.innerHTML = `<span style="color:var(--accent2)">All experiments exported to ${dir}</span>`; toast('All exported!', 'success'); }
}

async function generatePackage() {
  const name = document.getElementById('pkg-name').value;
  const dir = document.getElementById('pkg-dir').value;
  if (!name || !dir) { toast('Please fill in all fields.', 'error'); return; }
  const st = document.getElementById('pkg-status');
  st.innerHTML = '<span class="spinner"></span> Generating package...';
  const data = await api('/api/package', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, output_dir:dir})
  });
  if (data.error) { st.innerHTML = `<span style="color:var(--accent)">Error: ${data.error}</span>`; }
  else { st.innerHTML = `<span style="color:var(--accent2)">Package created: ${data.path}</span>`; toast('Package created!', 'success'); }
}

// ======================= INIT =======================
loadDashboard();
loadConfigs();
loadResultList();
loadARDemoExperiments();
loadARDemos();
loadExportExperiments();
</script>
</body>
</html>'''


# ========================= FLASK APP =========================

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string(HTML)

    # ---- Configs ----
    @app.route('/api/configs')
    def api_configs():
        configs = []
        for f in sorted(CONFIG_DIR.glob("*.yaml")):
            configs.append({"name": f.stem, "path": str(f.relative_to(PROJECT_ROOT))})
        return jsonify({"configs": configs})

    # ---- Results list ----
    @app.route('/api/results/list')
    def api_results_list():
        results = []
        if RESULTS_DIR.exists():
            for d in sorted(RESULTS_DIR.iterdir()):
                if d.is_dir():
                    results.append({"name": d.name, "path": str(d.relative_to(PROJECT_ROOT))})
        return jsonify({"results": results})

    # ---- Results all (for dashboard) ----
    @app.route('/api/results/all')
    def api_results_all():
        experiments = []
        summary_rows = []
        if RESULTS_DIR.exists():
            for d in sorted(RESULTS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                csv_path = d / "results.csv"
                if not csv_path.exists():
                    continue
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        recall_k = sorted([k for k in row.keys() if k.startswith("(")])
                        experiments.append({
                            "name": d.name,
                            "path": str(d.relative_to(PROJECT_ROOT)),
                            "recall_025m_2deg": row.get(recall_k[0], "0") if len(recall_k) > 0 else "0",
                            "n_query": row.get("n_query", "?"),
                            "n_localized": row.get("n_localized", "?"),
                        })

                # Read config for method info
                config_name = _resolve_config(d.name)
                config_path = CONFIG_DIR / config_name
                retrieval, detector, matcher = "", "", ""
                if config_path.exists():
                    try:
                        with open(config_path, encoding="utf-8") as f:
                            cfg = yaml_load(f)
                        retrieval = cfg.get("retrieval", {}).get("method", "").replace("Retrieval", "")
                        detector = cfg.get("detector", {}).get("method", "").replace("Detector", "")
                        matcher = cfg.get("matcher", {}).get("method", "").replace("Matcher", "")
                    except Exception:
                        pass

                # Read results.csv for summary
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        recall_k = sorted([k for k in row.keys() if k.startswith("(")])
                        summary_rows.append({
                            "experiment": d.name,
                            "retrieval": retrieval, "detector": detector, "matcher": matcher,
                            "dataset": row.get("dataset", ""), "scene": row.get("scene", ""),
                            "recall_0.25m_2deg": row.get(recall_k[0], "0") if len(recall_k) > 0 else "0",
                            "recall_0.5m_5deg": row.get(recall_k[1], "0") if len(recall_k) > 1 else "0",
                            "recall_5m_10deg": row.get(recall_k[2], "0") if len(recall_k) > 2 else "0",
                            "n_query": row.get("n_query", "?"),
                            "n_localized": row.get("n_localized", "?"),
                        })
        return jsonify({"experiments": experiments, "summary": summary_rows})

    # ---- Results detail ----
    @app.route('/api/results/detail')
    def api_results_detail():
        rel_path = request.args.get("path", "")
        d = PROJECT_ROOT / rel_path
        results = {}
        frames = []
        timing = {}

        csv_path = d / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, {})
                results = {k: v for k, v in row.items()}

        pf_path = d / "per_frame.csv"
        if pf_path.exists():
            with open(pf_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                frames = list(reader)

        timing_path = d / "timing.json"
        if timing_path.exists():
            with open(timing_path, encoding="utf-8") as f:
                timing = json.load(f)

        return jsonify({"results": results, "frames": frames, "timing": timing})

    # ---- Run experiment (streaming) ----
    @app.route('/api/run', methods=['POST'])
    def api_run():
        data = request.get_json()
        config = data.get("config", "")
        limit = data.get("limit", 5)
        build_sfm = data.get("build_sfm", False)

        def generate():
            cmd = [
                sys.executable, "scripts/run_pipeline.py",
                "--config", config,
                "--limit_queries", str(limit),
            ]
            if build_sfm:
                cmd.append("--build_sfm")

            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            yield json.dumps({"type": "log", "text": f"$ {' '.join(cmd)}\n"}) + "\n"

            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )

            total = max(limit, 1)
            for line in proc.stdout:
                line = line.rstrip()
                yield json.dumps({"type": "log", "text": line}) + "\n"

                if "queries" in line.lower() and "/" in line:
                    try:
                        parts = line.split("/")
                        current = int(''.join(filter(str.isdigit, parts[0].split()[-1])))
                        pct = min(int(current / total * 100), 99)
                        yield json.dumps({"type": "progress", "pct": pct, "text": line.strip()}) + "\n"
                    except Exception:
                        pass

            proc.wait()
            if proc.returncode == 0:
                yield json.dumps({"type": "done", "text": "Experiment completed!"}) + "\n"
            else:
                yield json.dumps({"type": "error", "text": f"Exit code {proc.returncode}"}) + "\n"

        return Response(generate(), mimetype="text/event-stream")

    # ---- AR Demo list ----
    @app.route('/api/ardemo/list')
    def api_ardemo_list():
        experiments = []
        images = []
        if AR_DEMO_DIR.exists():
            for d in sorted(AR_DEMO_DIR.iterdir()):
                if d.is_dir():
                    experiments.append(d.name)
        exp_filter = request.args.get("experiment", "")
        search_dirs = [AR_DEMO_DIR / exp_filter] if exp_filter else \
                      [AR_DEMO_DIR / d for d in experiments]

        for sd in search_dirs:
            if not sd.exists():
                continue
            for img_file in sorted(sd.glob("*_ar.jpg")):
                images.append({
                    "name": img_file.name,
                    "path": str(img_file.relative_to(PROJECT_ROOT)),
                    "experiment": sd.name,
                })

        return jsonify({"experiments": experiments, "images": images})

    # ---- AR Demo image serving ----
    @app.route('/api/ardemo/image')
    def api_ardemo_image():
        path = request.args.get("path", "")
        from flask import send_file
        full = PROJECT_ROOT / path
        if full.exists():
            return send_file(full, mimetype="image/jpeg")
        return "Not found", 404

    # ---- AR Demo generate (streaming) ----
    @app.route('/api/ardemo/generate', methods=['POST'])
    def api_ardemo_generate():
        data = request.get_json()
        experiment = data.get("experiment", "")
        limit = data.get("limit", 30)
        cube_size = data.get("cube_size", None)
        distance = data.get("distance", 1.0)
        alpha = data.get("alpha", 0.55)
        fps = data.get("fps", 5)

        results_path = PROJECT_ROOT / experiment
        poses_path = results_path / "pred_poses.json"
        if not poses_path.exists():
            def gen_err():
                yield json.dumps({"type": "error", "text": "No pred_poses.json found"}) + "\n"
            return Response(gen_err(), mimetype="text/event-stream")

        config_name = _resolve_config(experiment)
        config_path = CONFIG_DIR / config_name
        output_dir = PROJECT_ROOT / "outputs" / "ar_demo" / experiment

        cmd = [
            sys.executable, "scripts/ar_demo.py",
            "--config", str(config_path.relative_to(PROJECT_ROOT)),
            "--poses", str(poses_path.relative_to(PROJECT_ROOT)),
            "--output", str(output_dir.relative_to(PROJECT_ROOT)),
            "--limit", str(limit),
            "--alpha", str(alpha),
            "--fps", str(fps),
            "--cube_distance", str(distance),
        ]
        if cube_size is not None:
            cmd.extend(["--cube_size", str(cube_size)])

        def generate_ar():
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            yield json.dumps({"type": "log", "text": f"$ {' '.join(cmd)}\n"}) + "\n"
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )
            for line in proc.stdout:
                yield json.dumps({"type": "log", "text": line.rstrip()}) + "\n"
            proc.wait()
            if proc.returncode == 0:
                yield json.dumps({"type": "done", "text": f"AR demo saved to {output_dir.relative_to(PROJECT_ROOT)}"}) + "\n"
            else:
                yield json.dumps({"type": "error", "text": f"Failed (exit {proc.returncode})"}) + "\n"

        return Response(generate_ar(), mimetype="text/event-stream")

    # ---- Export single ----
    @app.route('/api/export', methods=['POST'])
    def api_export():
        data = request.get_json()
        experiment = data.get("experiment", "")
        output_dir = data.get("output_dir", "")
        if not experiment or not output_dir:
            return jsonify({"error": "Missing parameters"})

        src = PROJECT_ROOT / experiment
        dst = Path(output_dir) / experiment
        dst.mkdir(parents=True, exist_ok=True)

        import shutil
        for fname in ["results.csv", "per_frame.csv", "timing.json", "pred_poses.json"]:
            s = src / fname
            if s.exists():
                shutil.copy2(s, dst / fname)

        config_name = _resolve_config(experiment)
        config_path = CONFIG_DIR / config_name
        if config_path.exists():
            try:
                cmd = [
                    sys.executable, "scripts/generate_report.py",
                    "--results_dir", str(src.relative_to(PROJECT_ROOT)),
                    "--config", str(config_path.relative_to(PROJECT_ROOT)),
                    "--output", str((dst / f"{experiment}_report.html").relative_to(PROJECT_ROOT)),
                ]
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True)
            except Exception:
                pass

        return jsonify({"status": "ok", "output": str(dst)})

    # ---- Export all ----
    @app.route('/api/export/all', methods=['POST'])
    def api_export_all():
        data = request.get_json()
        output_dir = data.get("output_dir", "")
        if not output_dir:
            return jsonify({"error": "Missing output_dir"})

        import shutil
        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)

        if RESULTS_DIR.exists():
            for d in sorted(RESULTS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                for fname in ["results.csv", "per_frame.csv", "timing.json", "pred_poses.json"]:
                    s = d / fname
                    if s.exists():
                        dest_dir = dst / "results" / d.name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(s, dest_dir / fname)

        if AR_DEMO_DIR.exists():
            for d in sorted(AR_DEMO_DIR.iterdir()):
                if d.is_dir():
                    dest_dir = dst / "ar_demo" / d.name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for f in d.iterdir():
                        if f.is_file():
                            shutil.copy2(f, dest_dir / f.name)

        return jsonify({"status": "ok", "output": str(dst)})

    # ---- Package ----
    @app.route('/api/package', methods=['POST'])
    def api_package():
        data = request.get_json()
        name = data.get("name", "final-visual_localization")
        output_dir = data.get("output_dir", "")
        if not output_dir:
            return jsonify({"error": "Missing output_dir"})

        import shutil
        import zipfile

        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        zip_path = dst / f"{name}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                dirs[:] = [d for d in dirs if d not in (
                    "__pycache__", ".git", "node_modules", ".claude",
                    "outputs", "data", "weights", "third_party", "xrlocalization"
                )]
                for file in files:
                    if file.endswith(('.pyc', '.pth', '.zip', '.tar.gz', '.7z')):
                        continue
                    full_path = Path(root) / file
                    arc_name = full_path.relative_to(PROJECT_ROOT)
                    zf.write(full_path, f"{name}/{arc_name}")

            # Add results and reports
            for subdir in ["outputs/results", "outputs/ar_demo", "outputs/reports"]:
                sp = PROJECT_ROOT / subdir
                if sp.exists():
                    for root, dirs, files in os.walk(sp):
                        for file in files:
                            full_path = Path(root) / file
                            arc_name = full_path.relative_to(PROJECT_ROOT)
                            zf.write(full_path, f"{name}/{arc_name}")

        file_size = zip_path.stat().st_size
        return jsonify({"status": "ok", "path": str(zip_path),
                        "size_mb": round(file_size / (1024*1024), 1)})

    return app


def _resolve_config(exp_name):
    mapping = {
        "baseline_a": "baseline_a.yaml",
        "baseline_b": "baseline_b.yaml",
        "exp_retrieval": "exp_retrieval.yaml",
        "exp_match": "exp_match.yaml",
        "exp_full": "exp_full.yaml",
        "exp_crica": "exp_crica.yaml",
        "7scenes_stairs_baseline": "7scenes_stairs_baseline.yaml",
        "7scenes_stairs_eigenplaces": "7scenes_stairs_eigenplaces.yaml",
    }
    return mapping.get(exp_name, f"{exp_name}.yaml")


def yaml_load(path):
    """YAML loader that avoids the full yaml import if yaml is not available."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Visual Localization Web UI")
    p.add_argument("--port", type=int, default=5000, help="Server port")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    args = p.parse_args()

    if not HAS_FLASK:
        print("ERROR: Flask is required. Install with: pip install flask")
        print("Falling back to CLI UI...")
        import scripts.ui
        scripts.ui.main_menu()
        return

    app = create_app()
    print(f"\n  Visual Localization Web UI")
    print(f"  Open: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
