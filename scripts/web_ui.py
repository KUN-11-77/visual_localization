"""
视觉定位 Web 交互界面 — Flask 单页应用

基于 OpenXRLab XRLocalization 扩展框架，提供完整的实验运行、结果查看、
AR Demo 展示和导出功能。

用法:
  python scripts/web_ui.py [--port 5000]
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
AR_DEMO_DIR = PROJECT_ROOT / "outputs" / "ar_demo"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

try:
    from flask import Flask, render_template_string, request, jsonify, Response, send_file
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>视觉定位交互系统 — Visual Localization</title>
<style>
:root {
  --bg: #0f0f1a; --surface: #1a1a2e; --surface2: #16213e;
  --accent: #e94560; --accent2: #4CAF50; --text: #eee; --text2: #999;
  --border: #2a2a4a; --radius: 10px; --gold: #f0a500;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Microsoft YaHei','PingFang SC','Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
.container { max-width:1400px; margin:0 auto; padding:20px; }

/* Header */
header { text-align:center; padding:24px 0 16px; border-bottom:2px solid var(--accent); margin-bottom:24px; }
header h1 { font-size:26px; font-weight:800; background:linear-gradient(135deg, var(--accent), var(--gold)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
header p { color:var(--text2); margin-top:6px; font-size:13px; }

/* Tabs */
.tabs { display:flex; gap:3px; margin-bottom:20px; flex-wrap:wrap; }
.tab-btn { padding:10px 22px; border:none; background:var(--surface); color:var(--text2);
  cursor:pointer; border-radius:var(--radius) var(--radius) 0 0; font-size:14px; transition:.2s;
  font-family:inherit; }
.tab-btn:hover { background:var(--surface2); color:var(--text); }
.tab-btn.active { background:var(--accent); color:#fff; font-weight:700; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

/* Cards */
.card { background:var(--surface); border-radius:var(--radius); padding:22px; margin-bottom:18px; border:1px solid var(--border); }
.card h2 { font-size:17px; margin-bottom:14px; color:var(--accent2); border-bottom:1px solid var(--border); padding-bottom:8px; }
.card h3 { font-size:14px; margin:14px 0 8px; color:var(--text); }

/* Tables */
table { width:100%; border-collapse:collapse; font-size:13px; margin:8px 0; }
th { background:var(--surface2); color:var(--text); padding:10px 8px; text-align:left; font-weight:600; white-space:nowrap; }
td { padding:8px; border-bottom:1px solid var(--border); }
tr:hover td { background:rgba(233,69,96,.06); }

/* Buttons & Inputs */
.btn { padding:10px 20px; border:none; border-radius:6px; cursor:pointer; font-size:14px;
  font-weight:600; transition:.2s; display:inline-flex; align-items:center; gap:6px; font-family:inherit; }
.btn-primary { background:var(--accent); color:#fff; }
.btn-primary:hover { filter:brightness(1.15); }
.btn-green { background:var(--accent2); color:#fff; }
.btn-green:hover { filter:brightness(1.15); }
.btn-gold { background:var(--gold); color:#1a1a2e; }
.btn-gold:hover { filter:brightness(1.15); }
.btn-outline { background:transparent; border:1px solid var(--border); color:var(--text); }
.btn-outline:hover { background:var(--surface2); }
.btn-sm { padding:5px 12px; font-size:12px; }
.btn:disabled { opacity:.45; cursor:not-allowed; }

select, input[type=text], input[type=number] { padding:9px 12px; border:1px solid var(--border);
  border-radius:6px; background:var(--bg); color:var(--text); font-size:13px; width:100%; font-family:inherit; }
select:focus, input:focus { outline:none; border-color:var(--accent); }

.form-row { display:flex; gap:12px; align-items:end; flex-wrap:wrap; margin-bottom:10px; }
.form-group { flex:1; min-width:160px; }
.form-group label { display:block; margin-bottom:4px; font-size:11px; color:var(--text2); letter-spacing:.5px; }

/* Badges */
.badge { display:inline-block; padding:3px 10px; border-radius:10px; font-size:11px; font-weight:700; }
.badge-ok { background:rgba(76,175,80,.2); color:var(--accent2); }
.badge-fail { background:rgba(233,69,96,.2); color:var(--accent); }
.badge-info { background:rgba(33,150,243,.2); color:#42a5f5; }

/* Metric grid */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:14px; }
.metric-item { background:var(--bg); border-radius:8px; padding:14px; text-align:center; }
.metric-item .val { font-size:30px; font-weight:800; }
.metric-item .lbl { font-size:11px; color:var(--text2); margin-top:3px; }
.val-high { color:var(--accent2); }
.val-mid { color:var(--gold); }
.val-low { color:var(--accent); }

/* Experiment list */
.exp-card { display:flex; align-items:center; padding:12px 18px; background:var(--bg);
  border-radius:8px; margin-bottom:6px; cursor:pointer; transition:.2s; border:1px solid transparent; }
.exp-card:hover { border-color:var(--accent); }
.exp-card .exp-name { flex:1; font-weight:600; }
.exp-card .exp-meta { font-size:11px; color:var(--text2); margin:0 14px; text-align:right; }
.exp-card .exp-recall { font-size:17px; font-weight:700; }

/* Progress */
.progress-wrap { background:var(--bg); border-radius:8px; height:6px; margin:8px 0; overflow:hidden; }
.progress-bar { height:100%; background:var(--accent); width:0%; transition:width .3s; border-radius:8px; }

/* Log */
.log-output { background:#050510; color:#0f0; font-family:'Consolas','Courier New',monospace; font-size:12px;
  padding:14px; border-radius:8px; max-height:300px; overflow-y:auto; white-space:pre-wrap; line-height:1.4; }

/* Media grid (images + videos) */
.media-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
.media-card { background:var(--bg); border-radius:8px; overflow:hidden; border:1px solid var(--border); transition:.2s; }
.media-card:hover { border-color:var(--accent); }
.media-card img, .media-card video { width:100%; display:block; object-fit:cover; }
.media-card .caption { padding:8px 12px; font-size:11px; color:var(--text2); }
.media-card .tag { position:absolute; top:8px; right:8px; padding:3px 8px; border-radius:4px; font-size:10px; font-weight:700; }
.media-card .tag-video { background:var(--accent); color:#fff; }
.media-card .tag-img { background:var(--accent2); color:#fff; }

/* Video section highlight */
.video-section { background:linear-gradient(135deg, rgba(233,69,96,.08), rgba(240,165,0,.08));
  border:1px solid rgba(233,69,96,.25); border-radius:var(--radius); padding:18px; margin-bottom:18px; }
.video-section h3 { color:var(--accent); margin-bottom:12px; }

/* Modal */
.modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.8);
  z-index:1000; justify-content:center; align-items:center; }
.modal-overlay.show { display:flex; }
.modal { background:var(--surface); border-radius:var(--radius); max-width:90vw; max-height:90vh;
  border:1px solid var(--border); position:relative; }
.modal video { max-width:90vw; max-height:85vh; border-radius:var(--radius); }
.modal-close { position:absolute; top:8px; right:14px; background:none; border:none; color:#fff;
  font-size:28px; cursor:pointer; z-index:10; }

/* Toast */
.toast { position:fixed; bottom:24px; right:24px; padding:14px 24px; border-radius:8px;
  color:#fff; font-weight:600; z-index:2000; animation:fadeUp .3s; }
.toast-success { background:var(--accent2); }
.toast-error { background:var(--accent); }
@keyframes fadeUp { from{opacity:0;transform:translateY(20px);} to{opacity:1;transform:translateY(0);} }

/* Filter */
.filter-bar { display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap; align-items:center; }
.filter-bar input { max-width:260px; }
.filter-bar select { max-width:160px; }

/* Stats */
.stats-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
.stat-chip { background:var(--bg); padding:7px 14px; border-radius:16px; font-size:12px; }
.stat-chip strong { color:var(--accent); }

/* Spinner */
.spinner { display:inline-block; width:16px; height:16px; border:2px solid var(--text2);
  border-top-color:var(--accent); border-radius:50%; animation:spin .6s linear infinite; }
@keyframes spin { to{transform:rotate(360deg);} }

/* Responsive */
@media(max-width:768px) {
  .form-row { flex-direction:column; }
  .tabs { flex-direction:column; }
  .tab-btn { border-radius:var(--radius); }
  .metric-grid { grid-template-columns:repeat(2,1fr); }
  .media-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>视觉定位交互系统</h1>
  <p>OpenXRLab XRLocalization 扩展框架 | 检索 → 检测 → 匹配 → PnP 位姿估计 | Cambridge &amp; 7-Scenes</p>
</header>

<!-- 导航标签 -->
<div class="tabs">
  <button class="tab-btn active" data-tab="dashboard">📊 总览面板</button>
  <button class="tab-btn" data-tab="run">🚀 运行实验</button>
  <button class="tab-btn" data-tab="results">📋 结果查看</button>
  <button class="tab-btn" data-tab="ardemo">🧊 AR 演示</button>
</div>

<!-- ==================== 总览面板 ==================== -->
<div id="tab-dashboard" class="tab-panel active">
  <div class="card">
    <h2>实验概览</h2>
    <div id="dashboard-metric-grid" class="metric-grid"></div>
  </div>
  <div class="card">
    <h2>召回率对比 (0.25m, 2° 高精度)</h2>
    <div id="dashboard-recall-table"></div>
  </div>
  <div class="card">
    <h2>实验结果汇总</h2>
    <div style="overflow-x:auto;">
    <table id="dashboard-summary-table">
      <thead><tr>
        <th>实验名称</th><th>检索方法</th><th>检测方法</th><th>匹配方法</th>
        <th>数据集</th><th>(0.25m,2°)</th><th>(0.5m,5°)</th><th>(5m,10°)</th>
        <th>查询数</th><th>成功数</th></tr></thead>
      <tbody></tbody>
    </table>
    </div>
  </div>
</div>

<!-- ==================== 运行实验 ==================== -->
<div id="tab-run" class="tab-panel">
  <div class="card">
    <h2>运行定位实验</h2>
    <p style="color:var(--text2);font-size:12px;margin-bottom:14px;">
      输入数据集路径，选择方法配置，点击运行即可。结果自动保存到指定输出目录。</p>

    <!-- 数据集路径（必填） -->
    <div class="form-row">
      <div class="form-group" style="flex:2;">
        <label>📁 数据集根目录 <span style="color:var(--accent)">*必填</span></label>
        <input type="text" id="run-root" placeholder="例: D:/datasets/Cambridge/KingsCollege 或 data/cambridge/KingsCollege">
      </div>
      <div class="form-group" style="flex:1;">
        <label>📍 场景名称 <span style="color:var(--accent)">*必填</span></label>
        <input type="text" id="run-scene" placeholder="例: KingsCollege">
      </div>
      <div class="form-group" style="flex:1;">
        <label>🗂 数据集类型 <span style="color:var(--accent)">*必填</span></label>
        <select id="run-dsname">
          <option value="cambridge">Cambridge (COLMAP/NVM)</option>
          <option value="7scenes">7-Scenes (RGB-D)</option>
        </select>
      </div>
    </div>

    <!-- 方法配置 -->
    <div class="form-row">
      <div class="form-group">
        <label>🔧 方法配置</label>
        <select id="run-config"></select>
      </div>
      <div class="form-group">
        <label>🔍 查询数量 (0=全部)</label>
        <input type="number" id="run-limit" value="5" min="0" max="10000">
      </div>
    </div>

    <!-- 输出路径 -->
    <div class="form-row">
      <div class="form-group">
        <label>📤 输出目录（留空自动生成）</label>
        <input type="text" id="run-output" placeholder="例: outputs/results/my_experiment">
      </div>
    </div>

    <div style="display:flex;gap:10px;margin-top:6px;">
      <button class="btn btn-primary" id="btn-run" onclick="runExperiment()">▶ 开始运行</button>
      <button class="btn btn-outline" id="btn-run-stop" onclick="stopRun()" style="display:none">⏹ 停止</button>
    </div>
    <div id="run-progress" style="display:none;margin-top:14px;">
      <div class="progress-wrap"><div class="progress-bar" id="run-bar"></div></div>
      <p id="run-status" style="font-size:12px;color:var(--text2);margin-top:3px;"></p>
      <div class="log-output" id="run-log"></div>
    </div>
  </div>
</div>

<!-- ==================== 结果查看 ==================== -->
<div id="tab-results" class="tab-panel">
  <div class="card">
    <h2>选择实验</h2>
    <select id="results-select" onchange="loadResultDetail()" style="max-width:400px;"></select>
  </div>
  <div id="results-detail"></div>
</div>

<!-- ==================== AR 演示 ==================== -->
<div id="tab-ardemo" class="tab-panel">
  <div class="card">
    <h2>生成 AR 演示</h2>
    <p style="color:var(--text2);font-size:12px;margin-bottom:12px;">
      在重建场景中放置固定世界坐标的虚拟立方体，使用预测位姿渲染到查询图像上。
      定位越精确，立方体越稳定；抖动越明显，定位越不准确。</p>
    <div class="form-row">
      <div class="form-group">
        <label>实验名称</label>
        <select id="ar-experiment"></select>
      </div>
      <div class="form-group">
        <label>帧数</label>
        <input type="number" id="ar-limit" value="30" min="1" max="500">
      </div>
      <div class="form-group">
        <label>立方体边长 (米，留空自动)</label>
        <input type="text" id="ar-cube-size" placeholder="自动检测">
      </div>
    </div>
    <div style="display:flex;gap:10px;margin-top:6px;">
      <button class="btn btn-green" onclick="generateARDemo()">🎬 生成 AR 演示</button>
    </div>
    <div id="ar-progress" style="display:none;margin-top:14px;">
      <div class="log-output" id="ar-log"></div>
    </div>
  </div>

  <!-- AR 图片画廊 — 按场景筛选 -->
  <div class="card">
    <h2>AR 演示画廊</h2>
    <div class="filter-bar" style="display:flex;gap:10px;margin-bottom:14px;">
      <select id="ar-scene-filter" onchange="loadARDemos()" style="max-width:200px;">
        <option value="">全部场景</option>
        <option value="cambridge">🏛 Cambridge</option>
        <option value="7scenes">🏠 7-Scenes</option>
      </select>
      <select id="ar-gallery-exp" onchange="loadARDemos()" style="max-width:240px;"></select>
    </div>
    <div class="media-grid" id="ar-gallery"></div>
  </div>
</div>
</div><!-- .container -->

<!-- 视频模态框 -->
<div class="modal-overlay" id="video-modal" onclick="closeVideoModal(event)">
  <div class="modal">
    <button class="modal-close" onclick="document.getElementById('video-modal').classList.remove('show')">&times;</button>
    <video id="modal-video" controls autoplay loop style="max-width:90vw;max-height:85vh;border-radius:var(--radius);"></video>
  </div>
</div>

<script>
// ======================= 全局状态 =======================
let runAbort = false;

// ======================= 标签切换 =======================
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    switch(btn.dataset.tab) {
      case 'dashboard': loadDashboard(); break;
      case 'results': loadResultList(); break;
      case 'ardemo': loadARDemoExperiments(); loadARDemos(); break;
      case 'run': loadConfigs(); break;
    }
  });
});

// ======================= API 调用 =======================
async function api(url, opts={}) {
  try {
    const res = await fetch(url, opts);
    return await res.json();
  } catch(e) {
    console.error('API error:', e);
    return {error: e.message};
  }
}

// ======================= 提示消息 =======================
function toast(msg, type='success') {
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

// ======================= 总览面板 =======================
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
    <div class="metric-item"><div class="val">${nExp}</div><div class="lbl">实验组数</div></div>
    <div class="metric-item"><div class="val val-high">${(bestRecall*100).toFixed(1)}%</div><div class="lbl">最佳召回率 (0.25m, 2°)</div></div>
    <div class="metric-item"><div class="val">${bestName}</div><div class="lbl">最佳实验</div></div>
    <div class="metric-item"><div class="val">${totalQueries}</div><div class="lbl">总查询帧数</div></div>
  `;

  if (data.summary) {
    let rows = '';
    data.summary.forEach(r => {
      const r025 = parseFloat(r['recall_0.25m_2deg']||0);
      const cls = r025 > 0.6 ? 'val-high' : r025 > 0.3 ? 'val-mid' : 'val-low';
      const color = r025 > 0.6 ? '#4CAF50' : r025 > 0.3 ? '#f0a500' : '#f44336';
      rows += `<tr>
        <td><strong>${r.experiment||''}</strong></td>
        <td>${r.retrieval||''}</td><td>${r.detector||''}</td><td>${r.matcher||''}</td>
        <td>${r.dataset||''} / ${r.scene||''}</td>
        <td style="color:${color};font-weight:700">${(r025*100).toFixed(1)}%</td>
        <td>${(parseFloat(r['recall_0.5m_5deg']||0)*100).toFixed(1)}%</td>
        <td>${(parseFloat(r['recall_5m_10deg']||0)*100).toFixed(1)}%</td>
        <td>${r.n_query||''}</td>
        <td>${r.n_localized||''}</td>
      </tr>`;
    });
    document.querySelector('#dashboard-summary-table tbody').innerHTML = rows;
  }

  if (data.experiments) {
    let bars = data.experiments.map(e => {
      const r = parseFloat(e.recall_025m_2deg || 0) * 100;
      const color = r>60?'#4CAF50':r>30?'#f0a500':'#f44336';
      return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
        <span style="width:200px;font-size:12px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.name}</span>
        <div style="flex:1;background:var(--bg);border-radius:4px;height:22px;overflow:hidden">
          <div style="width:${Math.max(r,3)}%;height:100%;background:${color};border-radius:4px;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;font-size:11px;font-weight:700">${r.toFixed(1)}%</div>
        </div>
      </div>`;
    }).join('');
    document.getElementById('dashboard-recall-table').innerHTML = bars;
  }
}

// ======================= 运行实验 =======================
async function loadConfigs() {
  const data = await api('/api/configs');
  if (data.error) return;
  document.getElementById('run-config').innerHTML =
    data.configs.map(c => `<option value="${c.path}">${c.name}</option>`).join('');
}

async function runExperiment() {
  const config = document.getElementById('run-config').value;
  const limit = document.getElementById('run-limit').value;
  const rootPath = document.getElementById('run-root').value.trim();
  const sceneName = document.getElementById('run-scene').value.trim();
  const outputDir = document.getElementById('run-output').value.trim();
  const dsName = document.getElementById('run-dsname').value;

  if (!rootPath || !sceneName) { toast('请填写数据集路径和场景名称', 'error'); return; }

  runAbort = false;
  document.getElementById('btn-run').disabled = true;
  document.getElementById('btn-run-stop').style.display = 'inline-flex';
  document.getElementById('run-progress').style.display = 'block';
  document.getElementById('run-log').textContent = '';
  document.getElementById('run-bar').style.width = '0%';
  document.getElementById('run-status').textContent = '正在启动...';

  try {
    const res = await fetch('/api/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        config, limit:parseInt(limit),
        root_override: rootPath,
        scene_override: sceneName,
        dsname_override: dsName,
        output_dir: outputDir || null,
      })
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
            document.getElementById('run-status').textContent = '✅ 实验完成！';
            document.getElementById('run-bar').style.width = '100%';
            toast('实验运行成功！', 'success');
          } else if (msg.type === 'error') {
            document.getElementById('run-status').textContent = '❌ 错误：' + msg.text;
            toast('实验失败：' + msg.text, 'error');
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    toast('连接错误：' + e.message, 'error');
  }

  document.getElementById('btn-run').disabled = false;
  document.getElementById('btn-run-stop').style.display = 'none';
  loadDashboard();
}

function stopRun() { runAbort = true; }

// ======================= 结果查看 =======================
async function loadResultList() {
  const data = await api('/api/results/list');
  if (data.error) return;
  document.getElementById('results-select').innerHTML =
    '<option value="">-- 选择实验 --</option>' +
    data.results.map(r => `<option value="${r.path}">${r.name}</option>`).join('');
}

async function loadResultDetail() {
  const path = document.getElementById('results-select').value;
  if (!path) { document.getElementById('results-detail').innerHTML = ''; return; }

  const data = await api('/api/results/detail?path=' + encodeURIComponent(path));
  if (data.error) {
    document.getElementById('results-detail').innerHTML = '<p style="color:var(--accent)">加载失败。</p>';
    return;
  }

  let metricHtml = '';
  if (data.results) {
    const r = data.results;
    const recallKeys = Object.keys(r).filter(k => k.startsWith('(')).sort();
    metricHtml += '<div class="metric-grid">';
    recallKeys.forEach(k => {
      const v = parseFloat(r[k]) * 100;
      metricHtml += `<div class="metric-item"><div class="val ${v>60?'val-high':v>30?'val-mid':'val-low'}">${v.toFixed(1)}%</div><div class="lbl">召回率 ${k}</div></div>`;
    });
    metricHtml += `<div class="metric-item"><div class="val">${r.n_query||'?'}</div><div class="lbl">总查询数</div></div>`;
    metricHtml += `<div class="metric-item"><div class="val val-high">${r.n_localized||'?'}</div><div class="lbl">成功定位 (0.25m,2°)</div></div>`;
    metricHtml += '</div>';
  }

  let timingHtml = '';
  if (data.timing && Object.keys(data.timing).length > 0) {
    timingHtml = '<h3>各阶段耗时 (单帧平均)</h3><table><tr><th>阶段</th><th>平均 (ms)</th><th>标准差 (ms)</th></tr>';
    for (const [key,val] of Object.entries(data.timing)) {
      if (typeof val === 'object' && val.mean_ms !== undefined) {
        timingHtml += `<tr><td>${key}</td><td>${val.mean_ms.toFixed(1)}</td><td>${val.std_ms.toFixed(1)}</td></tr>`;
      }
    }
    timingHtml += '</table>';
  }

  let frameHtml = '';
  if (data.frames && data.frames.length > 0) {
    const nLoc = data.frames.filter(f => parseFloat(f.t_err) <= 0.25 && parseFloat(f.r_err) <= 2.0).length;
    frameHtml += `<div class="stats-row">
      <div class="stat-chip">📷 <strong>${data.frames.length}</strong> 帧</div>
      <div class="stat-chip">✅ <strong>${nLoc}</strong> 成功定位</div>
      <div class="stat-chip">📊 <strong>${(nLoc/data.frames.length*100).toFixed(1)}%</strong> 成功率</div>
    </div>`;
    frameHtml += `<div class="filter-bar">
      <input type="text" id="frame-search" placeholder="搜索图片名称..." oninput="filterFrames()">
      <select id="frame-status-filter" onchange="filterFrames()">
        <option value="all">全部</option><option value="True">成功定位</option><option value="False">定位失败</option>
      </select>
    </div>`;
    frameHtml += `<div style="max-height:450px;overflow-y:auto;"><table id="frame-table">
      <thead><tr><th>#</th><th>查询图像</th><th>关键点</th><th>匹配对</th><th>内点</th><th>平移误差 (m)</th><th>旋转误差 (°)</th><th>检索 Top-1</th><th>状态</th></tr></thead>
      <tbody>`;
    data.frames.forEach((f,i) => {
      const t = parseFloat(f.t_err) || 999;
      const r = parseFloat(f.r_err) || 999;
      const loc = t <= 0.25 && r <= 2.0;
      const statusStr = loc ? 'True' : 'False';
      frameHtml += `<tr data-status="${statusStr}" data-name="${f.query||''}"${loc?'':' style="opacity:0.55"'}>
        <td>${i+1}</td><td style="font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis" title="${f.query||''}">${(f.query||'').split('/').pop()}</td>
        <td>${f.n_query_kpts||'?'}</td><td>${f.n_correspondences||'?'}</td><td>${f.n_inliers||'?'}</td>
        <td>${f.t_err||'?'}</td><td>${f.r_err||'?'}</td>
        <td style="font-size:10px;max-width:140px;overflow:hidden;text-overflow:ellipsis" title="${f.retrieved_top1||''}">${(f.retrieved_top1||'').split('/').pop()||'?'}</td>
        <td><span class="badge ${loc?'badge-ok':'badge-fail'}">${loc?'成功':'失败'}</span></td>
      </tr>`;
    });
    frameHtml += '</tbody></table></div>';
  }

  document.getElementById('results-detail').innerHTML = `
    <div class="card"><h2>评估指标</h2>${metricHtml}</div>
    <div class="card"><h2>耗时分析</h2>${timingHtml||'<p style="color:var(--text2)">暂无耗时数据。</p>'}</div>
    <div class="card"><h2>逐帧结果</h2>${frameHtml||'<p style="color:var(--text2)">暂无逐帧数据。</p>'}</div>
  `;
}

function filterFrames() {
  const search = (document.getElementById('frame-search')?.value || '').toLowerCase();
  const status = document.getElementById('frame-status-filter')?.value || 'all';
  document.querySelectorAll('#frame-table tbody tr').forEach(row => {
    const name = (row.dataset.name || '').toLowerCase();
    const st = row.dataset.status;
    row.style.display = (status==='all' || st===status) && (!search || name.includes(search)) ? '' : 'none';
  });
}

// ======================= AR 演示 =======================
async function loadARDemoExperiments() {
  const data = await api('/api/results/list');
  if (data.error) return;
  document.getElementById('ar-experiment').innerHTML =
    data.results.map(r => `<option value="${r.path}">${r.name}</option>`).join('');
}

async function loadARDemos() {
  const exp = document.getElementById('ar-gallery-exp').value;
  const scene = document.getElementById('ar-scene-filter').value;
  const data = await api('/api/ardemo/list' + (exp ? '?experiment=' + encodeURIComponent(exp) : ''));
  const gallery = document.getElementById('ar-gallery');

  // 填充实验下拉框
  if (data.experiments) {
    const sel = document.getElementById('ar-gallery-exp');
    sel.innerHTML = '<option value="">全部实验</option>' +
      data.experiments.map(e => `<option value="${e}" ${e===exp?'selected':''}>${e}</option>`).join('');
  }

  // 渲染图片画廊 — 按场景过滤，每实验只取首帧
  if (data.images && data.images.length > 0) {
    const isCambridge = (e) => ['baseline_a','baseline_b','exp_','shopfacade'].some(p => e.startsWith(p));
    const is7Scenes = (e) => e.startsWith('7s');

    let filtered = data.images;
    if (scene === 'cambridge') filtered = filtered.filter(img => isCambridge(img.experiment));
    if (scene === '7scenes')   filtered = filtered.filter(img => is7Scenes(img.experiment));

    // 每实验只取第一帧
    const seen = {};
    const firstFrames = [];
    for (const img of filtered) {
      if (!seen[img.experiment]) {
        seen[img.experiment] = true;
        firstFrames.push(img);
      }
    }

    if (firstFrames.length > 0) {
      gallery.innerHTML = firstFrames.map(img => `
        <div class="media-card">
          <div style="position:relative;">
            <a href="/api/ardemo/image?path=${encodeURIComponent(img.path)}" target="_blank">
              <img src="/api/ardemo/image?path=${encodeURIComponent(img.path)}" alt="${img.experiment}"
                   loading="lazy" style="aspect-ratio:16/9;object-fit:cover;width:100%;">
            </a>
            <span class="tag tag-img" style="position:absolute;top:8px;right:8px;background:var(--accent2);color:#fff;padding:3px 8px;border-radius:4px;font-size:10px;">AR</span>
          </div>
          <div class="caption" style="padding:10px 12px;">
            <strong>${img.experiment}</strong><br>
            <span style="color:var(--text2);font-size:11px;">🖱 点击查看大图</span>
          </div>
        </div>`).join('');
    } else {
      gallery.innerHTML = '<p style="color:var(--text2);text-align:center;padding:20px;grid-column:1/-1">该场景暂无 AR 演示图片。</p>';
    }
  } else {
    gallery.innerHTML = '<p style="color:var(--text2);text-align:center;padding:20px;">暂无 AR 演示图片，请先生成！</p>';
  }

  // 隐藏旧视频区域
  const vs = document.getElementById('ar-video-section');
  if (vs) vs.innerHTML = '';
}

async function generateARDemo() {
  const experiment = document.getElementById('ar-experiment').value;
  const limit = document.getElementById('ar-limit').value;
  const cubeSize = document.getElementById('ar-cube-size').value;
  const alpha = document.getElementById('ar-alpha').value;
  const fps = document.getElementById('ar-fps').value;

  document.getElementById('ar-progress').style.display = 'block';
  const log = document.getElementById('ar-log');
  log.textContent = '正在生成 AR 演示...\n';

  try {
    const res = await fetch('/api/ardemo/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({experiment, limit:parseInt(limit), cube_size:cubeSize||null,
                            alpha:parseFloat(alpha), fps:parseInt(fps)})
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
  } catch(e) { toast('生成失败：' + e.message, 'error'); }

  document.getElementById('ar-progress').style.display = 'none';
}

// ======================= 初始化 =======================
loadDashboard();
loadConfigs();
loadResultList();
loadARDemoExperiments();
loadARDemos();
</script>
</body>
</html>'''


# ========================= FLASK 应用 =========================

def create_app():
    app = Flask(__name__)

    @app.after_request
    def no_cache(resp):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return resp

    @app.route('/favicon.ico')
    def favicon():
        return "", 204

    @app.route('/')
    def index():
        return render_template_string(HTML)

    # ---- 配置文件列表 ----
    @app.route('/api/configs')
    def api_configs():
        configs = []
        for f in sorted(CONFIG_DIR.glob("*.yaml")):
            configs.append({"name": f.stem, "path": str(f.relative_to(PROJECT_ROOT))})
        return jsonify({"configs": configs})

    # ---- 实验结果列表 ----
    @app.route('/api/results/list')
    def api_results_list():
        results = []
        if RESULTS_DIR.exists():
            for d in sorted(RESULTS_DIR.iterdir()):
                if d.is_dir():
                    results.append({"name": d.name, "path": str(d.relative_to(PROJECT_ROOT))})
        return jsonify({"results": results})

    # ---- 所有结果（总览面板用） ----
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

                # 读取 config 获取方法信息
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

    # ---- 实验结果详情 ----
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
                for row in reader:
                    # Normalize dot-containing column names for JS bracket access
                    row["localized"] = row.get("localized_0.25m_2deg", "False")
                    frames.append(row)

        timing_path = d / "timing.json"
        if timing_path.exists():
            with open(timing_path, encoding="utf-8") as f:
                timing = json.load(f)

        return jsonify({"results": results, "frames": frames, "timing": timing})

    # ---- 运行实验（SSE 流式输出） ----
    @app.route('/api/run', methods=['POST'])
    def api_run():
        data = request.get_json()
        config = data.get("config", "")
        limit = data.get("limit", 5)
        root_override = data.get("root_override", "")
        scene_override = data.get("scene_override", "")
        dsname_override = data.get("dsname_override", "")
        output_dir = data.get("output_dir", "")

        def generate():
            cmd = [
                sys.executable, "scripts/run_pipeline.py",
                "--config", config,
                "--limit_queries", str(limit),
            ]
            # Dataset path / scene overrides
            overrides = []
            if root_override:
                overrides.append(f"dataset.root={root_override}")
            if scene_override:
                overrides.append(f"dataset.scene={scene_override}")
            if dsname_override:
                overrides.append(f"dataset.name={dsname_override}")
            if overrides:
                cmd.extend(["--overrides"] + overrides)
            # Custom output dir
            if output_dir:
                cmd.extend(["--output_dir", output_dir])

            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            yield json.dumps({"type": "log", "text": f"$ {' '.join(cmd)}"}) + "\n"

            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )

            total = max(limit, 1)
            for line in proc.stdout:
                line = line.rstrip()
                yield json.dumps({"type": "log", "text": line}) + "\n"

                # 解析进度
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
                yield json.dumps({"type": "done", "text": "✅ 实验完成！"}) + "\n"
            else:
                yield json.dumps({"type": "error", "text": f"❌ 退出码 {proc.returncode}"}) + "\n"

        return Response(generate(), mimetype="text/event-stream")

    # ---- AR Demo 列表（含视频） ----
    @app.route('/api/ardemo/list')
    def api_ardemo_list():
        experiments = []
        images = []
        videos = []
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
            # 收集视频文件
            for vid_file in sorted(sd.glob("*.mp4")):
                videos.append({
                    "name": vid_file.name,
                    "path": str(vid_file.relative_to(PROJECT_ROOT)),
                    "experiment": sd.name,
                })
            # 收集图片文件
            for img_file in sorted(sd.glob("*_ar.jpg")):
                images.append({
                    "name": img_file.name,
                    "path": str(img_file.relative_to(PROJECT_ROOT)),
                    "experiment": sd.name,
                })

        return jsonify({"experiments": experiments, "images": images, "videos": videos})

    # ---- AR Demo 图片 ----
    @app.route('/api/ardemo/image')
    def api_ardemo_image():
        path = request.args.get("path", "")
        full = PROJECT_ROOT / path
        if full.exists():
            return send_file(full, mimetype="image/jpeg")
        return "文件未找到", 404

    # ---- AR Demo 视频 ----
    @app.route('/api/ardemo/video')
    def api_ardemo_video():
        path = request.args.get("path", "")
        full = (PROJECT_ROOT / path.replace("\\", "/")).resolve()
        if not full.exists():
            return "file not found", 404

        file_size = full.stat().st_size
        range_header = request.headers.get("Range")

        if range_header:
            import re
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                with open(full, "rb") as f:
                    f.seek(start)
                    data = f.read(length)
                resp = Response(data, 206, mimetype="video/mp4")
                resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Content-Length"] = str(length)
                return resp

        resp = send_file(str(full), mimetype="video/mp4")
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    # ---- AR Demo 生成（SSE 流式输出） ----
    @app.route('/api/ardemo/generate', methods=['POST'])
    def api_ardemo_generate():
        data = request.get_json()
        experiment = data.get("experiment", "")
        limit = data.get("limit", 30)
        cube_size = data.get("cube_size", None)
        alpha = data.get("alpha", 0.55)
        fps = data.get("fps", 5)

        results_path = PROJECT_ROOT / experiment
        poses_path = results_path / "pred_poses.json"
        if not poses_path.exists():
            def gen_err():
                yield json.dumps({"type": "error", "text": "❌ 未找到 pred_poses.json"}) + "\n"
            return Response(gen_err(), mimetype="text/event-stream")

        config_name = _resolve_config(experiment)
        config_path = CONFIG_DIR / config_name
        output_dir = AR_DEMO_DIR / experiment

        cmd = [
            sys.executable, "scripts/ar_demo.py",
            "--config", str(config_path.relative_to(PROJECT_ROOT)),
            "--poses", str(poses_path.relative_to(PROJECT_ROOT)),
            "--output", str(output_dir.relative_to(PROJECT_ROOT)),
            "--limit", str(limit),
            "--alpha", str(alpha),
            "--fps", str(fps),
        ]
        if cube_size is not None:
            cmd.extend(["--cube_size", str(cube_size)])

        def generate_ar():
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            yield json.dumps({"type": "log", "text": f"$ {' '.join(cmd)}"}) + "\n"
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )
            for line in proc.stdout:
                yield json.dumps({"type": "log", "text": line.rstrip()}) + "\n"
            proc.wait()
            if proc.returncode == 0:
                yield json.dumps({"type": "done", "text": f"✅ AR 演示已保存到 {output_dir.relative_to(PROJECT_ROOT)}"}) + "\n"
            else:
                yield json.dumps({"type": "error", "text": f"❌ 生成失败 (退出码 {proc.returncode})"}) + "\n"

        return Response(generate_ar(), mimetype="text/event-stream")

    return app


def _resolve_config(exp_name):
    """根据实验名称解析对应的 YAML 配置文件"""
    mapping = {
        "baseline_a": "baseline_a.yaml",
        "baseline_b": "baseline_b.yaml",
        "exp_retrieval": "exp_retrieval.yaml",
        "exp_match": "exp_match.yaml",
        "exp_full": "exp_full.yaml",
        "exp_crica": "exp_crica.yaml",
        "7scenes_stairs_baseline": "7scenes_stairs_baseline.yaml",
        "7scenes_stairs_eigenplaces": "7scenes_stairs_eigenplaces.yaml",
        # ShopFacade experiments
        "shopfacade_baseline_b": "shopfacade_baseline_b.yaml",
        "shopfacade_exp_retrieval": "shopfacade_exp_retrieval.yaml",
        "shopfacade_exp_match": "shopfacade_exp_match.yaml",
        "shopfacade_exp_full": "shopfacade_exp_full.yaml",
        "shopfacade_exp_crica": "shopfacade_exp_crica.yaml",
        "shopfacade_sp_sg": "shopfacade_sp_sg.yaml",
    }
    if exp_name in mapping:
        return mapping[exp_name]
    # Auto-resolve: try exp_name.yaml, and also try shopfacade_ prefix stripping
    candidate = f"{exp_name}.yaml"
    if (CONFIG_DIR / candidate).exists():
        return candidate
    return f"{exp_name}.yaml"


def yaml_load(path):
    """加载 YAML 配置文件"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main():
    import argparse
    p = argparse.ArgumentParser(description="视觉定位 Web 交互界面")
    p.add_argument("--port", type=int, default=5000, help="服务器端口")
    p.add_argument("--host", default="127.0.0.1", help="服务器地址")
    args = p.parse_args()

    if not HAS_FLASK:
        print("ERROR: Need Flask: pip install flask")
        print("Falling back to CLI...")
        import scripts.ui
        scripts.ui.main_menu()
        return

    app = create_app()
    print(f"\n  Visual Localization Web UI")
    print(f"  Open browser: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
