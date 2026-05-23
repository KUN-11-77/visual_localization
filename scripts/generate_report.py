"""
Generate an interactive HTML report from experiment results.

Reads per_frame.csv, results.csv, and optionally pred_poses.json +
AR demo images to produce a self-contained report page.

Usage:
  python scripts/generate_report.py --results_dir outputs/results/exp_match/ \
      --config configs/exp_match.yaml --output outputs/reports/exp_match.html
"""
import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="Experiment results directory")
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.add_argument("--output", required=True, help="Output HTML file path")
    p.add_argument("--ar_dir", default=None, help="AR demo output directory (optional)")
    p.add_argument("--limit_frames", type=int, default=50,
                   help="Max frames to show in gallery")
    return p.parse_args()


def img_to_b64(path, max_width=320):
    """Encode image as base64 data URI."""
    import cv2
    import numpy as np
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


def load_per_frame(results_dir):
    """Load per_frame.csv, return list of dicts."""
    path = Path(results_dir) / "per_frame.csv"
    frames = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row)
    return frames


def load_results(results_dir):
    """Load results.csv, return dict."""
    path = Path(results_dir) / "results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return row


def load_timing(results_dir):
    """Load timing.json."""
    path = Path(results_dir) / "timing.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_pct(value):
    """Format float as percentage string."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def generate_html(results_dir, config_path, ar_dir, limit_frames):
    results_dir = Path(results_dir)
    frames = load_per_frame(results_dir)
    results = load_results(results_dir)
    timing = load_timing(results_dir)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Limit frames for gallery
    frames_shown = frames[:limit_frames] if limit_frames > 0 else frames

    # Build per-frame gallery rows
    gallery_rows = ""
    for i, frame in enumerate(frames_shown):
        t_err = frame.get("t_err", "")
        r_err = frame.get("r_err", "")
        localized = frame.get("localized_0.25m_2deg", "False") == "True"
        query = frame.get("query", "N/A")
        n_kpts = frame.get("n_query_kpts", "?")
        n_corr = frame.get("n_correspondences", "?")
        n_inl = frame.get("n_inliers", "?")
        top1 = frame.get("retrieved_top1", "")

        status_color = "#4CAF50" if localized else "#f44336"
        status_text = "Localized" if localized else "Failed"
        t_str = f"{float(t_err):.3f}m" if t_err else "N/A"
        r_str = f"{float(r_err):.2f}°" if r_err else "N/A"

        gallery_rows += f"""
        <tr style="border-bottom:1px solid #ddd">
          <td style="padding:4px">{i+1}</td>
          <td style="padding:4px;font-size:12px;max-width:200px;overflow:hidden;text-overflow:ellipsis" title="{query}">{Path(query).name}</td>
          <td style="padding:4px;text-align:center">{n_kpts}</td>
          <td style="padding:4px;text-align:center">{n_corr}</td>
          <td style="padding:4px;text-align:center">{n_inl}</td>
          <td style="padding:4px">{t_str}</td>
          <td style="padding:4px">{r_str}</td>
          <td style="padding:4px;color:{status_color};font-weight:bold">{status_text}</td>
        </tr>"""

    # Build metrics table
    recall_items = sorted([k for k in results.keys() if k.startswith("(")])
    metrics_rows = ""
    for k in recall_items:
        metrics_rows += f"<tr><td>{k}</td><td>{format_pct(results[k])}</td></tr>\n"

    # Timing info
    timing_html = ""
    if timing:
        for key, val in timing.items():
            if isinstance(val, dict):
                timing_html += f"<tr><td>{key}</td><td>{val.get('mean_ms', 0):.1f}ms</td><td>{val.get('std_ms', 0):.1f}ms</td></tr>"

    # Config info
    config_rows = ""
    config_rows += f"<tr><td>Experiment</td><td>{cfg.get('name', 'N/A')}</td></tr>"
    config_rows += f"<tr><td>Dataset</td><td>{cfg['dataset'].get('name', 'N/A')} / {cfg['dataset'].get('scene', 'N/A')}</td></tr>"
    config_rows += f"<tr><td>Retrieval</td><td>{cfg['retrieval'].get('method', 'N/A')}</td></tr>"
    config_rows += f"<tr><td>Detector</td><td>{cfg['detector'].get('method', 'N/A')}</td></tr>"
    config_rows += f"<tr><td>Matcher</td><td>{cfg['matcher'].get('method', 'N/A')}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Localization Report — {cfg.get('name', 'Experiment')}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
  h2 {{ color: #555; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,.1); margin: 10px 0; }}
  th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #4CAF50; color: white; }}
  .metric-good {{ color: #4CAF50; font-weight: bold; }}
  .metric-ok {{ color: #FF9800; font-weight: bold; }}
  .metric-bad {{ color: #f44336; font-weight: bold; }}
  .summary-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); margin: 15px 0; }}
  .gallery-table {{ font-size: 13px; }}
  .gallery-table th {{ font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Visual Localization Experiment Report</h1>

  <div class="summary-card">
    <h2>Configuration</h2>
    <table>
      {config_rows}
    </table>
  </div>

  <div class="summary-card">
    <h2>Recall Metrics</h2>
    <table>
      <tr><th>Threshold</th><th>Recall</th></tr>
      {metrics_rows}
    </table>
    <p>Total queries: {results.get('n_query', 'N/A')} | Localized (0.25m, 2°): {results.get('n_localized', 'N/A')}</p>
  </div>

  <div class="summary-card">
    <h2>Timing (per query)</h2>
    <table>
      <tr><th>Stage</th><th>Mean</th><th>Std</th></tr>
      {timing_html}
    </table>
  </div>

  <div class="summary-card">
    <h2>Per-Frame Results (first {limit_frames})</h2>
    <table class="gallery-table">
      <tr><th>#</th><th>Query Image</th><th>Kpts</th><th>Corr</th><th>Inliers</th><th>t_err</th><th>r_err</th><th>Status</th></tr>
      {gallery_rows}
    </table>
  </div>

  <p style="text-align:center;color:#999;margin-top:40px;font-size:12px">
    Generated from {results_dir}
  </p>
</div>
</body>
</html>"""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_html(args.results_dir, args.config, args.ar_dir, args.limit_frames)
