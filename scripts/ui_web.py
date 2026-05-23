"""
Flask-based Web UI for Visual Localization Pipeline.

Provides:
  /           — Experiment runner: select dataset, methods, run, view results
  /showcase   — Results showcase: interactive comparison charts and analysis

Usage:
  python scripts/ui_web.py [--port 5000]
  Then open http://localhost:5000 in a browser.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
AR_DIR = PROJECT_ROOT / "outputs" / "ar_demo"

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))

# ── In-memory job tracker ──────────────────────────────────────────────
_jobs: dict[str, dict] = {}  # job_id -> {status, progress, result, error, config}


# ── Dataset presets ─────────────────────────────────────────────────────
DATASET_PRESETS = {
    "cambridge": {
        "KingsCollege": {
            "root": "data/cambridge/KingsCollege",
            "query_list": "dataset_test.txt",
            "db_list": "dataset_train.txt",
            "fx": 1670, "fy": 1670, "cx": 960, "cy": 540,
        },
        "OldHospital": {
            "root": "data/cambridge/OldHospital",
            "query_list": "dataset_test.txt",
            "db_list": "dataset_train.txt",
            "fx": 1670, "fy": 1670, "cx": 960, "cy": 540,
        },
        "ShopFacade": {
            "root": "data/cambridge/ShopFacade",
            "query_list": "dataset_test.txt",
            "db_list": "dataset_train.txt",
            "fx": 1670, "fy": 1670, "cx": 960, "cy": 540,
        },
        "StMarysChurch": {
            "root": "data/cambridge/StMarysChurch",
            "query_list": "dataset_test.txt",
            "db_list": "dataset_train.txt",
            "fx": 1670, "fy": 1670, "cx": 960, "cy": 540,
        },
        "GreatCourt": {
            "root": "data/cambridge/GreatCourt",
            "query_list": "dataset_test.txt",
            "db_list": "dataset_train.txt",
            "fx": 1670, "fy": 1670, "cx": 960, "cy": 540,
        },
    },
    "7scenes": {
        "chess": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "fire": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "heads": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "office": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "pumpkin": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "redkitchen": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
        "stairs": {
            "root": "data/7scenes",
            "fx": 585, "fy": 585, "cx": 320, "cy": 240,
        },
    },
}

METHOD_OPTIONS = {
    "retrieval": {
        "NetVLADRetrieval": {"pca_dim": 4096, "device": "cuda"},
        "EigenPlacesRetrieval": {"device": "cuda"},
        "CricaVPRRetrieval": {"device": "cuda"},
    },
    "detector": {
        "SIFTDetector": {"max_keypoints": 8192},
        "SuperPointDetector": {"max_keypoints": 4096, "device": "cuda"},
        "ALIKEDDetector": {"max_keypoints": 4096, "device": "cuda"},
    },
    "matcher": {
        "NNMatcher": {"ratio_thresh": 0.8, "mutual": True},
        "SuperGlueMatcher": {"weights": "outdoor", "confidence_threshold": 0.2, "device": "cuda"},
        "LightGlueMatcher": {"device": "cuda"},
    },
}


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main experiment runner page."""
    configs = _list_configs()
    return render_template("index.html", configs=configs,
                           datasets=DATASET_PRESETS,
                           methods=METHOD_OPTIONS)


@app.route("/showcase")
def showcase():
    """Results showcase page."""
    summary = _load_summary()
    results = _load_all_results()
    return render_template("showcase.html",
                           summary=summary,
                           results=results)


# ── API Endpoints ───────────────────────────────────────────────────────

@app.route("/api/datasets")
def api_datasets():
    """Return available dataset presets."""
    return jsonify(DATASET_PRESETS)


@app.route("/api/methods")
def api_methods():
    """Return available methods."""
    return jsonify(METHOD_OPTIONS)


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a localization experiment."""
    data = request.get_json()

    job_id = str(uuid.uuid4())[:8]
    results_dir_name = data.get("name", f"exp_{job_id}")
    results_dir = RESULTS_DIR / results_dir_name

    # Build config dict
    dataset_name = data["dataset"]
    scene = data["scene"]
    preset = DATASET_PRESETS.get(dataset_name, {}).get(scene, {})

    config = {
        "name": data.get("name", f"exp_{job_id}"),
        "description": data.get("description", ""),
        "retrieval": {
            "method": data["retrieval"],
            "top_k": int(data.get("top_k", 10)),
            "params": data.get("retrieval_params", {}),
        },
        "detector": {
            "method": data["detector"],
            "params": data.get("detector_params", {}),
        },
        "matcher": {
            "method": data["matcher"],
            "params": data.get("matcher_params", {}),
        },
        "dataset": {
            "name": dataset_name,
            "scene": scene,
            "root": data.get("root", preset.get("root", "")),
            "query_list": preset.get("query_list", "dataset_test.txt"),
            "db_list": preset.get("db_list", "dataset_train.txt"),
            "fx": preset.get("fx", 1670),
            "fy": preset.get("fy", 1670),
            "cx": preset.get("cx", 960),
            "cy": preset.get("cy", 540),
        },
        "pose_solver": {
            "method": "pnp_ransac",
            "ransac_thresh": float(data.get("ransac_thresh", 12.0)),
            "min_inliers": int(data.get("min_inliers", 12)),
        },
        "output": {
            "results_dir": str(results_dir),
        },
    }

    # Override root if provided
    if data.get("root"):
        config["dataset"]["root"] = data["root"]

    # Add NVM model for Cambridge
    if dataset_name == "cambridge":
        config["dataset"]["nvm_model"] = data.get("nvm_model", "reconstruction.nvm")

    job = {
        "id": job_id,
        "status": "starting",
        "progress": 0,
        "message": "Building configuration...",
        "result": None,
        "error": None,
        "config": config,
        "results_dir": str(results_dir),
        "process": None,
    }
    _jobs[job_id] = job

    thread = threading.Thread(target=_run_experiment, args=(job_id, config), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    """Poll job status. Returns progress, message, and final result."""
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
    }
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job["error"]

    return jsonify(response)


@app.route("/api/results")
def api_results():
    """Return all experiment results for showcase."""
    return jsonify(_load_all_results())


@app.route("/api/summary")
def api_summary():
    """Return summary of all experiments."""
    return jsonify(_load_summary())


@app.route("/api/configs")
def api_configs():
    """List saved experiment configs."""
    return jsonify(_list_configs())


@app.route("/api/results/image")
def api_result_image():
    """Serve an image file from the outputs directory."""
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "No path provided"}), 400
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(full_path), mimetype="image/jpeg")


@app.route("/api/config/<name>")
def api_config(name):
    """Load a specific config YAML."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        path = CONFIG_DIR / name
    if not path.exists():
        return jsonify({"error": "Config not found"}), 404
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return jsonify(cfg)


# ── Background execution ────────────────────────────────────────────────

def _run_experiment(job_id, config):
    """Run pipeline in subprocess, update job status."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["message"] = "Starting pipeline..."

    try:
        # Write temp config
        tmp_config = Path(tempfile.gettempdir()) / f"ui_experiment_{job_id}.yaml"
        with open(tmp_config, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--config", str(tmp_config),
            "--limit_queries", "0",
        ]

        job["message"] = f"Running: {' '.join(cmd)}"
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        job["process"] = proc

        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)

            # Parse progress from tqdm-like output
            if "DB encode" in line or "Localizing" in line:
                # Extract percentage from tqdm bar: "  DB encode:  45%|████     | 45/100"
                m = re.search(r'(\d+)%', line)
                if m:
                    job["progress"] = int(m.group(1))
                    job["message"] = line.strip()
            elif "Encoding database" in line:
                job["progress"] = 0
                job["message"] = line.strip()
            elif "Localizing" in line and "queries" in line:
                job["progress"] = 0
                job["message"] = line.strip()
            elif "Results" in line or "Recall" in line:
                job["message"] = line.strip()
            elif "saved to" in line.lower():
                job["message"] = line.strip()

        proc.wait(timeout=1800)  # 30 min timeout

        # Clean up temp config
        try:
            tmp_config.unlink()
        except OSError:
            pass

        if proc.returncode != 0:
            job["status"] = "error"
            job["error"] = "\n".join(output_lines[-30:])  # last 30 lines
            return

        # Read results
        results_dir = Path(job["results_dir"])
        result_data = {}
        csv_path = results_dir / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader)
                result_data["metrics"] = {
                    k: (float(v) if v else 0.0)
                    for k, v in row.items()
                }

        timing_path = results_dir / "timing.json"
        if timing_path.exists():
            with open(timing_path, encoding="utf-8") as f:
                result_data["timing"] = json.load(f)

        per_frame_path = results_dir / "per_frame.csv"
        if per_frame_path.exists():
            with open(per_frame_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                frames = list(reader)
            n_loc = sum(1 for frm in frames if frm.get("localized_0.25m_2deg", "") == "True")
            result_data["frames"] = {
                "total": len(frames),
                "localized": n_loc,
                "samples": frames[:20],
            }

        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Experiment completed!"
        job["result"] = result_data

    except subprocess.TimeoutExpired:
        job["status"] = "error"
        job["error"] = "Experiment timed out (30 minutes)"
        if job.get("process"):
            try:
                job["process"].kill()
            except Exception:
                pass
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ── Data helpers ────────────────────────────────────────────────────────

def _list_configs():
    """List available experiment config files."""
    configs = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            configs.append({
                "file": p.name,
                "name": cfg.get("name", p.stem),
                "description": cfg.get("description", ""),
                "retrieval": cfg.get("retrieval", {}).get("method", ""),
                "detector": cfg.get("detector", {}).get("method", ""),
                "matcher": cfg.get("matcher", {}).get("method", ""),
                "dataset": cfg.get("dataset", {}).get("name", ""),
                "scene": cfg.get("dataset", {}).get("scene", ""),
            })
        except Exception:
            pass
    return configs


def _load_summary():
    """Load summary CSV as list of dicts."""
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        return []
    rows = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k.strip()] = float(v)
                except (ValueError, TypeError):
                    parsed[k.strip()] = v
            rows.append(parsed)
    return rows


def _load_all_results():
    """Load all experiment result details, using best run from summary + dir data."""
    all_results = []
    if not RESULTS_DIR.exists():
        return all_results

    # First, gather metrics from individual result directories (primary source)
    dir_metrics = {}  # dir_name -> best metrics for that experiment
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        dir_csv = d / "results.csv"
        if not dir_csv.exists():
            continue
        with open(dir_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_name = row.get("experiment", d.name).strip()
                metrics = {}
                for k, v in row.items():
                    if v is None or v == "":
                        metrics[k.strip()] = 0.0
                    else:
                        try:
                            metrics[k.strip()] = float(v)
                        except (ValueError, TypeError):
                            metrics[k.strip()] = v
                # Keep best run per experiment
                cur_best = dir_metrics.get(exp_name, {}).get("(0.25m, 2deg)", -1)
                new_val = metrics.get("(0.25m, 2deg)", -1)
                if isinstance(new_val, (int, float)) and new_val > (cur_best if isinstance(cur_best, (int, float)) else -1):
                    dir_metrics[exp_name] = metrics

    # Supplement with summary.csv (may have newer format or additional runs)
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_name = row.get("experiment", "").strip()
                if not exp_name:
                    continue
                # Normalize column names: summary may use recall_0.25m_2deg or (0.25m, 2deg)
                metrics = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    if v is None or v == "":
                        metrics[k_clean] = 0.0
                    else:
                        try:
                            metrics[k_clean] = float(v)
                        except (ValueError, TypeError):
                            metrics[k_clean] = v
                # Map summary column names to standard format
                if "recall_0.25m_2deg" in metrics:
                    metrics["(0.25m, 2deg)"] = metrics.pop("recall_0.25m_2deg")
                if "recall_0.5m_5deg" in metrics:
                    metrics["(0.5m, 5deg)"] = metrics.pop("recall_0.5m_5deg")
                if "recall_5m_10deg" in metrics:
                    metrics["(5.0m, 10deg)"] = metrics.pop("recall_5m_10deg")

                # Merge: update if this run is better
                if exp_name in dir_metrics:
                    cur_best = dir_metrics[exp_name].get("(0.25m, 2deg)", -1)
                    new_val = metrics.get("(0.25m, 2deg)", -1)
                    if isinstance(new_val, (int, float)) and isinstance(cur_best, (int, float)) and new_val > cur_best:
                        dir_metrics[exp_name] = metrics
                else:
                    dir_metrics[exp_name] = metrics

    # Now collect directory data (timing, frames, AR)
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        result = {"name": d.name, "path": str(d), "metrics": {}}

        # Determine experiment name from directory's results.csv
        exp_name = d.name
        dir_csv = d / "results.csv"
        if dir_csv.exists():
            with open(dir_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                try:
                    row = next(reader)
                    exp_name = row.get("experiment", d.name).strip()
                except StopIteration:
                    pass
            result["name"] = exp_name

        # Match to best metrics: try exact match, then directory name match, then prefix match
        if exp_name in dir_metrics:
            result["metrics"] = dir_metrics[exp_name]
        elif d.name in dir_metrics:
            result["metrics"] = dir_metrics[d.name]
        else:
            for key, val in dir_metrics.items():
                if key.startswith(d.name) or d.name.startswith(key):
                    result["metrics"] = val
                    break

        # Override with summary data if available (summary often has the best run)
        summary_name = d.name.replace("_netvlad_sift_nn", "").replace("_netvlad_superpoint_superglue", "") \
                            .replace("_eigenplaces_superpoint_superglue", "").replace("_netvlad_aliked_lightglue", "") \
                            .replace("_eigenplaces_aliked_lightglue", "").replace("_cricavpr_aliked_lightglue", "")
        if summary_name in dir_metrics:
            summary_metrics = dir_metrics[summary_name]
            summary_recall = summary_metrics.get("(0.25m, 2deg)", -1)
            cur_recall = result["metrics"].get("(0.25m, 2deg)", -1)
            if isinstance(summary_recall, (int, float)) and isinstance(cur_recall, (int, float)):
                if summary_recall > cur_recall:
                    result["metrics"] = summary_metrics

        timing_path = d / "timing.json"
        if timing_path.exists():
            with open(timing_path, encoding="utf-8") as f:
                result["timing"] = json.load(f)

        per_frame_path = d / "per_frame.csv"
        if per_frame_path.exists():
            with open(per_frame_path, newline="", encoding="utf-8") as f:
                frames = list(csv.DictReader(f))
            n_loc = sum(1 for frm in frames if frm.get("localized_0.25m_2deg", "") == "True")
            result["frames_total"] = len(frames)
            result["frames_localized"] = n_loc

        # AR demo images
        ar_path = AR_DIR / d.name
        if ar_path.exists():
            ar_images = sorted(ar_path.glob("*.jpg"))[:8]
            result["ar_images"] = [str(p.relative_to(PROJECT_ROOT)) for p in ar_images]

        all_results.append(result)

    return all_results


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visual Localization Web UI")
    p.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    p.add_argument("--debug", action="store_true", help="Debug mode")
    args = p.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       Visual Localization Pipeline — Web UI                  ║
║                                                              ║
║  Experiment Runner:  http://{args.host}:{args.port}/           ║
║  Results Showcase:   http://{args.host}:{args.port}/showcase   ║
║                                                              ║
║  Press Ctrl+C to stop the server.                            ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host=args.host, port=args.port, debug=args.debug)
