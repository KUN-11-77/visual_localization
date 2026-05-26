#!/usr/bin/env bash
set -e

echo ""
echo "============================================================"
echo "  Visual Localization — 一键环境配置 (Linux / macOS)"
echo "============================================================"
echo ""

cd "$(dirname "$0")/.."

# 1. Check Python
echo "[1/3] Checking Python..."
python3 --version || { echo "ERROR: Python 3.9+ required"; exit 1; }

# 2. Install pip dependencies
echo "[2/3] Installing Python dependencies..."
pip install -r requirements.txt

# 3. Download model weights
echo "[3/3] Downloading model weights..."
python3 scripts/download_weights.py

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Quick start:"
echo "    python3 scripts/web_ui.py --port 5000"
echo "    (Open http://127.0.0.1:5000)"
echo "============================================================"
echo ""
