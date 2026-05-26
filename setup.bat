@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   Visual Localization — 一键环境配置 (Windows)
echo ============================================================
echo.

cd /d "%~dp0\.."

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.9+ first.
    pause
    exit /b 1
)
python --version
echo.

:: 2. Install pip dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some pip packages failed. Continuing...
)
echo.

:: 3. Install PyTorch (separate due to platform-specific wheels)
echo [2/3] PyTorch already in requirements.txt. If it failed:
echo        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo        (Replace 'cpu' with 'cu121' for CUDA 12.1)
echo.

:: 4. Download model weights
echo [3/3] Downloading model weights...
python scripts/download_weights.py
echo.

echo ============================================================
echo   Setup complete!
echo.
echo   Quick start:
echo     python scripts/web_ui.py --port 5000
echo     (Open http://127.0.0.1:5000)
echo ============================================================
echo.
pause
