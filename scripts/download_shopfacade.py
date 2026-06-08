#!/usr/bin/env python3
"""
Download the Cambridge Landmarks — ShopFacade dataset (~2.9 GB).

After running this script you will have:
    data/cambridge/ShopFacade/
        dataset_test.txt              query images with GT poses
        dataset_train.txt             database images with poses
        seq*/                         RGB images (multiple sequences)
        reconstruction.nvm            VisualSFM reconstruction (NVM format)
        colmap_model/                 COLMAP reconstruction (from PoseNet release)

Usage:
    python scripts/download_shopfacade.py
    python scripts/download_shopfacade.py --skip-images  # only txt + reconstruction
"""

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "cambridge"
EXTRACT_ROOT = DATA_ROOT / "ShopFacade"

# Cambridge University Repository — PoseNet dataset
SHOPFACADE_ZIP_URL = (
    "https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip"
)

# COLMAP reconstructions published by Torsten Sattler (hosted on Google Drive)
# Source: https://github.com/cvg/Hierarchical-Localization
COLMAP_GDRIVE_ID = "1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15"
COLMAP_GDRIVE_URL = (
    "https://drive.google.com/uc?export=download&id=" + COLMAP_GDRIVE_ID
)

SHOPFACADE_ZIP_NAME = "ShopFacade.zip"
COLMAP_ZIP_NAME = "CambridgeLandmarks_Colmap_Retriangulated_1024px.zip"

# ShopFacade zip ~2.9 GB, COLMAP zip ~1.5 GB (all scenes)
EXPECTED_SHOPFACADE_BYTES = 2_900_000_000
EXPECTED_COLMAP_BYTES = 1_500_000_000


def _print(msg):
    print(f"[download_shopfacade] {msg}", flush=True)


def _download(url: str, dest: Path, expected_size: int = 0):
    """Stream-download with resume support and auto-retry on connection errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If file already fully downloaded, skip
    if dest.exists() and expected_size > 0 and dest.stat().st_size >= expected_size * 0.99:
        _print(f"  already exists: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    total_expected = expected_size
    max_retries = 10

    for attempt in range(1, max_retries + 1):
        try:
            # Determine resume offset from existing partial file
            resume_offset = 0
            if dest.exists():
                resume_offset = dest.stat().st_size
                if resume_offset > 1_000_000:
                    _print(f"  resuming from {resume_offset/1e6:.1f} MB "
                           f"(attempt {attempt}/{max_retries})")
                else:
                    resume_offset = 0
            else:
                _print(f"  downloading: {url}")
                _print(f"  to: {dest}")

            headers = {"User-Agent": "Mozilla/5.0"}
            mode = "ab" if resume_offset > 0 else "wb"
            if resume_offset > 0:
                headers["Range"] = f"bytes={resume_offset}-"

            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=60) as resp:
                # Handle 206 Partial Content vs 200 OK
                if resp.status == 206:
                    # Server accepted range request
                    content_range = resp.headers.get("Content-Range", "")
                    if content_range:
                        # "bytes X-Y/Z"
                        try:
                            total_expected = int(content_range.split("/")[-1])
                        except (ValueError, IndexError):
                            pass
                    total = total_expected
                    _print(f"  server supports resume, total={total/1e6:.0f} MB")
                elif resp.status == 200:
                    # Server ignored range request — restart from 0
                    if resume_offset > 0:
                        _print(f"  server ignored range request, restarting download")
                        mode = "wb"
                        resume_offset = 0
                    content_len = resp.headers.get("Content-Length")
                    total = int(content_len) if content_len else total_expected
                else:
                    total = total_expected

                if total == 0:
                    total = total_expected

                chunk = 8 * 1024 * 1024  # 8 MB
                downloaded = resume_offset

                with open(dest, mode) as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total > 0:
                            pct = 100 * downloaded / total
                            mb = downloaded / 1e6
                            total_mb = total / 1e6
                            sys.stdout.write(
                                f"\r    {mb:7.1f} / {total_mb:7.1f} MB  ({pct:5.1f}%)"
                            )
                            sys.stdout.flush()
                print()

                # Verify completeness
                if total > 0 and dest.stat().st_size < total * 0.99:
                    _print(f"  WARNING: file incomplete "
                           f"({dest.stat().st_size/1e6:.1f} / {total/1e6:.1f} MB)")
                    continue  # retry (resume)

                _print(f"  download complete: {dest.stat().st_size/1e6:.1f} MB")
                return dest

        except (ConnectionResetError, ConnectionAbortedError,
                TimeoutError, urllib.error.URLError) as e:
            _print(f"  connection error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait = min(2 ** attempt, 60)  # exponential backoff: 2, 4, 8, 16, 32, 60...
                _print(f"  retrying in {wait}s...")
                time.sleep(wait)
            else:
                _print(f"  FAILED after {max_retries} attempts")
                raise


def _download_gdrive(file_id: str, dest: Path, expected_size: int = 0):
    """Download a large file from Google Drive, handling the confirmation page."""
    import html

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size > 0 and dest.stat().st_size >= expected_size * 0.99:
        _print(f"  already exists: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    max_retries = 5
    session_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    for attempt in range(1, max_retries + 1):
        try:
            _print(f"  Google Drive download (attempt {attempt}/{max_retries})")
            _print(f"  to: {dest}")

            # First request: get the confirmation page if needed
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(session_url, headers=headers)

            with urllib.request.urlopen(req, timeout=60) as resp:
                page = resp.read().decode("utf-8", errors="ignore")

            # Check for virus scan confirmation form (large files)
            confirm_token = None
            import re
            match = re.search(r'confirm=([0-9A-Za-z_\-]+)', page)
            if match:
                confirm_token = match.group(1)
                _print(f"  large file detected, using confirmation token")
                download_url = (
                    f"https://drive.google.com/uc?export=download"
                    f"&id={file_id}&confirm={confirm_token}"
                )
            else:
                # Small file or no confirmation needed
                download_url = session_url

            # Download the actual file
            _download(download_url, dest, expected_size)
            return dest

        except Exception as e:
            _print(f"  Google Drive error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)
                _print(f"  retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _extract(zip_path: Path, target_dir: Path):
    if target_dir.exists() and any(target_dir.iterdir()):
        _print(f"  already extracted: {target_dir}")
        return
    _print(f"  extracting: {zip_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def _extract_colmap_scene(zip_path: Path, scene: str, target_dir: Path):
    """Extract only a single scene subfolder from the COLMAP zip.

    Zip internal layout (may vary):
        CambridgeLandmarks_Colmap_Retriangulated_1024px/<Scene>/
            empty_all/
            model_train/
            list_db_linux.txt
            list_query_linux.txt
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    # Avoid re-extracting if model_train already has files
    model_dir = target_dir / "model_train"
    if model_dir.exists() and any(model_dir.iterdir()):
        _print(f"  COLMAP model already extracted for {scene}")
        return

    _print(f"  extracting: {zip_path} -> {target_dir}  (scene={scene})")
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene}/"
        for name in zf.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                # Strip prefix to land directly in target_dir
                rel = name[len(prefix):]
                dest = target_dir / rel.replace("/", "\\")  # zip uses / even on win
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    _print(f"  COLMAP model ready: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Cambridge Landmarks — ShopFacade (~2.9 GB)"
    )
    parser.add_argument("--skip-images", action="store_true",
                        help="Only download reconstruction files (skip ~2.9 GB zip)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip the COLMAP reconstruction download")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Main ShopFacade zip (~2.9 GB) — images + NVM + GT txt
    # ------------------------------------------------------------------
    if not args.skip_images:
        zip_path = DATA_ROOT / SHOPFACADE_ZIP_NAME
        _print("Step 1/2: downloading ShopFacade (~2.9 GB)")
        _print("           this may take 5-15 minutes depending on bandwidth")
        _download(SHOPFACADE_ZIP_URL, zip_path,
                  expected_size=EXPECTED_SHOPFACADE_BYTES)

        _print("Extracting ShopFacade...")
        # The zip internally contains a "ShopFacade/" root folder,
        # so extract to DATA_ROOT (cambridge/) to get cambridge/ShopFacade/ directly
        _extract(zip_path, DATA_ROOT)

        # Cleanup zip
        keep_zip = (DATA_ROOT / "KEEP_ZIP").exists()
        if not keep_zip:
            _print(f"Removing zip to save disk space "
                   f"(touch {DATA_ROOT / 'KEEP_ZIP'} to keep it)")
            zip_path.unlink(missing_ok=True)
    else:
        _print("Step 1/2: skipped (--skip-images)")

    # ------------------------------------------------------------------
    # 2) COLMAP reconstruction (~1.5 GB total, extract only ShopFacade)
    # ------------------------------------------------------------------
    if not args.skip_colmap:
        colmap_zip = DATA_ROOT / COLMAP_ZIP_NAME
        _print("Step 2/2: downloading COLMAP reconstruction (~1.5 GB, all scenes)")
        _print("           (Google Drive) extracting only ShopFacade subfolder")
        _print("           If this fails, NVM model will be used as fallback.")

        colmap_ok = False
        try:
            _download_gdrive(COLMAP_GDRIVE_ID, colmap_zip,
                             expected_size=EXPECTED_COLMAP_BYTES)
            colmap_ok = True
        except Exception as e:
            _print(f"  Google Drive download failed: {e}")
            _print(f"  Trying direct URL fallback...")
            try:
                _download(COLMAP_GDRIVE_URL, colmap_zip,
                          expected_size=EXPECTED_COLMAP_BYTES)
                colmap_ok = True
            except Exception as e2:
                _print(f"  Direct URL also failed: {e2}")

        if colmap_ok and colmap_zip.exists():
            colmap_target = EXTRACT_ROOT / "colmap_model" / \
                            "CambridgeLandmarks_Colmap_Retriangulated_1024px" / "ShopFacade"
            _extract_colmap_scene(colmap_zip, "ShopFacade", colmap_target)

            # Cleanup colmap zip
            keep_zip = (DATA_ROOT / "KEEP_ZIP").exists()
            if not keep_zip:
                _print("Removing COLMAP zip to save disk space")
                colmap_zip.unlink(missing_ok=True)
        else:
            _print("  COLMAP model unavailable — NVM model will be used instead")
    else:
        _print("Step 2/2: skipped (--skip-colmap)")

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    _print("\n=== Sanity check ===")
    checks = [
        ("dataset_test.txt", EXTRACT_ROOT / "dataset_test.txt"),
        ("dataset_train.txt", EXTRACT_ROOT / "dataset_train.txt"),
        ("reconstruction.nvm", EXTRACT_ROOT / "reconstruction.nvm"),
        ("seq1/", EXTRACT_ROOT / "seq1"),
        ("colmap_model/.../ShopFacade/model_train/",
         EXTRACT_ROOT / "colmap_model" /
         "CambridgeLandmarks_Colmap_Retriangulated_1024px" / "ShopFacade" / "model_train"),
    ]
    for label, p in checks:
        if p.is_dir():
            n = sum(1 for _ in p.iterdir())
            msg = f"exists=True  entries={n}"
        else:
            msg = f"exists={p.exists()}"
        _print(f"  {label:50s} {msg}")

    _print("\nDone. To run:")
    _print("  python scripts/run_pipeline.py --config configs/cambridge_shopfacade_baseline.yaml")
    _print("  python scripts/run_pipeline.py --preset cambridge ShopFacade --limit_queries 5")


if __name__ == "__main__":
    main()
