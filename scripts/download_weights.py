#!/usr/bin/env python3
"""
Download all model weights required by the visual localization pipeline.

Usage:
  python scripts/download_weights.py              # download all
  python scripts/download_weights.py --list       # list required files
  python scripts/download_weights.py --skip large # skip files > 100MB
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (destination_relative_to_project, download_url, description, expected_size_mb)
WEIGHTS = [
    # --- Retrieval: NetVLAD ---
    (
        "vendor/netvlad/models/Pitts30K_struct.mat",
        "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",
        "NetVLAD Pittsburgh 30K weights",
        529,
    ),
    # --- Retrieval: EigenPlaces ---
    (
        "vendor/eigenplaces/weights/resnet50_2048_eigenplaces.pth",
        "https://github.com/gmberton/EigenPlaces/releases/download/v1.0/resnet50_2048_eigenplaces.pth",
        "EigenPlaces ResNet50 2048-dim weights",
        106,
    ),
    # --- Retrieval: CricaVPR ---
    (
        "weights/CricaVPR.pth",
        "https://github.com/Lu-Feng/CricaVPR/releases/download/v1.0/CricaVPR.pth",
        "CricaVPR cross-image correlation weights",
        562,
    ),
    (
        "weights/dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "DINOv2 ViT-B/14 backbone (required by CricaVPR)",
        331,
    ),
    # --- Detector: ALIKED ---
    (
        "weights/aliked-n16.pth",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
        "ALIKED keypoint detector N16 weights",
        3,
    ),
    # --- Matcher: LightGlue ---
    (
        "weights/aliked_lightglue.pth",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth",
        "LightGlue matcher weights (ALIKED features)",
        46,
    ),
    (
        "weights/superpoint_lightglue.pth",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
        "LightGlue matcher weights (SuperPoint features)",
        46,
    ),
    # --- Detector + Matcher: SuperPoint / SuperGlue ---
    (
        "vendor/superglue/weights/superpoint_v1.pth",
        "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth",
        "SuperPoint detector weights",
        5,
    ),
    (
        "vendor/superglue/weights/superglue_indoor.pth",
        "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth",
        "SuperGlue matcher weights (indoor)",
        46,
    ),
    (
        "vendor/superglue/weights/superglue_outdoor.pth",
        "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth",
        "SuperGlue matcher weights (outdoor)",
        46,
    ),
]


def format_size(mb: float) -> str:
    return f"{mb:.0f} MB" if mb < 1000 else f"{mb / 1024:.1f} GB"


def download_file(url: str, dest: Path, desc: str) -> bool:
    """Download a file with progress display. Returns True on success."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {desc} — already exists ({size_mb:.0f} MB)")
        return True

    print(f"  [DOWNLOAD] {desc}")
    print(f"    URL: {url}")
    print(f"    To:  {dest}")

    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    OK ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for visual localization"
    )
    parser.add_argument("--list", action="store_true",
                        help="List all required weight files and exit")
    parser.add_argument("--skip", choices=["large", "cricavpr", "all"],
                        default=None,
                        help="Skip certain files (large: >500MB)")
    args = parser.parse_args()

    if args.list:
        total = 0
        print("\nRequired model weights:\n")
        print(f"{'Size':>8}  {'File':<55}  Description")
        print("-" * 100)
        for rel_path, url, desc, size_mb in WEIGHTS:
            total += size_mb
            print(f"{format_size(size_mb):>8}  {rel_path:<55}  {desc}")
        print("-" * 100)
        print(f"{format_size(total):>8}  TOTAL\n")
        return

    print("\n" + "=" * 60)
    print("  Visual Localization — Model Weight Downloader")
    print("=" * 60 + "\n")

    success = 0
    failed = 0
    skipped = 0
    total_mb = 0

    for rel_path, url, desc, size_mb in WEIGHTS:
        if args.skip == "all":
            print(f"  [SKIP] {desc}")
            skipped += 1
            continue
        if args.skip == "large" and size_mb > 500:
            print(f"  [SKIP] {desc} (>{format_size(500)})")
            skipped += 1
            continue
        if args.skip == "cricavpr" and "CricaVPR" in desc:
            print(f"  [SKIP] {desc} (CricaVPR)")
            skipped += 1
            continue

        dest = PROJECT_ROOT / rel_path
        ok = download_file(url, dest, desc)
        if ok:
            success += 1
            total_mb += size_mb
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {success} ok, {skipped} skipped, {failed} failed")
    print(f"  Downloaded: {format_size(total_mb)}")
    print(f"{'=' * 60}\n")

    if failed > 0:
        print("Some downloads failed. Re-run the script to retry.")
        print("For manual download, place the files at the paths listed above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
