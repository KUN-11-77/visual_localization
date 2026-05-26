#!/usr/bin/env python3
"""
Export a clean submission zip to the desktop.
Excludes .git, data, outputs, weights, __pycache__, and large binary files.
"""

import zipfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESKTOP = Path.home() / "Desktop"

EXCLUDE_DIRS = {
    ".git", "__pycache__", "data", "weights",
    "models", ".claude", "node_modules", ".mypy_cache", ".pytest_cache",
    "third_party", "xrlocalization",
}

EXCLUDE_EXT = {".pth", ".mat", ".pyc", ".zip", ".tar.gz", ".7z",
               ".aux", ".log", ".out", ".toc", ".xdv", ".bbl", ".blg",
               ".synctex.gz", ".fdb_latexmk", ".fls"}

EXCLUDE_FILES = set()


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    suffixes = path.suffixes
    if suffixes and suffixes[-1] in EXCLUDE_EXT:
        return True
    if path.suffix in EXCLUDE_EXT:
        return True
    name = path.name
    if any(name.endswith(ext) for ext in EXCLUDE_EXT):
        return True
    if name in EXCLUDE_FILES:
        return True
    return False


def main():
    zip_name = "visual_localization_submission.zip"
    zip_path = DESKTOP / zip_name

    # Count files first
    files_to_add = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Prune excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in filenames:
            fp = Path(root) / f
            if should_exclude(fp):
                continue
            files_to_add.append(fp)

    print(f"Adding {len(files_to_add)} files to {zip_path}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in files_to_add:
            arcname = fp.relative_to(PROJECT_ROOT)
            zf.write(fp, arcname)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Done! {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
