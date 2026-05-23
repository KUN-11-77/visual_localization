"""
Interactive Command-Line Interface for Visual Localization Pipeline.

Provides a menu-driven interface to:
  1. Select and configure experiments
  2. Run the localization pipeline
  3. View results (metrics, per-frame details)
  4. Generate AR demo visualizations
  5. Export reports to user-specified locations

Usage:
  python scripts/ui.py
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


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    clear_screen()
    print("=" * 60)
    print("    Visual Localization Pipeline — Interactive UI")
    print("    OpenXRLab XRLocalization Extended")
    print("=" * 60)


def list_configs():
    """List available experiment configurations."""
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    return [(i + 1, c.stem, c) for i, c in enumerate(configs)]


def list_results():
    """List experiment result directories."""
    if not RESULTS_DIR.exists():
        return []
    dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    return [(i + 1, d.name, d) for i, d in enumerate(dirs)]


def menu_select_config():
    """Select experiment configuration."""
    print_header()
    print("\nAvailable Configurations:\n")
    configs = list_configs()
    for idx, name, path in configs:
        print(f"  [{idx}] {name}")
    print(f"  [0] Back")

    choice = input("\nSelect configuration > ").strip()
    try:
        idx = int(choice)
        if idx == 0:
            return None
        if 1 <= idx <= len(configs):
            return configs[idx - 1][2]
    except ValueError:
        pass
    return None


def menu_run_experiment():
    """Run a localization experiment."""
    config_path = menu_select_config()
    if config_path is None:
        return

    print_header()
    print(f"\nConfiguration: {config_path.stem}")

    limit = input("\nQuery limit (0=all, default=5 for quick test) > ").strip()
    limit = int(limit) if limit.isdigit() else 5

    build_sfm = input("Build SP+SG SfM model? (y/N) > ").strip().lower()
    build_sfm_flag = "--build_sfm" if build_sfm in ("y", "yes") else ""

    output_dir = input(f"Output directory (default: outputs/results/{config_path.stem}) > ").strip()
    if not output_dir:
        output_dir = f"outputs/results/{config_path.stem}"

    # Override output in config via env or args isn't directly supported
    # Use the default output from config

    print("\nRunning pipeline...")
    cmd = (f'PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE '
           f'python scripts/run_pipeline.py '
           f'--config {config_path} '
           f'--limit_queries {limit} '
           f'{build_sfm_flag}')
    print(f"Command: {cmd}")
    input("\nPress Enter to execute (Ctrl+C to cancel)...")

    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        print("\nExperiment completed successfully!")
    else:
        print(f"\nExperiment failed with exit code {result.returncode}")
    input("\nPress Enter to continue...")


def menu_view_results():
    """View experiment results."""
    print_header()
    print("\nExperiment Results:\n")
    results = list_results()
    for idx, name, path in results:
        print(f"  [{idx}] {name}")
    print(f"  [0] Back")

    choice = input("\nSelect results to view > ").strip()
    try:
        idx = int(choice)
        if idx == 0:
            return
        if 1 <= idx <= len(results):
            _, name, path = results[idx - 1]
            _display_experiment_results(path)
    except ValueError:
        pass


def _display_experiment_results(path):
    """Display detailed results for one experiment."""
    print_header()
    print(f"\nExperiment: {path.name}\n")

    # Summary results
    csv_path = path / "results.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            print("  Summary Metrics:")
            recall_items = sorted([k for k in row.keys() if k.startswith("(")])
            for k in recall_items:
                pct = float(row[k]) * 100
                print(f"    {k}: {pct:.2f}%")
            print(f"    Total queries: {row.get('n_query', '?')}")
            print(f"    Localized: {row.get('n_localized', '?')}")

    # Timing
    timing_path = path / "timing.json"
    if timing_path.exists():
        with open(timing_path, encoding="utf-8") as f:
            timing = json.load(f)
        print("\n  Timing (per query):")
        for key, val in timing.items():
            if isinstance(val, dict):
                print(f"    {key}: mean={val.get('mean_ms', 0):.1f}ms, std={val.get('std_ms', 0):.1f}ms")

    # Per-frame summary (first 10)
    per_frame_path = path / "per_frame.csv"
    if per_frame_path.exists():
        with open(per_frame_path, newline="") as f:
            reader = csv.DictReader(f)
            frames = list(reader)
        n_loc = sum(1 for f in frames if f.get("localized_0.25m_2deg", "False") == "True")
        print(f"\n  Per-Frame (first 10 of {len(frames)}):")
        print(f"  {'Query':<35} {'t_err':>8} {'r_err':>8} {'Status'}")
        print(f"  {'-'*60}")
        for f in frames[:10]:
            q = Path(f["query"]).name[:32] if f.get("query") else "?"
            t = f.get("t_err", "?")
            r = f.get("r_err", "?")
            s = "Localized" if f.get("localized_0.25m_2deg") == "True" else "Failed"
            print(f"  {q:<35} {t:>8} {r:>8} {s}")
        if len(frames) > 10:
            print(f"  ... ({len(frames) - 10} more frames)")

    input("\nPress Enter to continue...")


def menu_ar_demo():
    """Generate AR demo visualization."""
    print_header()
    print("\nAR Demo Generator\n")
    results = list_results()
    for idx, name, path in results:
        print(f"  [{idx}] {name}")
    print(f"  [0] Back")

    choice = input("\nSelect experiment for AR demo > ").strip()
    try:
        idx = int(choice)
        if idx == 0:
            return
        if 1 <= idx <= len(results):
            _, name, path = results[idx - 1]
            _generate_ar_demo(path)
    except ValueError:
        pass


def _generate_ar_demo(results_path):
    """Generate AR demo for an experiment."""
    poses_path = results_path / "pred_poses.json"
    if not poses_path.exists():
        print(f"\nNo pred_poses.json found in {results_path}")
        print("Run the experiment first with the updated pipeline to save poses.")
        input("\nPress Enter to continue...")
        return

    # Find matching config
    config_mapping = {
        "baseline_a": "baseline_a.yaml",
        "exp_match": "exp_match.yaml",
        "exp_full": "exp_full.yaml",
        "exp_crica": "exp_crica.yaml",
        "7scenes_stairs_baseline": "7scenes_stairs_baseline.yaml",
    }
    config_name = config_mapping.get(results_path.name, f"{results_path.name}.yaml")
    config_path = CONFIG_DIR / config_name

    limit = input("Frames to render (default=20) > ").strip()
    limit = int(limit) if limit.isdigit() else 20

    cube_size = input("Cube size in meters (default=1.5) > ").strip()
    cube_size = float(cube_size) if cube_size else 1.5

    output_dir = f"outputs/ar_demo/{results_path.name}"

    cmd = (f'PYTHONPATH="." python scripts/ar_demo.py '
           f'--config {config_path} '
           f'--poses {poses_path} '
           f'--output {output_dir} '
           f'--limit {limit} '
           f'--cube_size {cube_size}')
    print(f"\nCommand: {cmd}")
    input("Press Enter to execute...")

    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        print(f"\nAR demo saved to {output_dir}")
    else:
        print(f"\nAR demo failed with exit code {result.returncode}")
    input("\nPress Enter to continue...")


def menu_export_report():
    """Export experiment results to user-specified location."""
    print_header()
    print("\nExport Report\n")
    results = list_results()
    for idx, name, path in results:
        print(f"  [{idx}] {name}")
    print(f"  [0] Back")

    choice = input("\nSelect experiment to export > ").strip()
    try:
        idx = int(choice)
        if idx == 0:
            return
        if 1 <= idx <= len(results):
            _, name, path = results[idx - 1]
            _export_report(path)
    except ValueError:
        pass


def _export_report(results_path):
    """Export report files to user-specified directory."""
    output_dir = input("Export to directory > ").strip()
    if not output_dir:
        print("No directory specified.")
        input("\nPress Enter to continue...")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find matching config
    config_mapping = {
        "baseline_a": "baseline_a.yaml",
        "exp_match": "exp_match.yaml",
        "exp_full": "exp_full.yaml",
        "exp_crica": "exp_crica.yaml",
        "7scenes_stairs_baseline": "7scenes_stairs_baseline.yaml",
    }
    config_name = config_mapping.get(results_path.name, f"{results_path.name}.yaml")
    config_path = CONFIG_DIR / config_name

    # Generate HTML report
    html_output = output_path / f"{results_path.name}_report.html"
    cmd = (f'PYTHONPATH="." python scripts/generate_report.py '
           f'--results_dir {results_path} '
           f'--config {config_path} '
           f'--output {html_output}')
    subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))

    # Copy raw data
    import shutil
    for fname in ["results.csv", "per_frame.csv", "timing.json", "pred_poses.json"]:
        src = results_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    print(f"\nReport exported to {output_dir}:")
    for f in sorted(output_path.iterdir()):
        print(f"  {f.name}")

    input("\nPress Enter to continue...")


def main_menu():
    """Main interactive menu."""
    while True:
        print_header()
        print("\nMain Menu:\n")
        print("  [1] Run Experiment")
        print("  [2] View Results")
        print("  [3] Generate AR Demo")
        print("  [4] Export Report")
        print("  [0] Exit\n")

        choice = input("Select option > ").strip()

        if choice == "1":
            menu_run_experiment()
        elif choice == "2":
            menu_view_results()
        elif choice == "3":
            menu_ar_demo()
        elif choice == "4":
            menu_export_report()
        elif choice == "0":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Press Enter to retry...")
            input()


if __name__ == "__main__":
    main_menu()
