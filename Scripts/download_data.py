#!/usr/bin/env python3
"""
Extensible dataset downloader for the FL-Demo cybersecurity project.

Adding a new dataset
--------------------
1. Implement a download function with signature: fn(data_dir: Path) -> None
2. Register it in DATASET_REGISTRY with a short key and description.
3. Run: python download_data.py --dataset <key>

Authentication (Kaggle datasets)
---------------------------------
Option A – JSON credentials:
    Download kaggle.json from https://www.kaggle.com/settings/account
    and place it at ~/.kaggle/kaggle.json (chmod 600).
Option B – Environment variables:
    export KAGGLE_USERNAME=<your_username>
    export KAGGLE_KEY=<your_api_key>
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

def _kaggle_cli() -> str:
    """Return the path to the kaggle CLI executable."""
    cli = shutil.which("kaggle")
    if cli:
        return cli
    # Fallback: same directory as the current Python interpreter
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "kaggle CLI not found. Install it with:  pip install kaggle"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "Data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_kaggle_credentials() -> None:
    """Raise a clear error if Kaggle credentials are not configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json = kaggle_json.exists()
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not (has_json or has_env):
        raise EnvironmentError(
            "\nKaggle credentials not found.\n"
            "Steps to fix:\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. Click 'Create New Token' → downloads kaggle.json\n"
            "  3a. Place it at: ~/.kaggle/kaggle.json  (recommended)\n"
            "  3b. OR set:  export KAGGLE_USERNAME=...  export KAGGLE_KEY=...\n"
        )


def _find_file(directory: Path, filename: str) -> Path | None:
    """Recursively find a file by name under directory."""
    matches = list(directory.rglob(filename))
    return matches[0] if matches else None


def _kaggle_download_file(dataset_id: str, file_path: str, dest: Path) -> None:
    """
    Download a single file from a Kaggle dataset using the kaggle CLI.
    Handles zip extraction and moves the file to the top of dest/.
    """
    _check_kaggle_credentials()
    dest.mkdir(parents=True, exist_ok=True)

    filename = Path(file_path).name
    final_path = dest / filename
    if final_path.exists():
        log.info("File already present: %s – skipping download.", final_path)
        return

    cmd = [
        _kaggle_cli(), "datasets", "download",
        "-d", dataset_id,
        "-f", file_path,
        "-p", str(dest),
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Kaggle wraps the file in a .zip
    zip_path = dest / f"{filename}.zip"
    if zip_path.exists():
        log.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()

    # If extraction placed file in a sub-directory, move it up
    if not final_path.exists():
        found = _find_file(dest, filename)
        if found:
            shutil.move(str(found), str(final_path))
            log.info("Moved file to %s", final_path)
        else:
            raise FileNotFoundError(
                f"Could not locate '{filename}' after extraction in {dest}"
            )

    log.info("Saved to %s", final_path)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _download_edge_iiotset(data_dir: Path) -> None:
    """
    Download the Edge-IIoTset DNN-ready CSV from Kaggle.

    Reference:
        Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security
        Dataset of IoT and IIoT Applications for Centralized and Federated Learning"
        TechRxiv 2022. DOI: 10.36227/techrxiv.18857336.v1
    """
    _kaggle_download_file(
        dataset_id="mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        file_path="Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
        dest=data_dir / "EdgeIIoTset",
    )


# ---------------------------------------------------------------------------
# Dataset registry
# Add new datasets by appending entries here.
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "edge_iiotset": {
        "description": (
            "Edge-IIoTset Cyber Security Dataset of IoT & IIoT (Kaggle) – "
            "14 attack types across 5 threat categories, DNN-ready CSV."
        ),
        "download_fn": _download_edge_iiotset,
    },
    # ── Future datasets ───────────────────────────────────────────────────
    # "cicids2017": {
    #     "description": "CIC-IDS 2017 Intrusion Detection Evaluation Dataset",
    #     "download_fn": _download_cicids2017,
    # },
    # "unsw_nb15": {
    #     "description": "UNSW-NB15 Network Intrusion Dataset",
    #     "download_fn": _download_unsw_nb15,
    # },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download cybersecurity datasets for FL-Demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available datasets:\n"
        + "\n".join(f"  {k:20s} {v['description']}" for k, v in DATASET_REGISTRY.items()),
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all registered datasets).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT,
        help=f"Root data directory (default: {DATA_ROOT}).",
    )
    args = parser.parse_args()

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]

    for key in targets:
        entry = DATASET_REGISTRY[key]
        log.info("=== Dataset: %s ===", key)
        log.info("    %s", entry["description"])
        entry["download_fn"](args.data_dir)

    log.info("All downloads complete.")


if __name__ == "__main__":
    main()
