#!/usr/bin/env python3
"""
run_experiment.py
=================
End-to-end pipeline orchestrator.

Runs the full experiment in order:
    1. Centralized IDS training
    2. Federated Learning simulation
    3. Comparison report + figures

All intermediate results are cached under Results/ so individual stages
can be skipped if their output files already exist (use --force to re-run).

Usage
-----
    # Full pipeline with defaults (30% of data, fast demo)
    python Scripts/run_experiment.py

    # Full dataset, more rounds
    python Scripts/run_experiment.py --sample-frac 1.0 --rounds 30

    # Skip stages that already have results
    python Scripts/run_experiment.py --skip-centralized

    # Force re-run everything
    python Scripts/run_experiment.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "Results"

# Add Scripts/ to path so sibling modules are importable
sys.path.insert(0, str(SCRIPTS))

from train_centralized import run as run_centralized, _parse_args as _c_args
from train_federated   import run as run_federated,   _parse_args as _f_args
from compare           import run as run_compare


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end FL-Demo experiment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Shared data options
    p.add_argument("--sample-frac", type=float, default=0.3,
                   help="Fraction of dataset (default 0.3 for speed; 1.0 = full).")

    # Centralized options
    p.add_argument("--c-epochs",     type=int,   default=20)
    p.add_argument("--c-batch-size", type=int,   default=512)
    p.add_argument("--c-lr",         type=float, default=1e-3)

    # FL options
    p.add_argument("--rounds",        type=int,   default=20)
    p.add_argument("--local-epochs",  type=int,   default=5)
    p.add_argument("--num-clients",   type=int,   default=5)
    p.add_argument("--fl-lr",         type=float, default=1e-3)
    p.add_argument("--fl-batch-size", type=int,   default=512)

    # Pipeline control
    p.add_argument("--skip-centralized", action="store_true",
                   help="Skip centralized training if results already exist.")
    p.add_argument("--skip-federated",   action="store_true",
                   help="Skip federated training if results already exist.")
    p.add_argument("--force", action="store_true",
                   help="Re-run all stages even if results already exist.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _stage_centralized(args: argparse.Namespace) -> None:
    out = RESULTS_DIR / "centralized_results.json"
    if out.exists() and args.skip_centralized and not args.force:
        log.info("[SKIP] Centralized results already exist: %s", out)
        return

    log.info("\n" + "#" * 60)
    log.info("#  STAGE 1 – CENTRALIZED TRAINING")
    log.info("#" + "=" * 59)

    # Build a namespace compatible with train_centralized._parse_args
    c_ns = argparse.Namespace(
        epochs=args.c_epochs,
        batch_size=args.c_batch_size,
        lr=args.c_lr,
        sample_frac=args.sample_frac,
    )
    run_centralized(c_ns)


def _stage_federated(args: argparse.Namespace) -> None:
    out = RESULTS_DIR / "federated_results.json"
    if out.exists() and args.skip_federated and not args.force:
        log.info("[SKIP] Federated results already exist: %s", out)
        return

    log.info("\n" + "#" * 60)
    log.info("#  STAGE 2 – FEDERATED LEARNING SIMULATION")
    log.info("#" + "=" * 59)

    f_ns = argparse.Namespace(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        num_clients=args.num_clients,
        batch_size=args.fl_batch_size,
        lr=args.fl_lr,
        sample_frac=args.sample_frac,
    )
    run_federated(f_ns)


def _stage_compare() -> None:
    log.info("\n" + "#" * 60)
    log.info("#  STAGE 3 – COMPARISON REPORT & FIGURES")
    log.info("#" + "=" * 59)
    run_compare()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse()

    log.info("FL-Demo Experiment Pipeline")
    log.info("  sample_frac   = %.2f", args.sample_frac)
    log.info("  c_epochs      = %d",   args.c_epochs)
    log.info("  fl_rounds     = %d",   args.rounds)
    log.info("  fl_local_ep   = %d",   args.local_epochs)
    log.info("  fl_clients    = %d",   args.num_clients)

    _stage_centralized(args)
    _stage_federated(args)
    _stage_compare()

    log.info("\nPipeline complete.  Results in %s", RESULTS_DIR)
    log.info("Figures in %s", RESULTS_DIR / "figures")


if __name__ == "__main__":
    main()
