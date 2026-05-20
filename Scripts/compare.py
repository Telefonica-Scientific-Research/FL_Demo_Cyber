#!/usr/bin/env python3
"""
compare.py
==========
Generates a comprehensive comparison report between the centralized IDS
baseline and the Federated Learning approach.

Requires both result files produced by the training scripts:
    Results/centralized_results.json
    Results/federated_results.json

Outputs (all saved under Results/figures/)
-----------------------------------------
    01_overall_comparison.png    – accuracy / macro-F1 / weighted-F1 side-by-side
    02_per_class_f1.png          – per-attack-type F1: centralized vs. FL global
    03_fl_convergence.png        – FL macro-F1 and accuracy over rounds
    04_unseen_attack_benefit.png – KEY FIGURE: local vs. FL F1 on attacks
                                   each client had NEVER locally observed
    05_client_overview.png       – per-client macro-F1 before/after federation

A text summary table is also printed to stdout.

Usage
-----
    python Scripts/compare.py
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"
FIGURES_DIR = RESULTS_DIR / "figures"

BLUE  = "#2196F3"
GREEN = "#4CAF50"
RED   = "#F44336"
AMBER = "#FF9800"
PURPLE = "#9C27B0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 140,
})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load() -> tuple[dict, dict]:
    c_path = RESULTS_DIR / "centralized_results.json"
    f_path = RESULTS_DIR / "federated_results.json"
    for p in (c_path, f_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\n"
                "Run train_centralized.py and train_federated.py first."
            )
    return (
        json.loads(c_path.read_text()),
        json.loads(f_path.read_text()),
    )


# ---------------------------------------------------------------------------
# Figure 1 – Overall performance comparison
# ---------------------------------------------------------------------------

def _fig_overall(c: dict, f: dict) -> None:
    metrics   = ["accuracy", "macro_f1", "weighted_f1"]
    labels    = ["Accuracy", "Macro F1", "Weighted F1"]
    c_vals    = [c[m] for m in metrics]
    fl_vals   = [f["fl_global"][m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, c_vals,  w, color=BLUE,  label="Centralized",        zorder=3)
    b2 = ax.bar(x + w / 2, fl_vals, w, color=GREEN, label="Federated (FL)",     zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Overall Performance: Centralized vs. Federated Learning\n"
        "(Edge-IIoTset, 5 non-IID clients, privacy-preserving)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.bar_label(b1, fmt="%.3f", padding=4, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=4, fontsize=9)
    plt.tight_layout()
    _save("01_overall_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2 – Per-class F1
# ---------------------------------------------------------------------------

def _fig_per_class(c: dict, f: dict) -> None:
    classes = list(c["per_class_f1"].keys())
    c_f1    = [c["per_class_f1"][cl]                           for cl in classes]
    fl_f1   = [f["fl_global"]["per_class_f1"].get(cl, 0.0)    for cl in classes]

    df = pd.DataFrame({"class": classes, "Centralized": c_f1, "Federated": fl_f1})
    df = df.sort_values("Federated", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.45)))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["Centralized"], 0.38, color=BLUE,  label="Centralized", zorder=3)
    ax.barh(y + 0.2, df["Federated"],   0.38, color=GREEN, label="Federated",   zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["class"], fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(
        "Per-Class F1: Centralized vs. Federated Learning",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("02_per_class_f1.png")


# ---------------------------------------------------------------------------
# Figure 3 – FL convergence curve
# ---------------------------------------------------------------------------

def _fig_convergence(f: dict) -> None:
    rounds   = [m["round"]    for m in f["fl_round_metrics"]]
    macro_f1 = [m["macro_f1"] for m in f["fl_round_metrics"]]
    accuracy = [m["accuracy"] for m in f["fl_round_metrics"]]

    # Optional: overlay centralized horizontal line if available
    c_macro = None
    c_path  = RESULTS_DIR / "centralized_results.json"
    if c_path.exists():
        c_macro = json.loads(c_path.read_text())["macro_f1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, macro_f1, "o-",  color=GREEN, linewidth=2, markersize=4,
            label="FL Global – Macro F1")
    ax.plot(rounds, accuracy, "s--", color=AMBER, linewidth=2, markersize=4,
            label="FL Global – Accuracy")
    if c_macro is not None:
        ax.axhline(c_macro, color=BLUE, linestyle=":", linewidth=1.8,
                   label=f"Centralized Macro F1 = {c_macro:.3f}")

    ax.set_xlabel("FL Communication Round", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Federated Learning Convergence\n(data never leaves individual clients)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save("03_fl_convergence.png")


# ---------------------------------------------------------------------------
# Figure 4 – KEY FIGURE: unseen-attack benefit
# ---------------------------------------------------------------------------

def _fig_unseen_attacks(f: dict) -> None:
    """
    For each attack type that appeared as *unseen* in at least one client,
    compare the average F1 achieved by local-only training vs. the FL global
    model.  This is the core visual proof of FL value.
    """
    rows = []
    for cid, cdata in f["client_comparison"].items():
        for atk, scores in cdata["per_unseen_attack"].items():
            rows.append({
                "client": f"C{cdata['client']} ({cdata['local_attack_category']})",
                "attack": atk,
                "Local F1": scores["local_f1"],
                "FL F1":    scores["fl_f1"],
                "delta":    scores["delta"],
            })

    if not rows:
        log.warning("No unseen-attack data found – skipping figure 4.")
        return

    df = pd.DataFrame(rows)
    avg = (
        df.groupby("attack")[["Local F1", "FL F1"]]
        .mean()
        .sort_values("FL F1", ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(avg) * 0.50)))
    y = np.arange(len(avg))
    ax.barh(y - 0.2, avg["Local F1"], 0.38, color=RED,   alpha=0.85, zorder=3,
            label="Local-only (attack never seen locally)")
    ax.barh(y + 0.2, avg["FL F1"],    0.38, color=GREEN, alpha=0.85, zorder=3,
            label="FL global model (federated knowledge)")

    ax.set_yticks(y)
    ax.set_yticklabels(avg["attack"], fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("F1 Score (averaged over clients that never saw this attack)", fontsize=10)
    ax.set_title(
        "FL Value: Detection of Attacks NEVER Seen Locally\n"
        "FL enables zero-day-like generalization via privacy-preserving collaboration",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
               label="F1 = 0.5 reference")
    ax.legend(fontsize=10)

    # Annotate delta
    for i, row in avg.iterrows():
        delta = row["FL F1"] - row["Local F1"]
        ax.text(
            max(row["FL F1"], row["Local F1"]) + 0.01, i,
            f"Δ {delta:+.2f}", va="center", fontsize=8, color="black"
        )

    plt.tight_layout()
    _save("04_unseen_attack_benefit.png")


# ---------------------------------------------------------------------------
# Figure 5 – Per-client macro F1 overview
# ---------------------------------------------------------------------------

def _fig_client_overview(f: dict) -> None:
    records = list(f["client_comparison"].values())
    if not records:
        return

    labels  = [f"C{r['client']}\n({r['local_attack_category']})" for r in records]
    local_f = [r["local_macro_f1"] for r in records]
    fl_f    = [r["fl_macro_f1"]    for r in records]

    x = np.arange(len(records))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.5), 5))
    b1 = ax.bar(x - w / 2, local_f, w, color=RED,   alpha=0.85, label="Local-only", zorder=3)
    b2 = ax.bar(x + w / 2, fl_f,    w, color=GREEN, alpha=0.85, label="After FL",   zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro F1 (global test set)", fontsize=11)
    ax.set_title(
        "Per-Client Macro F1: Local Training vs. After Federation\n"
        "(Global test set – all attack types)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    _save("05_client_overview.png")


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------

def _print_summary(c: dict, f: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FL DEMO – SUMMARY REPORT")
    print(f"  Dataset: Edge-IIoTset  |  Model: DNN  |  Algorithm: FedAvg")
    print(sep)

    print(f"\n{'Metric':<30} {'Centralized':>14} {'Federated':>14} {'Δ':>10}")
    print("-" * 72)
    for key, label in [
        ("accuracy",    "Accuracy"),
        ("macro_f1",    "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]:
        cv = c[key]
        fv = f["fl_global"][key]
        print(f"  {label:<28} {cv:>14.4f} {fv:>14.4f} {fv - cv:>+10.4f}")

    if "macro_precision" in f["fl_global"]:
        print(f"\n  {'FL Macro Precision':<28} {'':>14} "
              f"{f['fl_global']['macro_precision']:>14.4f}")
        print(f"  {'FL Macro Recall':<28} {'':>14} "
              f"{f['fl_global']['macro_recall']:>14.4f}")

    # FL convergence
    rounds = f["fl_round_metrics"]
    if rounds:
        best_rnd = max(rounds, key=lambda r: r["macro_f1"])
        print(f"\n  FL best macro F1 achieved at round {best_rnd['round']}: "
              f"{best_rnd['macro_f1']:.4f}")

    # Unseen attack improvements
    all_deltas = []
    for cdata in f["client_comparison"].values():
        for scores in cdata["per_unseen_attack"].values():
            all_deltas.append(scores["delta"])

    if all_deltas:
        print(f"\n  Unseen-Attack F1 Improvement (via FL vs. local-only):")
        print(f"    Mean  Δ : {np.mean(all_deltas):+.4f}")
        print(f"    Max   Δ : {max(all_deltas):+.4f}")
        print(f"    Min   Δ : {min(all_deltas):+.4f}")
        n_improved = sum(1 for d in all_deltas if d > 0.05)
        print(f"    Attacks with Δ > 0.05: {n_improved}/{len(all_deltas)}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    central, federated = _load()

    _fig_overall(central, federated)
    _fig_per_class(central, federated)
    _fig_convergence(federated)
    _fig_unseen_attacks(federated)
    _fig_client_overview(federated)
    _print_summary(central, federated)

    log.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    run()
