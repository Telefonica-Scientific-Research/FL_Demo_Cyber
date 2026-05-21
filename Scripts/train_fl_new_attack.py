#!/usr/bin/env python3
"""
train_fl_new_attack.py
======================
Federated Learning – Zero-Day Attack Discovery Scenario

Scenario
--------
• All FL clients share the full dataset EXCEPT one designated "zero-day" attack.
  Each client trains a model whose output space covers all present classes
  (num_total_classes − 1 standard attacks + Normal).  The zero-day class is
  present in the model architecture but receives no local supervision until the
  discovery event — its output neurons are only shaped by FedProx regularisation
  before that point.

• 10 % of the FULL dataset (including zero-day samples) is reserved exclusively
  as the FL server's evaluation set.  It is NEVER used for training.

• After a warm-up phase (rounds 1 … --discovery-round), ONE designated client
  ("the discovering client") has the zero-day samples added to its local database
  and continues participating in FL normally.

• The experiment records the global model's per-round zero-day F1 on the server
  evaluation set, quantifying how rapidly a single client's new threat intelligence
  propagates to the entire federated network.

Outputs
-------
    Results/new_attack_results.json
    Results/figures/06_zero_day_discovery.png
    Results/figures/07_per_class_checkpoints.png

Usage
-----
    python Scripts/train_fl_new_attack.py \\
        --zero-day-attack XSS \\
        --discovery-round 10 \\
        --discovery-client 0 \\
        --rounds 20 --local-epochs 2 --n-clients 5 --sample-frac 0.3
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from preprocess import DATA_PATH, load_and_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Data partitioning for this scenario
# ---------------------------------------------------------------------------

def make_zero_day_partitions(
    X: np.ndarray,
    y: np.ndarray,
    le,
    zero_day_attack: str,
    n_clients: int = 5,
    server_frac: float = 0.10,
    seed: int = 42,
) -> tuple:
    """
    Partition data for the zero-day discovery scenario.

    Returns
    -------
    clients_base_data  : list of (X_i, y_i) — each client's pre-discovery data
                         (Normal + all standard attacks, NO zero-day samples)
    zero_day_pool      : (X_zd, y_zd) — training-split zero-day samples reserved
                         for the discovery event
    server_eval        : (X_eval, y_eval) — 10 % server evaluation set with ALL
                         classes including zero-day
    zero_day_class_idx : integer label of the zero-day attack
    """
    class_names = list(le.classes_)
    if zero_day_attack not in class_names:
        closest = [c for c in class_names if zero_day_attack.lower() in c.lower()]
        if closest:
            zero_day_attack = closest[0]
            log.warning("Zero-day attack name adjusted to '%s'", zero_day_attack)
        else:
            raise ValueError(
                f"Attack '{zero_day_attack}' not found.\n"
                f"Available classes: {class_names}"
            )
    zero_day_idx = int(le.transform([zero_day_attack])[0])
    log.info("Zero-day attack: '%s'  (class index %d)", zero_day_attack, zero_day_idx)

    # ── 1. Server evaluation set (stratified, 10 %, all classes) ─────────
    all_idx = np.arange(len(X))
    idx_train, idx_server = train_test_split(
        all_idx, test_size=server_frac, stratify=y, random_state=seed
    )
    X_server, y_server = X[idx_server], y[idx_server]
    X_tr,    y_tr    = X[idx_train],  y[idx_train]

    log.info(
        "Server eval set: %d samples  (zero-day samples in eval: %d)",
        len(y_server), int((y_server == zero_day_idx).sum()),
    )

    # ── 2. Separate zero-day from standard traffic in training pool ────────
    std_mask = y_tr != zero_day_idx
    zd_mask  = y_tr == zero_day_idx
    X_std, y_std = X_tr[std_mask], y_tr[std_mask]
    X_zd,  y_zd  = X_tr[zd_mask],  y_tr[zd_mask]

    log.info(
        "Training pool: %d standard samples | %d zero-day samples (held for discovery)",
        len(y_std), len(y_zd),
    )

    # ── 3. IID split of standard data across clients ──────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_std))
    client_splits = np.array_split(perm, n_clients)
    clients_base_data = [
        (X_std[idx], y_std[idx]) for idx in client_splits
    ]

    for i, (Xc, yc) in enumerate(clients_base_data):
        present_attacks = sorted({
            class_names[lab] for lab in np.unique(yc)
            if class_names[lab] != "Normal"
        })
        log.info("Client %d: %d samples | attacks: %s", i, len(Xc), present_attacks)

    return clients_base_data, (X_zd, y_zd), (X_server, y_server), zero_day_idx


# ---------------------------------------------------------------------------
# Training / aggregation helpers (self-contained — no dependency on other
# training scripts so this file can be run independently)
# ---------------------------------------------------------------------------

def _make_balanced_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """DataLoader with per-class balanced sampling."""
    class_counts = np.bincount(y).astype(np.float64)
    weights = 1.0 / np.maximum(class_counts[y], 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights.astype(np.float32)),
        num_samples=len(y),
        replacement=True,
    )
    ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def _local_train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    global_params: list | None = None,
    mu: float = 0.01,
) -> nn.Module:
    """FedProx local training with balanced sampling."""
    loader = _make_balanced_loader(X, y, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if mu > 0.0 and global_params is not None:
                prox = sum(
                    ((lp - gp.detach()) ** 2).sum()
                    for lp, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _predict(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048
) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        preds.extend(model(xb).argmax(1).cpu().numpy())
    return np.array(preds)


def _fedavg(
    global_model: nn.Module,
    client_models: list[nn.Module],
    client_sizes: list[int],
    discovery_client: int | None = None,
    discovery_active: bool = False,
    discovery_boost: float = 1.0,
) -> nn.Module:
    """FedAvg with optional boost for the discovering client.

    When discovery_active=True and discovery_boost>1, the discovering client's
    effective sample count is multiplied by *discovery_boost* so that its
    newly acquired zero-day knowledge is not overwhelmed by the majority of
    clients that have never seen the attack.
    """
    effective_sizes = [
        n * (discovery_boost if (discovery_active and i == discovery_client) else 1.0)
        for i, n in enumerate(client_sizes)
    ]
    total = sum(effective_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state:
        new_state[key] = sum(
            m.state_dict()[key].float() * (n / total)
            for m, n in zip(client_models, effective_sizes)
        )
    global_model.load_state_dict(new_state)
    return global_model


def _eval_on_server(
    model: nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    zero_day_idx: int,
) -> dict:
    preds = _predict(model, X_eval, device)
    labels = list(range(num_classes))
    per_class = f1_score(y_eval, preds, average=None, zero_division=0, labels=labels)
    return {
        "accuracy":    float(accuracy_score(y_eval, preds)),
        "macro_f1":    float(f1_score(y_eval, preds, average="macro",    zero_division=0, labels=labels)),
        "weighted_f1": float(f1_score(y_eval, preds, average="weighted", zero_division=0, labels=labels)),
        "zero_day_f1": float(per_class[zero_day_idx]),
        "zero_day_precision": float(
            precision_score(y_eval, preds, average=None, zero_division=0, labels=labels)[zero_day_idx]
        ),
        "zero_day_recall": float(
            recall_score(y_eval, preds, average=None, zero_division=0, labels=labels)[zero_day_idx]
        ),
        "per_class_f1": {class_names[i]: float(v) for i, v in enumerate(per_class)},
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    X, y, le = load_and_preprocess(DATA_PATH, sample_frac=args.sample_frac)
    class_names  = list(le.classes_)
    num_classes  = len(class_names)
    input_dim    = X.shape[1]

    # ── Partition data ─────────────────────────────────────────────────────
    clients_base, (X_zd, y_zd), (X_eval, y_eval), zd_idx = make_zero_day_partitions(
        X, y, le,
        zero_day_attack=args.zero_day_attack,
        n_clients=args.n_clients,
        server_frac=args.server_frac,
        seed=42,
    )
    zero_day_name = class_names[zd_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu     = args.mu
    log.info(
        "Device: %s | Clients: %d | Rounds: %d | Local epochs: %d | "
        "FedProx μ=%.4f | Discovery: client %d at round %d",
        device, args.n_clients, args.rounds, args.local_epochs, mu,
        args.discovery_client, args.discovery_round,
    )
    log.info("Zero-day attack: '%s'", zero_day_name)

    # ── Initialise global model ────────────────────────────────────────────
    global_model = build_model(input_dim, num_classes).to(device)

    per_round_metrics: list[dict] = []

    # ── FL rounds ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FL TRAINING  (zero-day='%s'  discovery=round %d  client %d)",
             zero_day_name, args.discovery_round, args.discovery_client)
    log.info("=" * 70)

    for rnd in tqdm(range(1, args.rounds + 1), desc="FL rounds"):

        discovery_active = rnd > args.discovery_round
        phase = "pre_discovery" if not discovery_active else "post_discovery"

        # Build each client's dataset for this round
        client_datasets: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (Xc, yc) in enumerate(clients_base):
            if discovery_active and i == args.discovery_client:
                # Discovering client now has zero-day samples
                Xc = np.concatenate([Xc, X_zd], axis=0)
                yc = np.concatenate([yc, y_zd], axis=0)
            client_datasets.append((Xc, yc))

        # Local training (FedProx)
        # Post-discovery: disable FedProx for the discovering client so the new
        # attack knowledge is not pulled back toward the (zero-day-ignorant) global
        # model.  All other clients keep the standard μ.
        global_params = [p.detach().clone() for p in global_model.parameters()]
        client_models, client_sizes = [], []
        for ci, (Xc, yc) in enumerate(client_datasets):
            is_discoverer = discovery_active and ci == args.discovery_client
            mu_ci = 0.0 if is_discoverer else mu
            m = copy.deepcopy(global_model)
            m = _local_train(
                m, Xc, yc, args.local_epochs, args.batch_size, args.lr,
                device, global_params=global_params, mu=mu_ci,
            )
            client_models.append(m)
            client_sizes.append(len(Xc))

        # Federated aggregation
        global_model = _fedavg(
            global_model, client_models, client_sizes,
            discovery_client=args.discovery_client,
            discovery_active=discovery_active,
            discovery_boost=args.discovery_boost,
        )

        # Evaluate on server set
        metrics = _eval_on_server(
            global_model, X_eval, y_eval, device, num_classes, class_names, zd_idx
        )
        metrics.update({
            "round": rnd,
            "phase": phase,
            "discovery_active": discovery_active,
            "rounds_since_discovery": max(0, rnd - args.discovery_round),
        })
        per_round_metrics.append(metrics)

        log.info(
            "  Round %02d [%s]  zero_day_F1=%.4f  macro_F1=%.4f  acc=%.4f",
            rnd, phase, metrics["zero_day_f1"], metrics["macro_f1"], metrics["accuracy"],
        )

    # ── Final evaluation ───────────────────────────────────────────────────
    final_preds = _predict(global_model, X_eval, device)
    log.info("\n=== FINAL GLOBAL MODEL (server eval set) ===")
    print(classification_report(
        y_eval, final_preds, target_names=class_names, zero_division=0
    ))

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "config": {
            "zero_day_attack":  zero_day_name,
            "discovery_round":  args.discovery_round,
            "discovery_client": args.discovery_client,
            "discovery_boost":  args.discovery_boost,
            "n_clients":        args.n_clients,
            "rounds":           args.rounds,
            "local_epochs":     args.local_epochs,
            "mu":               mu,
            "sample_frac":      args.sample_frac,
            "server_frac":      args.server_frac,
        },
        "class_names": class_names,
        "per_round_metrics": per_round_metrics,
        "final_metrics": per_round_metrics[-1],
    }

    out = RESULTS_DIR / "new_attack_results.json"
    out.write_text(json.dumps(results, indent=2))
    torch.save(global_model.state_dict(), RESULTS_DIR / "new_attack_global_model.pt")
    log.info("Results → %s", out)
    log.info("Model   → %s", RESULTS_DIR / "new_attack_global_model.pt")

    _generate_figures(results)
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _generate_figures(results: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg       = results["config"]
    zd_name   = cfg["zero_day_attack"]
    disc_rnd  = cfg["discovery_round"]
    rounds_m  = results["per_round_metrics"]
    class_names = results["class_names"]

    rounds       = [m["round"]        for m in rounds_m]
    zd_f1        = [m["zero_day_f1"]  for m in rounds_m]
    macro_f1     = [m["macro_f1"]     for m in rounds_m]
    accuracy     = [m["accuracy"]     for m in rounds_m]
    zd_precision = [m.get("zero_day_precision", 0) for m in rounds_m]
    zd_recall    = [m.get("zero_day_recall",    0) for m in rounds_m]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    # ── Figure 6: Zero-day discovery curve ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    # Top panel: zero-day F1 / precision / recall
    ax = axes[0]
    ax.plot(rounds, zd_f1,        "o-",  color="#F44336", linewidth=2.2, markersize=5,
            label=f"'{zd_name}' F1")
    ax.plot(rounds, zd_precision, "s--", color="#FF9800", linewidth=1.6, markersize=4,
            label=f"'{zd_name}' Precision", alpha=0.85)
    ax.plot(rounds, zd_recall,    "^--", color="#9C27B0", linewidth=1.6, markersize=4,
            label=f"'{zd_name}' Recall",    alpha=0.85)

    ax.axvline(disc_rnd + 0.5, color="#212121", linewidth=1.8, linestyle="--", zorder=5)
    ax.annotate(
        f"← Discovery event\n  (Client {cfg['discovery_client']} receives\n"
        f"  '{zd_name}' samples)",
        xy=(disc_rnd + 0.5, max(zd_f1) * 0.6),
        xytext=(disc_rnd + 1.5, max(zd_f1) * 0.45 + 0.1),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#212121"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Shade pre-/post-discovery regions
    ax.axvspan(0.5, disc_rnd + 0.5, alpha=0.07, color="#F44336", label="Pre-discovery phase")
    ax.axvspan(disc_rnd + 0.5, max(rounds) + 0.5, alpha=0.07, color="#4CAF50",
               label="Post-discovery phase")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Zero-Day Attack Discovery via Federated Learning\n"
        f"Attack: '{zd_name}' | {cfg['n_clients']} clients | "
        f"FedProx μ={cfg['mu']} | Discovery boost ×{cfg.get('discovery_boost', 1.0)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")

    # Bottom panel: global model health
    ax2 = axes[1]
    ax2.plot(rounds, macro_f1, "o-",  color="#2196F3", linewidth=2.2, markersize=5,
             label="Global Macro F1")
    ax2.plot(rounds, accuracy,  "s--", color="#4CAF50", linewidth=1.8, markersize=4,
             label="Global Accuracy", alpha=0.85)
    ax2.axvline(disc_rnd + 0.5, color="#212121", linewidth=1.8, linestyle="--")
    ax2.axvspan(0.5, disc_rnd + 0.5, alpha=0.07, color="#F44336")
    ax2.axvspan(disc_rnd + 0.5, max(rounds) + 0.5, alpha=0.07, color="#4CAF50")
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("FL Communication Round", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    _savefig("06_zero_day_discovery.png")

    # ── Figure 7: Per-class F1 at key checkpoints ────────────────────────
    checkpoints = {
        f"Round {disc_rnd}\n(before discovery)":
            next(m for m in rounds_m if m["round"] == disc_rnd),
        f"Round {disc_rnd + 1}\n(1st round post-discovery)":
            next((m for m in rounds_m if m["round"] == disc_rnd + 1), rounds_m[-1]),
        f"Round {max(rounds)}\n(final)":
            rounds_m[-1],
    }

    valid_checkpoints = {k: v for k, v in checkpoints.items() if v is not None}
    n_cp = len(valid_checkpoints)
    colors = ["#F44336", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(13, max(6, len(class_names) * 0.5)))
    y_pos = np.arange(len(class_names))
    bar_h = 0.8 / n_cp

    for j, (cp_label, cp_data) in enumerate(valid_checkpoints.items()):
        f1_vals = [cp_data["per_class_f1"].get(cn, 0.0) for cn in class_names]
        offset = (j - n_cp / 2 + 0.5) * bar_h
        bars = ax.barh(
            y_pos + offset, f1_vals, bar_h * 0.9,
            color=colors[j], alpha=0.85, label=cp_label, zorder=3
        )

    # Highlight the zero-day attack row
    zd_y = class_names.index(zd_name)
    ax.axhspan(zd_y - 0.5, zd_y + 0.5, alpha=0.12, color="#F44336", zorder=0)
    ax.text(
        1.01, zd_y, "← zero-day",
        va="center", ha="left", fontsize=8,
        color="#F44336", fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("F1 Score (server evaluation set)", fontsize=11)
    ax.set_title(
        f"Per-Class F1 at Key Checkpoints\n"
        f"Zero-Day: '{zd_name}' | Discovery at round {disc_rnd}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    _savefig("07_per_class_checkpoints.png")


def _savefig(name: str) -> None:
    import matplotlib.pyplot as plt
    path = FIGURES_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FL zero-day attack discovery scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--zero-day-attack", type=str, default="SQL_injection",
                   help="Name of the attack to withhold as 'zero-day' (default: SQL_injection).")
    p.add_argument("--discovery-round", type=int, default=5,
                   help="Round after which the discovering client gets zero-day data (default: 5).")
    p.add_argument("--discovery-client", type=int, default=0,
                   help="Index of the client that discovers the zero-day attack (default: 0).")
    p.add_argument("--rounds",       type=int,   default=20,
                   help="Total FL rounds (default: 20).")
    p.add_argument("--local-epochs", type=int,   default=2,
                   help="Local training epochs per round (default: 2).")
    p.add_argument("--n-clients",    type=int,   default=5,
                   help="Number of FL clients (default: 5).")
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--mu",           type=float, default=0.01,
                   help="FedProx proximal coefficient (default: 0.01).")
    p.add_argument("--discovery-boost", type=float, default=4.0,
                   help="Multiply discovering client weight in FedAvg post-event (default: 4.0).")
    p.add_argument("--sample-frac",  type=float, default=0.3,
                   help="Fraction of dataset to use (default: 0.3).")
    p.add_argument("--server-frac",  type=float, default=0.10,
                   help="Fraction of data reserved for server evaluation (default: 0.10).")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
