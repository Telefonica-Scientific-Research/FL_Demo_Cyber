#!/usr/bin/env python3
"""
train_federated.py
==================
Federated Learning simulation using FedAvg.

Scenario – "Zero-Day Detection via FL"
---------------------------------------
Five FL clients each simulate an isolated IoT network segment that has only
ever observed *normal* traffic plus the attacks specific to its threat domain
(DoS/DDoS, Recon, Injection, MITM, Malware).

Key FL Value Proposition
-------------------------
A client that has NEVER locally seen, e.g., SQL-injection traffic will still
gain the ability to detect it after federated rounds because the global model
aggregates knowledge from the Injection-specialist client.  This is impossible
with purely local training – and is achieved WITHOUT any raw data leaving the
respective client.

Comparison baseline
--------------------
Before FL begins, each client also trains a local-only model (same total
number of epochs, no aggregation) to serve as the "no-federation" baseline.

Usage
-----
    python Scripts/train_federated.py [--rounds 20] [--local-epochs 5]
                                      [--num-clients 5] [--sample-frac 0.3]

Outputs
-------
    Results/federated_global_model.pt    – final aggregated model
    Results/federated_results.json       – full metrics including per-client
                                           comparison on unseen attack types
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from preprocess import DATA_PATH, load_and_preprocess, make_fl_partitions

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_balanced_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> DataLoader:
    """
    DataLoader with per-class balanced sampling via WeightedRandomSampler.

    Each class contributes equally to every mini-batch, preventing the
    majority Normal class from dominating local training and causing
    model collapse.
    """
    class_counts = np.bincount(y).astype(np.float64)
    # Weight each sample inversely proportional to its class frequency
    sample_weights = 1.0 / np.maximum(class_counts[y], 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(y),
        replacement=True,
    )
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _local_train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    global_model_params: list | None = None,
    mu: float = 0.0,
) -> nn.Module:
    """
    Train a client model locally using balanced mini-batches.

    Balanced sampling (WeightedRandomSampler) ensures every attack class
    contributes equally to each batch, preventing the majority Normal class
    from causing model collapse on highly imbalanced non-IID data.

    Parameters
    ----------
    global_model_params : fixed snapshot of global model parameters for FedProx.
    mu : FedProx proximal coefficient (0 = standard FedAvg).
        Penalises ||w_local - w_global||² to limit client drift on non-IID data.
    """
    loader = _make_balanced_loader(X, y, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)

            # FedProx: (μ/2) * ||w_local − w_global||²
            if mu > 0.0 and global_model_params is not None:
                prox = sum(
                    ((lp - gp.detach()) ** 2).sum()
                    for lp, gp in zip(model.parameters(), global_model_params)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    for (xb,) in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.array(preds)


def _fedavg(
    global_model: nn.Module,
    client_models: list[nn.Module],
    client_sizes: list[int],
) -> nn.Module:
    """
    Weighted FedAvg: aggregate client weights proportional to dataset size.

    ∀ layer l:  w_global_l = Σ_i (n_i / N) · w_client_i_l
    where N = Σ_i n_i
    """
    total = sum(client_sizes)
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state:
        new_state[key] = sum(
            m.state_dict()[key].float() * (n / total)
            for m, n in zip(client_models, client_sizes)
        )
    global_model.load_state_dict(new_state)
    return global_model


def _eval_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    num_classes: int,
) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy":    float(accuracy_score(targets, preds)),
        "macro_f1":    float(f1_score(targets, preds, average="macro",    zero_division=0, labels=labels)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0, labels=labels)),
        "macro_precision": float(precision_score(targets, preds, average="macro",    zero_division=0, labels=labels)),
        "macro_recall":    float(recall_score(targets, preds,   average="macro",    zero_division=0, labels=labels)),
        "per_class_f1": {
            class_names[i]: float(v)
            for i, v in enumerate(
                f1_score(targets, preds, average=None, zero_division=0, labels=labels)
            )
        },
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load & partition data ─────────────────────────────────────────────
    X, y, le = load_and_preprocess(DATA_PATH, sample_frac=args.sample_frac)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    input_dim = X.shape[1]

    clients_data, (X_test, y_test), client_categories = make_fl_partitions(
        X, y, le, num_clients=args.num_clients, test_size=0.20, seed=42
    )
    n_clients = len(clients_data)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = getattr(args, "mu", 0.01)
    log.info(
        "Device: %s  |  Clients: %d  |  Rounds: %d  |  Local epochs: %d  |  FedProx μ=%.4f",
        device, n_clients, args.rounds, args.local_epochs, mu,
    )
    log.info("Using balanced mini-batch sampling to prevent majority-class collapse.")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1 – Local-only baseline
    # Each client trains independently for rounds × local_epochs total epochs
    # using balanced sampling, but NO knowledge is shared across clients.
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 1 – LOCAL-ONLY BASELINE (balanced sampling, no federation)")
    log.info("=" * 60)

    local_models: list[nn.Module] = []
    local_metrics: list[dict] = []

    total_local_epochs = args.rounds * args.local_epochs
    for i, (X_c, y_c) in enumerate(clients_data):
        m = build_model(input_dim, num_classes).to(device)
        m = _local_train(
            m, X_c, y_c, total_local_epochs, args.batch_size, args.lr,
            device, global_model_params=None, mu=0.0,
        )
        local_models.append(m)

        preds_local = _predict(m, X_test, device)
        metrics = _eval_metrics(preds_local, y_test, class_names, num_classes)
        local_metrics.append(metrics)

        log.info(
            "  Client %d [%s]  macro_F1=%.4f  acc=%.4f",
            i, client_categories[i], metrics["macro_f1"], metrics["accuracy"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2 – Federated Learning (FedProx / FedAvg simulation)
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 2 – FEDERATED LEARNING (FedProx μ=%.4f)", mu)
    log.info("=" * 60)

    global_model = build_model(input_dim, num_classes).to(device)
    fl_round_metrics: list[dict] = []

    for rnd in tqdm(range(1, args.rounds + 1), desc="FL rounds"):
        # Each client gets a fresh copy of the global model and trains locally
        round_client_models: list[nn.Module] = []
        round_client_sizes: list[int] = []

        global_params = [p.detach().clone() for p in global_model.parameters()]
        for X_c, y_c in clients_data:
            m = copy.deepcopy(global_model)
            m = _local_train(
                m, X_c, y_c, args.local_epochs, args.batch_size, args.lr,
                device, global_model_params=global_params, mu=mu,
            )
            round_client_models.append(m)
            round_client_sizes.append(len(X_c))

        # Aggregate (FedAvg)
        global_model = _fedavg(global_model, round_client_models, round_client_sizes)

        # Track global model performance after each round
        preds_global = _predict(global_model, X_test, device)
        rnd_metrics = _eval_metrics(preds_global, y_test, class_names, num_classes)
        rnd_metrics["round"] = rnd
        fl_round_metrics.append(rnd_metrics)

        log.info(
            "  Round %02d/%02d | macro_F1=%.4f  acc=%.4f",
            rnd, args.rounds, rnd_metrics["macro_f1"], rnd_metrics["accuracy"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3 – Per-client comparison: local-only vs. FL global model
    # Focus: detection of attack types NEVER seen locally
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 – PER-CLIENT COMPARISON (local vs. FL)")
    log.info("=" * 60)

    global_preds = _predict(global_model, X_test, device)
    client_comparison: dict[str, dict] = {}

    for i, (cat, local_m) in enumerate(zip(client_categories, local_models)):
        local_preds = _predict(local_m, X_test, device)

        # Classes this client SAW during local training (attack classes only)
        client_class_ids = set(clients_data[i][1].tolist())
        normal_id = next(
            (j for j, c in enumerate(class_names) if c.lower() == "normal"), None
        )
        seen_attack_ids = {
            ci for ci in client_class_ids if ci != normal_id
        }
        unseen_attack_ids = {
            j for j in range(num_classes)
            if j not in seen_attack_ids and j != normal_id
        }

        local_f1_per_class = f1_score(
            y_test, local_preds, average=None, zero_division=0,
            labels=list(range(num_classes))
        )
        fl_f1_per_class = f1_score(
            y_test, global_preds, average=None, zero_division=0,
            labels=list(range(num_classes))
        )

        unseen_detail: dict[str, dict] = {}
        for j in sorted(unseen_attack_ids):
            cn = class_names[j]
            lf = float(local_f1_per_class[j])
            ff = float(fl_f1_per_class[j])
            unseen_detail[cn] = {
                "local_f1": lf,
                "fl_f1": ff,
                "delta": round(ff - lf, 4),
            }

        avg_delta_unseen = float(
            np.mean([v["delta"] for v in unseen_detail.values()])
        ) if unseen_detail else 0.0

        record = {
            "client": i,
            "local_attack_category": cat,
            "seen_attacks": sorted(class_names[j] for j in seen_attack_ids),
            "unseen_attacks_count": len(unseen_attack_ids),
            "local_macro_f1":    float(f1_score(y_test, local_preds,  average="macro", zero_division=0)),
            "fl_macro_f1":       float(f1_score(y_test, global_preds, average="macro", zero_division=0)),
            "local_accuracy":    float(accuracy_score(y_test, local_preds)),
            "fl_accuracy":       float(accuracy_score(y_test, global_preds)),
            "avg_delta_unseen":  avg_delta_unseen,
            "per_unseen_attack": unseen_detail,
        }
        client_comparison[f"client_{i}"] = record

        log.info(
            "\nClient %d [%s]:", i, cat
        )
        log.info(
            "  Macro F1  local=%.4f  →  FL=%.4f  (Δ %+.4f)",
            record["local_macro_f1"], record["fl_macro_f1"],
            record["fl_macro_f1"] - record["local_macro_f1"],
        )
        log.info("  Avg F1 gain on %d unseen attacks: %+.4f",
                 len(unseen_attack_ids), avg_delta_unseen)
        for atk, scores in sorted(unseen_detail.items(), key=lambda kv: kv[1]["delta"], reverse=True):
            log.info(
                "    %-40s  local=%.3f  FL=%.3f  Δ=%+.3f",
                atk, scores["local_f1"], scores["fl_f1"], scores["delta"],
            )

    # ── Final global model evaluation ─────────────────────────────────────
    final_metrics = _eval_metrics(global_preds, y_test, class_names, num_classes)
    log.info("\n=== FINAL FL GLOBAL MODEL ===")
    log.info("Accuracy       : %.4f", final_metrics["accuracy"])
    log.info("Macro F1       : %.4f", final_metrics["macro_f1"])
    log.info("Weighted F1    : %.4f", final_metrics["weighted_f1"])
    log.info("Macro Precision: %.4f", final_metrics["macro_precision"])
    log.info("Macro Recall   : %.4f", final_metrics["macro_recall"])
    print(
        "\n"
        + classification_report(
            y_test, global_preds, target_names=class_names, zero_division=0
        )
    )

    # ── Aggregate results dict ─────────────────────────────────────────────
    results = {
        "fl_global":        {**final_metrics, "class_names": class_names},
        "fl_round_metrics": fl_round_metrics,
        "client_comparison": client_comparison,
        "local_baselines": [
            {
                "client": i,
                "category": client_categories[i],
                **local_metrics[i],
            }
            for i in range(n_clients)
        ],
    }

    out = RESULTS_DIR / "federated_results.json"
    out.write_text(json.dumps(results, indent=2))
    torch.save(global_model.state_dict(), RESULTS_DIR / "federated_global_model.pt")
    log.info("\nResults → %s", out)
    log.info("Model   → %s", RESULTS_DIR / "federated_global_model.pt")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Federated Learning (FedAvg) simulation on Edge-IIoTset."
    )
    p.add_argument("--rounds", type=int, default=20,
                   help="Number of FL communication rounds (default: 20).")
    p.add_argument("--local-epochs", type=int, default=2,
                   help="Local training epochs per round per client (default: 5).")
    p.add_argument("--num-clients", type=int, default=5,
                   help="Number of FL clients / threat domains (default: 5).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Mini-batch size for local training (default: 512).")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Local learning rate (default: 5e-4).")
    p.add_argument("--mu", type=float, default=0.01,
                   help="FedProx proximal coefficient μ (default: 0.01). "
                        "Set 0 for standard FedAvg.")
    p.add_argument("--sample-frac", type=float, default=0.3,
                   help="Fraction of dataset to use (default: 0.3 for speed). "
                        "Set 1.0 for the full dataset.")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
