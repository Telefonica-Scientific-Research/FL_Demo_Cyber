#!/usr/bin/env python3
"""
train_centralized.py
====================
Centralized baseline: train a DNN on the FULL labelled dataset (all clients'
data pooled together) and evaluate on a held-out test set.

This represents the theoretical upper bound where a single organisation
has unrestricted access to all network traffic from every segment.

Usage
-----
    python Scripts/train_centralized.py [--epochs 20] [--batch-size 512]
                                        [--lr 1e-3] [--sample-frac 0.3]

Outputs
-------
    Results/centralized_model.pt          – best model checkpoint
    Results/centralized_results.json      – metrics (accuracy, F1, per-class F1)
"""

import argparse
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
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Allow running from project root or Scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from preprocess import DATA_PATH, load_and_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


@torch.no_grad()
def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return np.array(preds), np.array(targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    X, y, le = load_and_preprocess(DATA_PATH, sample_frac=args.sample_frac)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=42
    )
    log.info(
        "Split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test)
    )

    # ── Class weights to handle imbalance ─────────────────────────────────
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = torch.tensor(1.0 / np.maximum(counts, 1))
    class_weights /= class_weights.sum()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_loader = _make_loader(X_train, y_train, args.batch_size)
    val_loader = _make_loader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, args.batch_size, shuffle=False)

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    model = build_model(X.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1, best_state = 0.0, None
    history = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Centralized training"):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_targets = _predict(model, val_loader, device)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        scheduler.step()

        history.append(
            {"epoch": epoch, "loss": tr_loss, "train_acc": tr_acc, "val_macro_f1": val_f1}
        )
        log.info(
            "Epoch %02d/%02d | loss=%.4f  train_acc=%.4f  val_macro_f1=%.4f",
            epoch, args.epochs, tr_loss, tr_acc, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────
    preds, targets = _predict(model, test_loader, device)

    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
    per_class_f1 = {
        class_names[i]: float(v)
        for i, v in enumerate(
            f1_score(targets, preds, average=None, zero_division=0,
                     labels=list(range(num_classes)))
        )
    }

    log.info("\n=== Centralized Test Results ===")
    log.info("Accuracy     : %.4f", acc)
    log.info("Macro F1     : %.4f", macro_f1)
    log.info("Weighted F1  : %.4f", weighted_f1)
    print(
        "\n"
        + classification_report(
            targets, preds, target_names=class_names, zero_division=0
        )
    )

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "class_names": class_names,
        "training_history": history,
    }

    (RESULTS_DIR / "centralized_results.json").write_text(
        json.dumps(results, indent=2)
    )
    torch.save(best_state, RESULTS_DIR / "centralized_model.pt")
    log.info("Results → %s", RESULTS_DIR / "centralized_results.json")
    log.info("Model   → %s", RESULTS_DIR / "centralized_model.pt")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centralized DNN training on Edge-IIoTset."
    )
    p.add_argument("--epochs", type=int, default=20,
                   help="Number of training epochs (default: 20).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Mini-batch size (default: 512).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate (default: 1e-3).")
    p.add_argument("--sample-frac", type=float, default=0.3,
                   help="Fraction of dataset to use (default: 0.3 for speed). "
                        "Set 1.0 for the full dataset.")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
