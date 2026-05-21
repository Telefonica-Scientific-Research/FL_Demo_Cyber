#!/usr/bin/env python3
"""
train_fl_new_attack.py
======================
Federated Learning – Incremental Attack Discovery using **Flower (flwr)**.

Scenario
--------
All FL clients start with an IID slice of the FULL dataset, including a
small baseline share of the target attack class ("SQL_injection" by default).
After --discovery-round warm-up rounds, Client 0 ("the discoverer") starts
receiving a large additional pool of that attack class in its local traffic,
simulating a network that is suddenly being targeted by an attack the FL
system partially already knows.

The server holds 10 % of the full dataset (stratified) for evaluation and
measures per-round F1 on the target class to quantify how fast threat
intelligence propagates across the federation.

Two FL strategies are compared
-------------------------------
FedAvg (baseline)
    Standard weighted-average aggregation proportional to local dataset sizes.
    The discoverer's new knowledge is diluted by the other clients.

FedDiv (Power-Weighted Divergence FedAvg)
    After each round each client's effective weight is proportional to the
    p-th power of the L2 distance between its local model and the current
    global model:

        w_i  ∝  ||w_i_local − w_global||₂^p

    p=1 (linear): mild amplification of divergent clients.
    p=2 (quadratic, default): if client 0 has 10× higher divergence than
         the mean of others, it captures ~96 % of the aggregate weight
         instead of ~71 % with p=1 — propagating its new knowledge faster.

Data partitioning
-----------------
  server_frac (10 %)   : stratified eval set — never used for training
  discovery_frac (80 %) : target-attack samples held for discovery pool
  remaining 20 % of target attack + all other traffic : IID split N ways

  → each client starts with ALL 15 attack classes (including ~20 % of the
    target), so the global model already detects it at round 1 but
    imperfectly; after the discovery event it improves measurably.

Outputs
-------
    Results/new_attack_results.json
    Results/figures/06_zeroday_comparison.png
    Results/figures/07_per_class_final.png

Usage
-----
    python Scripts/train_fl_new_attack.py \\
        --target-attack SQL_injection \\
        --discovery-round 5 \\
        --rounds 20 --local-epochs 2 --n-clients 5 --sample-frac 0.3
"""

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import (
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from preprocess import DATA_PATH, load_and_preprocess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("flwr").setLevel(logging.WARNING)
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------

def make_iid_partitions(X, y, le, target_attack, n_clients=5,
                         server_frac=0.10, discovery_frac=0.80, seed=42):
    """
    IID partition where every client starts with ALL attack classes.

    A fraction *discovery_frac* of target-attack samples is withheld as a
    discovery pool to be injected into Client 0 after the discovery round.
    The remaining (1-discovery_frac) is included in the regular IID split so
    the global model already has weak baseline detection from round 1.

    Returns
    -------
    clients_base  : list of (X_i, y_i) — each client's initial training data
    discovery_pool: (X_pool, y_pool) — extra samples added to Client 0 post-event
    server_eval   : (X_eval, y_eval) — held-out evaluation set (all classes)
    target_idx    : integer label of the target attack
    """
    class_names = list(le.classes_)
    if target_attack not in class_names:
        closest = [c for c in class_names if target_attack.lower() in c.lower()]
        if closest:
            target_attack = closest[0]
            log.warning("Attack name adjusted to '%s'", target_attack)
        else:
            raise ValueError(
                f"Attack '{target_attack}' not found. "
                f"Available: {class_names}"
            )
    target_idx = int(le.transform([target_attack])[0])
    log.info("Target attack: '%s'  (class index %d)", target_attack, target_idx)

    # ── 1. Server eval set (stratified 10 %, all classes) ─────────────────
    all_idx = np.arange(len(X))
    idx_tr, idx_sv = train_test_split(
        all_idx, test_size=server_frac, stratify=y, random_state=seed
    )
    X_sv, y_sv = X[idx_sv], y[idx_sv]
    X_tr, y_tr = X[idx_tr], y[idx_tr]

    log.info("Server eval: %d samples (target in eval: %d)",
             len(y_sv), int((y_sv == target_idx).sum()))

    # ── 2. Split target-attack samples: base (20%) vs discovery pool (80%) ─
    ta_mask  = y_tr == target_idx
    X_ta     = X_tr[ta_mask];   y_ta    = y_tr[ta_mask]
    X_other  = X_tr[~ta_mask];  y_other = y_tr[~ta_mask]

    X_ta_base, X_pool, y_ta_base, y_pool = train_test_split(
        X_ta, y_ta, test_size=discovery_frac, random_state=seed
    )
    log.info("Target-attack split: %d base (IID) | %d discovery pool",
             len(y_ta_base), len(y_pool))

    # ── 3. IID split of (all other traffic + 20% target attack) ───────────
    X_iid = np.concatenate([X_other, X_ta_base])
    y_iid = np.concatenate([y_other, y_ta_base])

    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(X_iid))
    splits = np.array_split(perm, n_clients)
    clients_base = [(X_iid[idx], y_iid[idx]) for idx in splits]

    for i, (Xc, yc) in enumerate(clients_base):
        ta_cnt = int((yc == target_idx).sum())
        log.info("Client %d: %d samples  (target class: %d samples)",
                 i, len(Xc), ta_cnt)

    return clients_base, (X_pool, y_pool), (X_sv, y_sv), target_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_balanced_loader(X, y, batch_size):
    counts  = np.bincount(y).astype(np.float64)
    weights = 1.0 / np.maximum(counts[y], 1.0)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights.astype(np.float32)), len(y), replacement=True
    )
    ds = TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


@torch.no_grad()
def _predict(model, X, device, batch_size=2048):
    model.eval()
    preds = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        preds.extend(model(xb).argmax(1).cpu().numpy())
    return np.array(preds)


def _params_to_model(ndarrays, input_dim, num_classes, device):
    model = build_model(input_dim, num_classes).to(device)
    state_dict = OrderedDict(
        (k, torch.from_numpy(np.copy(v)))
        for k, v in zip(model.state_dict().keys(), ndarrays)
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def _eval_on_server(model, X_eval, y_eval, device, num_classes,
                    class_names, target_idx):
    preds  = _predict(model, X_eval, device)
    labels = list(range(num_classes))
    pc_f1  = f1_score(y_eval, preds, average=None,
                      zero_division=0, labels=labels)
    return {
        "accuracy":         float(accuracy_score(y_eval, preds)),
        "macro_f1":         float(f1_score(y_eval, preds, average="macro",
                                            zero_division=0, labels=labels)),
        "weighted_f1":      float(f1_score(y_eval, preds, average="weighted",
                                            zero_division=0, labels=labels)),
        "target_f1":        float(pc_f1[target_idx]),
        "target_precision": float(precision_score(y_eval, preds, average=None,
                                                   zero_division=0,
                                                   labels=labels)[target_idx]),
        "target_recall":    float(recall_score(y_eval, preds, average=None,
                                                zero_division=0,
                                                labels=labels)[target_idx]),
        "per_class_f1":     {class_names[i]: float(v)
                              for i, v in enumerate(pc_f1)},
    }


# ---------------------------------------------------------------------------
# Flower client (shared by both strategies)
# ---------------------------------------------------------------------------

class DiscoveryClient(fl.client.NumPyClient):
    """
    Standard Flower client for the discovery scenario.

    The server injects two config keys each round:
      proximal_mu      – FedProx regularisation coefficient (0.0 = disabled)
      discovery_active – whether this round is post-discovery
    """

    def __init__(self, cid, X_base, y_base, X_pool, y_pool,
                 num_classes, input_dim, local_epochs, batch_size, lr):
        self.cid          = int(cid)
        self.X_base       = X_base
        self.y_base       = y_base
        self.X_pool       = X_pool   # None for non-discovery clients
        self.y_pool       = y_pool
        self.device       = torch.device("cpu")
        self.model        = build_model(input_dim, num_classes).to(self.device)
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.lr           = lr

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            (k, torch.from_numpy(np.copy(v)))
            for k, v in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mu               = float(config.get("proximal_mu",    0.0))
        discovery_active = bool(config.get("discovery_active", False))

        if discovery_active and self.X_pool is not None:
            X = np.concatenate([self.X_base, self.X_pool])
            y = np.concatenate([self.y_base, self.y_pool])
        else:
            X, y = self.X_base, self.y_base

        global_params = [p.clone().detach() for p in self.model.parameters()]
        loader    = _make_balanced_loader(X, y, self.batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                if mu > 0.0:
                    prox = sum(
                        ((lp - gp) ** 2).sum()
                        for lp, gp in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = _predict(self.model, self.X_base, self.device)
        return (0.0, len(self.X_base), {
            "accuracy": float(accuracy_score(self.y_base, preds)),
        })


# ---------------------------------------------------------------------------
# FL Strategy 1 — FedAvg (baseline, mu=0)
# ---------------------------------------------------------------------------

class FedAvgStrategy(FedAvg):
    """Standard FedAvg (mu=0 = pure FedAvg, no FedProx constraint)."""

    def __init__(self, discovery_client_id, discovery_round,
                 proximal_mu=0.0, **kwargs):
        super().__init__(**kwargs)
        self.discovery_client_id = int(discovery_client_id)
        self.discovery_round     = int(discovery_round)
        self.proximal_mu         = float(proximal_mu)

    def configure_fit(self, server_round, parameters, client_manager):
        base             = super().configure_fit(server_round, parameters, client_manager)
        discovery_active = server_round > self.discovery_round
        return [
            (cp, FitIns(fi.parameters, {
                "proximal_mu":      self.proximal_mu,
                "discovery_active": discovery_active,
            }))
            for cp, fi in base
        ]


# ---------------------------------------------------------------------------
# FL Strategy 2 — FedDiv (Divergence-Weighted FedAvg)
# ---------------------------------------------------------------------------

class FedDivStrategy(FedAvg):
    """
    Power-Weighted Divergence FedAvg (FedDiv).

    Each round client i's effective contribution is proportional to:

        ||w_i_local − w_global||₂^p

    With p=2 (default) the most divergent client dominates aggregation
    far more aggressively than with linear (p=1) weighting.  This maximises
    the influence of the node that first encounters the new attack pattern.
    """

    def __init__(self, discovery_client_id, discovery_round,
                 proximal_mu=0.0, div_power=2, **kwargs):
        super().__init__(**kwargs)
        self.discovery_client_id = int(discovery_client_id)
        self.discovery_round     = int(discovery_round)
        self.proximal_mu         = float(proximal_mu)
        self.div_power           = float(div_power)
        init_params = kwargs.get("initial_parameters")
        self._global_ndarrays = (parameters_to_ndarrays(init_params)
                                 if init_params is not None else None)

    def configure_fit(self, server_round, parameters, client_manager):
        base             = super().configure_fit(server_round, parameters, client_manager)
        discovery_active = server_round > self.discovery_round
        return [
            (cp, FitIns(fi.parameters, {
                "proximal_mu":      self.proximal_mu,
                "discovery_active": discovery_active,
            }))
            for cp, fi in base
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if self._global_ndarrays is None:
            agg = super().aggregate_fit(server_round, results, failures)
            if agg[0] is not None:
                self._global_ndarrays = parameters_to_ndarrays(agg[0])
            return agg

        # Compute L2 divergence of each client from the current global model
        divergences = []
        for _, fit_res in results:
            local_nds = parameters_to_ndarrays(fit_res.parameters)
            div = float(np.sqrt(sum(
                np.sum((lp.astype(np.float64) - gp.astype(np.float64)) ** 2)
                for lp, gp in zip(local_nds, self._global_ndarrays)
            )))
            divergences.append(max(div, 1e-8))

        # Raise each divergence to the configured power (default p=2)
        powered = [d ** self.div_power for d in divergences]
        total_powered = sum(powered)
        total_n       = sum(r.num_examples for _, r in results)

        # Re-weight: each client's share ∝ divergence^p
        boosted = []
        for (proxy, fit_res), pw in zip(results, powered):
            effective_n = max(int(round((pw / total_powered) * total_n)), 1)
            boosted.append((proxy, dc_replace(fit_res, num_examples=effective_n)))

        agg = super().aggregate_fit(server_round, boosted, failures)
        if agg[0] is not None:
            self._global_ndarrays = parameters_to_ndarrays(agg[0])
        return agg


# ---------------------------------------------------------------------------
# Single strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(strategy_name, strategy, client_fn, n_clients, n_rounds,
                  X_eval, y_eval, input_dim, num_classes,
                  class_names, target_idx, device):
    """Run one Flower simulation and return (per_round_metrics, final_ndarrays)."""

    per_round_metrics = []
    _last_params      = [None]

    def evaluate_fn(server_round, parameters, config):
        # Flower passes NDArrays (list) directly to evaluate_fn — no conversion needed
        ndarrays = parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        model    = _params_to_model(ndarrays, input_dim, num_classes, device)
        phase    = ("pre_discovery"
                    if server_round <= strategy.discovery_round
                    else "post_discovery")
        m = _eval_on_server(model, X_eval, y_eval, device,
                             num_classes, class_names, target_idx)
        m.update({
            "round":                  server_round,
            "phase":                  phase,
            "discovery_active":       server_round > strategy.discovery_round,
            "rounds_since_discovery": max(0, server_round - strategy.discovery_round),
        })
        per_round_metrics.append(m)
        _last_params[0] = ndarrays
        log.info("  [%s] R%02d [%s]  target_F1=%.4f  macro_F1=%.4f  acc=%.4f",
                 strategy_name, server_round, phase,
                 m["target_f1"], m["macro_f1"], m["accuracy"])
        return 0.0, {"target_f1": m["target_f1"], "macro_f1": m["macro_f1"]}

    strategy.evaluate_fn = evaluate_fn

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        ray_init_args={"ignore_reinit_error": True},
    )

    return per_round_metrics, _last_params[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y, le = load_and_preprocess(DATA_PATH, sample_frac=args.sample_frac)
    class_names = list(le.classes_)
    num_classes  = len(class_names)
    input_dim    = X.shape[1]
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients_base, (X_pool, y_pool), (X_eval, y_eval), target_idx = \
        make_iid_partitions(
            X, y, le,
            target_attack=args.target_attack,
            n_clients=args.n_clients,
            server_frac=args.server_frac,
            discovery_frac=args.discovery_frac,
            seed=42,
        )
    target_name = class_names[target_idx]

    log.info(
        "Flower %s | Device: %s | Clients: %d | Rounds: %d | "
        "Local epochs: %d | mu=%.4f | Discovery: client %d at round %d",
        fl.__version__, device, args.n_clients, args.rounds,
        args.local_epochs, args.mu,
        args.discovery_client, args.discovery_round,
    )

    init_ndarrays   = [v.cpu().numpy() for _, v in
                       build_model(input_dim, num_classes).state_dict().items()]
    init_parameters = ndarrays_to_parameters(init_ndarrays)

    def client_fn(cid):
        cid_int = int(cid)
        Xb, yb  = clients_base[cid_int]
        xp = X_pool if cid_int == args.discovery_client else None
        yp = y_pool if cid_int == args.discovery_client else None
        return DiscoveryClient(
            cid_int, Xb, yb, xp, yp,
            num_classes, input_dim,
            args.local_epochs, args.batch_size, args.lr,
        ).to_client()

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.n_clients,
        min_available_clients=args.n_clients,
        initial_parameters=init_parameters,
    )

    # ─── Run 1: FedAvg ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STRATEGY 1/2 : FedAvg  (mu=%.4f)", args.mu)
    log.info("=" * 70)

    fedavg_strategy = FedAvgStrategy(
        discovery_client_id=args.discovery_client,
        discovery_round=args.discovery_round,
        proximal_mu=args.mu,
        **common_kwargs,
    )
    fedavg_metrics, fedavg_params = _run_strategy(
        "FedAvg", fedavg_strategy, client_fn,
        args.n_clients, args.rounds,
        X_eval, y_eval, input_dim, num_classes,
        class_names, target_idx, device,
    )

    # ─── Run 2: FedDiv ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STRATEGY 2/2 : FedDiv  (divergence^%.0f weighted, mu=%.4f)",
             args.div_power, args.mu)
    log.info("=" * 70)

    feddiv_strategy = FedDivStrategy(
        discovery_client_id=args.discovery_client,
        discovery_round=args.discovery_round,
        proximal_mu=args.mu,
        div_power=args.div_power,
        **common_kwargs,
    )
    feddiv_metrics, feddiv_params = _run_strategy(
        "FedDiv", feddiv_strategy, client_fn,
        args.n_clients, args.rounds,
        X_eval, y_eval, input_dim, num_classes,
        class_names, target_idx, device,
    )

    # ─── Final reports ────────────────────────────────────────────────────
    for name, params in [("FedAvg", fedavg_params), ("FedDiv", feddiv_params)]:
        model = _params_to_model(params, input_dim, num_classes, device)
        preds = _predict(model, X_eval, device)
        log.info("\n=== FINAL GLOBAL MODEL — %s (server eval set) ===", name)
        print(classification_report(y_eval, preds,
                                     target_names=class_names, zero_division=0))
        torch.save(model.state_dict(),
                   RESULTS_DIR / f"new_attack_{name.lower()}_model.pt")

    results = {
        "config": {
            "target_attack":    target_name,
            "discovery_round":  args.discovery_round,
            "discovery_client": args.discovery_client,
            "discovery_frac":   args.discovery_frac,
            "n_clients":        args.n_clients,
            "rounds":           args.rounds,
            "local_epochs":     args.local_epochs,
            "mu":               args.mu,
            "div_power":        args.div_power,
            "sample_frac":      args.sample_frac,
            "server_frac":      args.server_frac,
        },
        "class_names": class_names,
        "framework":   f"flwr {fl.__version__}",
        "fedavg": {
            "per_round_metrics": fedavg_metrics,
            "final_metrics":     fedavg_metrics[-1],
        },
        "feddiv": {
            "per_round_metrics": feddiv_metrics,
            "final_metrics":     feddiv_metrics[-1],
        },
    }

    out = RESULTS_DIR / "new_attack_results.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Results -> %s", out)

    _generate_figures(results)
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _generate_figures(results):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg         = results["config"]
    target_name = cfg["target_attack"]
    disc_rnd    = cfg["discovery_round"]
    div_power   = cfg.get("div_power", 2)
    fd_label    = f"FedDiv (p={div_power:.0f})"
    class_names = results["class_names"]

    fa = results["fedavg"]["per_round_metrics"]
    fd = results["feddiv"]["per_round_metrics"]

    rounds    = [m["round"]     for m in fa]
    fa_tgt    = [m["target_f1"] for m in fa]
    fd_tgt    = [m["target_f1"] for m in fd]
    fa_macro  = [m["macro_f1"]  for m in fa]
    fd_macro  = [m["macro_f1"]  for m in fd]
    fa_acc    = [m["accuracy"]  for m in fa]
    fd_acc    = [m["accuracy"]  for m in fd]

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
    })

    # Figure 06 — Target-attack F1 + global metrics, FedAvg vs FedDiv
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    ax = axes[0]
    ax.plot(rounds, fa_tgt, "o-", color="#2196F3", lw=2.2, ms=5,
            label="FedAvg — target F1")
    ax.plot(rounds, fd_tgt, "s-", color="#F44336", lw=2.2, ms=5,
            label=f"{fd_label} — target F1")
    ax.axvline(disc_rnd + 0.5, color="#212121", lw=1.8, ls="--", zorder=5)
    y_ann = max(max(fa_tgt), max(fd_tgt)) * 0.55
    ax.annotate(
        f"<- Discovery event\n  Client {cfg['discovery_client']} gets\n"
        f"  {target_name} pool",
        xy=(disc_rnd + 0.5, y_ann),
        xytext=(disc_rnd + 1.2, max(y_ann - 0.15, 0.05)),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#212121"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    ax.axvspan(0.5, disc_rnd + 0.5, alpha=0.06, color="#F44336",
               label="Pre-discovery")
    ax.axvspan(disc_rnd + 0.5, max(rounds) + 0.5, alpha=0.06, color="#4CAF50",
               label="Post-discovery")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(
        f"FL Attack Discovery: FedAvg vs FedDiv  |  target='{target_name}'\n"
        f"{cfg['n_clients']} clients  |  mu={cfg['mu']}  |  "
        f"discovery_frac={cfg['discovery_frac']}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")

    ax2 = axes[1]
    ax2.plot(rounds, fa_macro, "o-",  color="#2196F3", lw=1.8, ms=4,
             label="FedAvg Macro F1")
    ax2.plot(rounds, fd_macro, "s-",  color="#F44336", lw=1.8, ms=4,
             label=f"{fd_label} Macro F1")
    ax2.plot(rounds, fa_acc,   "--",  color="#2196F3", lw=1.2, alpha=0.6,
             label="FedAvg Accuracy")
    ax2.plot(rounds, fd_acc,   "--",  color="#F44336", lw=1.2, alpha=0.6,
             label=f"{fd_label} Accuracy")
    ax2.axvline(disc_rnd + 0.5, color="#212121", lw=1.8, ls="--")
    ax2.axvspan(0.5, disc_rnd + 0.5, alpha=0.06, color="#F44336")
    ax2.axvspan(disc_rnd + 0.5, max(rounds) + 0.5, alpha=0.06, color="#4CAF50")
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("FL Communication Round", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    _savefig("06_zeroday_comparison.png")

    # Figure 07 — Per-class F1 at round 20, FedAvg vs FedDiv
    fa_final = [fa[-1]["per_class_f1"].get(cn, 0.0) for cn in class_names]
    fd_final = [fd[-1]["per_class_f1"].get(cn, 0.0) for cn in class_names]

    order        = sorted(range(len(class_names)),
                          key=lambda i: fd_final[i], reverse=True)
    sorted_names = [class_names[i] for i in order]
    fa_sorted    = [fa_final[i] for i in order]
    fd_sorted    = [fd_final[i] for i in order]
    tgt_row      = sorted_names.index(target_name)

    y_pos = np.arange(len(sorted_names))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(13, max(6, len(class_names) * 0.6)))
    ax.barh(y_pos - bar_h / 2, fa_sorted, bar_h,
            color="#2196F350", edgecolor="#2196F3", lw=1.2,
            label="FedAvg (round 20)", zorder=3)
    ax.barh(y_pos + bar_h / 2, fd_sorted, bar_h,
            color="#F4433650", edgecolor="#F44336", lw=1.2,
            label=f"{fd_label} (round 20)", zorder=3)

    ax.axhspan(tgt_row - 0.5, tgt_row + 0.5, alpha=0.10,
               color="#FF9800", zorder=0)
    ax.text(1.02, tgt_row, "<- DISCOVERY TARGET",
            va="center", ha="left", fontsize=8, color="#E65100",
            fontweight="bold", transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score (server evaluation set)", fontsize=11)
    ax.set_title(
        f"Per-Class F1 at Round {max(rounds)} — FedAvg vs {fd_label}\n"
        f"Target: '{target_name}' | discovery at round {disc_rnd}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    _savefig("07_per_class_final.png")


def _savefig(name):
    import matplotlib.pyplot as plt
    path = FIGURES_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved figure -> %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="FL incremental attack discovery: FedAvg vs FedDiv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--target-attack",    type=str,   default="SQL_injection",
                   help="Attack class the discovering client encounters post-event.")
    p.add_argument("--discovery-round",  type=int,   default=5)
    p.add_argument("--discovery-client", type=int,   default=0)
    p.add_argument("--discovery-frac",   type=float, default=0.80,
                   help="Fraction of target-attack samples held as discovery pool.")
    p.add_argument("--rounds",           type=int,   default=20)
    p.add_argument("--local-epochs",     type=int,   default=2)
    p.add_argument("--n-clients",        type=int,   default=5)
    p.add_argument("--batch-size",       type=int,   default=512)
    p.add_argument("--lr",               type=float, default=5e-4)
    p.add_argument("--mu",               type=float, default=0.0,
                   help="FedProx mu (0.0 = disabled, full client divergence).")
    p.add_argument("--div-power",        type=float, default=2.0,
                   help="Exponent for FedDiv divergence weighting (1=linear, "
                        "2=quadratic default, higher = winner-takes-most).")
    p.add_argument("--sample-frac",      type=float, default=0.3)
    p.add_argument("--server-frac",      type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
