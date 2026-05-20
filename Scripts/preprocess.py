"""
preprocess.py
=============
Shared data loading, cleaning, encoding, and FL-partitioning utilities.

FL Partitioning Strategy (non-IID)
-----------------------------------
Each FL client simulates an isolated IoT network segment that only observes
normal traffic PLUS the attack types characteristic of its threat domain:

    Client 0  –  DoS / DDoS threats
    Client 1  –  Reconnaissance / Information-Gathering threats
    Client 2  –  Injection threats  (SQL, XSS, File Upload)
    Client 3  –  Man-in-the-Middle / Credential threats
    Client 4  –  Malware threats  (Backdoor, Ransomware)

A client that has NEVER seen, say, an SQL-injection attack locally will still
be able to detect it after federated aggregation – that's the core FL value
proposition this project demonstrates.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle as sk_shuffle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "EdgeIIoTset" / "DNN-EdgeIIoT-dataset.csv"

# ---------------------------------------------------------------------------
# Column lists (from the original Edge-IIoTset preprocessing notebook)
# ---------------------------------------------------------------------------
DROP_COLUMNS: list[str] = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
    "tcp.dstport", "udp.port", "mqtt.msg",
]

CATEGORICAL_COLUMNS: list[str] = [
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
]

TARGET_COLUMN = "Attack_type"

# ---------------------------------------------------------------------------
# Attack-category keyword map (case-insensitive substring matching)
# Used to assign attack types to FL clients by threat domain.
# ---------------------------------------------------------------------------
ATTACK_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "DoS_DDoS":  ["ddos", "dos", "flood"],
    "Recon":     ["port_scan", "os_finger", "vulnerability"],
    "Injection": ["sql", "xss", "upload"],
    "MITM":      ["mitm", "man_in", "password"],
    "Malware":   ["backdoor", "ransomware"],
}
# Ordered list used for client assignment (one category per client)
CATEGORY_ORDER: list[str] = list(ATTACK_CATEGORY_KEYWORDS.keys())


def _categorize(attack_name: str) -> str:
    """Return the threat-category name for a given attack type string."""
    low = attack_name.lower()
    if low == "normal":
        return "Normal"
    for cat, keywords in ATTACK_CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def load_and_preprocess(
    path: Path = DATA_PATH,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load the Edge-IIoTset DNN CSV, clean it, dummy-encode categoricals,
    standardise features, and encode the target.

    Returns
    -------
    X : float32 array, shape (n_samples, n_features)
    y : int64 array,   shape (n_samples,)
    le : fitted LabelEncoder (maps integer label → attack-type name)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run  python Scripts/download_data.py  first."
        )

    log.info("Loading dataset from %s …", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Raw shape: %s", df.shape)

    # Drop irrelevant / high-cardinality columns
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

    # Remove NaN, duplicates, then shuffle
    df.dropna(axis=0, how="any", inplace=True)
    df.drop_duplicates(inplace=True)
    df = sk_shuffle(df, random_state=seed).reset_index(drop=True)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        log.info("Sampled %.0f %% → shape: %s", sample_frac * 100, df.shape)

    # Dummy-encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=np.float32)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Encode target label
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN]).astype(np.int64)

    # Feature matrix
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)

    # Standardise
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    log.info(
        "Preprocessed → X: %s  |  classes (%d): %s",
        X.shape, len(le.classes_), list(le.classes_),
    )
    return X, y, le


# ---------------------------------------------------------------------------
# FL data partitioning
# ---------------------------------------------------------------------------

def make_fl_partitions(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    num_clients: int = 5,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray], list[str]]:
    """
    Create a global held-out test set and per-client non-IID training sets.

    Each client receives:
        • All "Normal" samples from the training split (proportional share).
        • Attack samples belonging exclusively to its assigned threat category.

    This simulates the real-world scenario where network segments experience
    different threat landscapes and cannot share raw traffic data.

    Returns
    -------
    clients_data        : list of (X_train_i, y_train_i) per client
    (X_test, y_test)    : global test set (contains ALL classes)
    client_categories   : list of category name assigned to each client
    """
    class_names: list[str] = list(le.classes_)
    n_classes = len(class_names)

    # Identify normal class index
    normal_idx = next(
        (i for i, c in enumerate(class_names) if c.lower() == "normal"), None
    )
    if normal_idx is None:
        log.warning("No 'Normal' class found; all samples treated as attacks.")

    # ── Global train / test split (stratified) ────────────────────────────
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    X_test, y_test = X[idx_test], y[idx_test]
    X_tr, y_tr = X[idx_train], y[idx_train]

    # ── Map every attack class to a threat category ────────────────────────
    class_to_cat: dict[int, str] = {
        i: _categorize(c) for i, c in enumerate(class_names)
    }

    # Collect unique categories that appear in training data
    present_cats = sorted({
        v for v in class_to_cat.values() if v not in ("Normal", "Other")
    })

    n_clients = min(num_clients, len(present_cats))
    if n_clients < num_clients:
        log.warning(
            "Requested %d clients but only %d attack categories found; "
            "using %d clients.", num_clients, n_clients, n_clients
        )

    client_categories = [CATEGORY_ORDER[i] for i in range(n_clients)
                         if CATEGORY_ORDER[i] in present_cats]
    # Pad with remaining categories if needed
    for cat in present_cats:
        if cat not in client_categories:
            client_categories.append(cat)
    client_categories = client_categories[:n_clients]

    # ── Build per-client datasets ─────────────────────────────────────────
    clients_data: list[tuple[np.ndarray, np.ndarray]] = []
    for i, cat in enumerate(client_categories):
        assigned_class_ids = {
            ci for ci, cc in class_to_cat.items() if cc == cat
        }
        keep_ids = assigned_class_ids | ({normal_idx} if normal_idx is not None else set())
        mask = np.isin(y_tr, list(keep_ids))

        X_c, y_c = X_tr[mask], y_tr[mask]
        clients_data.append((X_c, y_c))

        attack_labels = sorted(class_names[ci] for ci in assigned_class_ids)
        log.info(
            "Client %d [%s] : %6d samples | attacks: %s",
            i, cat, len(X_c), attack_labels,
        )

    return clients_data, (X_test, y_test), client_categories
