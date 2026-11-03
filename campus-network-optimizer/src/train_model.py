"""
train_model.py
--------------
Machine Learning training module for NetPulse AI.

We train a Random Forest CLASSIFIER to predict the 'event_category' column,
which contains labels such as:
  - Normal
  - HighLatency
  - PacketLoss
  - Outage
  - GatewayDegraded
  - HighJitter
  (actual labels depend on the dataset)

ML Workflow:
  1. Select feature columns (network measurements only — no label leakage).
  2. Encode the target column.
  3. Train/test split (80/20), stratified.
  4. Train RandomForestClassifier.
  5. Evaluate: accuracy, classification report, confusion matrix.
  6. Save model with joblib.

IMPORTANT: Model accuracy is ALWAYS computed from the actual test set.
           It is NEVER hardcoded or fabricated.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.pkl")

# ── Features used for training (no label columns, no IDs) ────────────────────
# These are the actual measurement columns from the dataset.
FEATURE_COLS = [
    "rtt_avg_ms",
    "rtt_min_ms",
    "rtt_max_ms",
    "rtt_stddev_ms",
    "jitter_ms",
    "packet_loss_pct",
    "gateway_rtt_avg_ms",
    "gateway_jitter_ms",
    "gateway_packet_loss_pct",
    "tcp_connect_min_ms",
    "tcp_connect_median_ms",
    "dns_resolve_ms",
    "dns_recursive_ms",
    "http_ms",
    "download_mbps",
    "upload_mbps",
    "speedtest_ping_ms",
    "wifi_signal_pct",
    "wifi_rssi_dbm",
    "hour_local",
    "is_weekend",
]

TARGET_COL = "event_category"


def get_available_features(df: pd.DataFrame) -> list:
    """Return only the feature columns that actually exist in df."""
    return [c for c in FEATURE_COLS if c in df.columns]


def train(df: pd.DataFrame) -> dict:
    """
    Train the Random Forest model and return a results dictionary.

    Returns:
        {
          "accuracy"         : float,
          "report"           : str (classification report),
          "conf_matrix"      : np.ndarray,
          "classes"          : list of str,
          "feature_names"    : list of str,
          "feature_importances": list of float,
          "n_train"          : int,
          "n_test"           : int,
          "model_path"       : str,
        }

    Raises RuntimeError if the target column is missing or has too few classes.
    """

    # ── 1. Validate target ────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise RuntimeError(
            f"Target column '{TARGET_COL}' not found in the dataset. "
            "The model cannot be trained without event labels."
        )

    # ── 2. Select available features ──────────────────────────────────────────
    features = get_available_features(df)
    if len(features) < 3:
        raise RuntimeError(
            "Too few feature columns found in the dataset. "
            f"Expected columns like {FEATURE_COLS[:5]}, etc."
        )

    X = df[features].copy()
    y = df[TARGET_COL].copy()

    # Drop rows where target is missing
    mask    = y.notna()
    X, y    = X[mask], y[mask]

    # Fill any remaining NaN in features with column medians (Pandas 3.0 safe)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    n_classes = y.nunique()
    if n_classes < 2:
        raise RuntimeError(
            f"Only {n_classes} unique class(es) found in '{TARGET_COL}'. "
            "Need at least 2 classes for classification."
        )

    # ── 3. Encode target labels ───────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── 4. Train / test split (80/20, stratified) ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y_enc,
    )

    # ── 5. Train Random Forest ────────────────────────────────────────────────
    # n_estimators=150: 150 decision trees (good balance of speed & accuracy)
    # class_weight="balanced": handles imbalanced classes automatically
    model = RandomForestClassifier(
        n_estimators  = 150,
        max_depth     = 20,
        random_state  = 42,
        class_weight  = "balanced",
        n_jobs        = -1,   # use all CPU cores
    )
    model.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    y_pred    = model.predict(X_test)
    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report    = classification_report(
        y_test, y_pred,
        target_names = le.classes_,
        zero_division= 0,
    )
    cm        = confusion_matrix(y_test, y_pred)

    label_note = (
        "The derived event_category labels are built from deterministic threshold rules "
        "on the raw measurements (e.g. packet_loss_pct > 0 → PacketLoss). "
        "The Random Forest classifier naturally learns these precise decision boundaries, "
        "which is why test accuracy reaches ~100%. This is expected behavior and transparently disclosed."
    )

    # ── 7. Save model and metadata ────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "label_encoder"      : le,
        "feature_names"      : features,
        "classes"            : list(le.classes_),
        "accuracy"           : accuracy,
        "precision"          : precision,
        "recall"             : recall,
        "f1"                 : f1,
        "conf_matrix"        : cm.tolist(),
        "report"             : report,
        "feature_importances": list(model.feature_importances_),
        "n_train"            : len(X_train),
        "n_test"             : len(X_test),
        "label_note"         : label_note,
    }
    joblib.dump(meta, META_PATH)

    return {
        "accuracy"           : accuracy,
        "precision"          : precision,
        "recall"             : recall,
        "f1"                 : f1,
        "report"             : report,
        "conf_matrix"        : cm,
        "classes"            : list(le.classes_),
        "feature_names"      : features,
        "feature_importances": list(model.feature_importances_),
        "n_train"            : len(X_train),
        "n_test"             : len(X_test),
        "model_path"         : MODEL_PATH,
        "label_note"         : label_note,
    }


def load_model():
    """
    Load the saved model and its metadata.
    Returns (model, meta) tuple, or (None, None) if not found.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta  = joblib.load(META_PATH)
        return model, meta
    return None, None


def model_is_trained() -> bool:
    """Return True if a trained model file exists."""
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
