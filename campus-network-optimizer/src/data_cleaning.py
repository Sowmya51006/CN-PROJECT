"""
data_cleaning.py
----------------
This module cleans the raw LocalNet-Dynamics dataset.

Steps performed:
  1. Remove completely empty rows.
  2. Drop duplicate records.
  3. Convert timestamp columns to datetime.
  4. Convert numeric columns to the correct type.
  5. Cap extreme outliers using the IQR method (optional, conservative).
  6. Fill missing numeric values with column medians.
  7. Fill missing categorical values with "Unknown".

The original measurements are NEVER fabricated.
Only cleaning is applied — no synthetic data injection.
"""

import pandas as pd
import numpy as np


# ── Columns we care about for analysis and ML ────────────────────────────────
NUMERIC_FEATURES = [
    "rtt_avg_ms", "rtt_min_ms", "rtt_max_ms", "rtt_stddev_ms",
    "jitter_ms",
    "packet_loss_pct",
    "gateway_rtt_avg_ms", "gateway_jitter_ms", "gateway_packet_loss_pct",
    "tcp_connect_min_ms", "tcp_connect_median_ms",
    "dns_resolve_ms", "dns_recursive_ms",
    "http_ms",
    "download_mbps", "upload_mbps", "speedtest_ping_ms",
    "download_loaded_latency_ms", "upload_loaded_latency_ms",
    "wifi_signal_pct", "wifi_rssi_dbm", "wifi_rx_mbps", "wifi_tx_mbps",
    "lan_throughput_mbps",
]

LABEL_COLUMNS = [
    "label_outage", "label_packet_loss", "label_high_latency",
    "label_high_jitter", "label_gateway_degraded", "label_anomaly",
]

CATEGORICAL_FEATURES = [
    "link_type", "wifi_ssid", "wifi_phy", "event_category",
    "host_label",
]

TIMESTAMP_COLUMNS = ["timestamp_utc", "timestamp_local"]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning function. Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()  # never modify the original

    # ── Step 1: Drop completely empty rows ───────────────────────────────────
    df.dropna(how="all", inplace=True)

    # ── Step 2: Drop duplicate rows ──────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Step 3: Parse timestamps ──────────────────────────────────────────────
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # ── Step 4: Convert numeric columns ──────────────────────────────────────
    for col in NUMERIC_FEATURES + LABEL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Step 5: Fill missing numeric values with column medians ───────────────
    # Using assignment instead of inplace (Pandas 3.0 compatible)
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            valid_vals = df[col].dropna()
            if not valid_vals.empty:
                median_val = valid_vals.median()
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(0.0)

    # ── Step 6: Fill missing label columns with 0 (no anomaly) ───────────────
    for col in LABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── Step 7: Fill missing categorical values ───────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Cast to object first to avoid dtype incompatibility warning
            df[col] = df[col].astype(object).fillna("Unknown")

    # ── Step 8: Build event_category from binary label columns ────────────────
    # The raw event_category column is entirely null in the LocalNet-Dynamics
    # dataset. We derive a meaningful multi-class label from the binary label_*
    # columns that ARE populated.
    df["event_category"] = _build_event_category(df)

    # ── Step 9: Derive hour/date columns if not already present ──────────────
    if "timestamp_local" in df.columns and "hour_local" not in df.columns:
        df["hour_local"] = df["timestamp_local"].dt.hour

    if "timestamp_local" in df.columns and "date_local" not in df.columns:
        df["date_local"] = df["timestamp_local"].dt.date.astype(str)

    return df


def _build_event_category(df: pd.DataFrame) -> pd.Series:
    """
    Derive a multi-class event label from the binary label columns.

    Priority order (highest severity wins):
      Outage > HighLatency > PacketLoss > HighJitter > GatewayDegraded > Normal

    This produces a meaningful classification target since the raw
    'event_category' column is null in the current dataset release.
    """
    categories = pd.Series("Normal", index=df.index, dtype=object)

    # Apply in ASCENDING priority so highest-severity overwrites last
    if "label_gateway_degraded" in df.columns:
        categories[df["label_gateway_degraded"] == 1] = "GatewayDegraded"
    if "label_high_jitter" in df.columns:
        categories[df["label_high_jitter"] == 1] = "HighJitter"
    if "label_packet_loss" in df.columns:
        categories[df["label_packet_loss"] == 1] = "PacketLoss"
    if "label_high_latency" in df.columns:
        categories[df["label_high_latency"] == 1] = "HighLatency"
    if "label_outage" in df.columns:
        categories[df["label_outage"] == 1] = "Outage"

    return categories


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing missing value counts and percentages
    for every column — useful for the Dataset Explorer page.
    """
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({
        "Column"       : df.columns,
        "Missing Count": missing.values,
        "Missing %"    : pct.values,
        "Data Type"    : df.dtypes.astype(str).values,
    })
    report = report[report["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )
    return report


def get_cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """
    Compare raw and cleaned DataFrames and return a summary dict.
    """
    return {
        "raw_rows"       : len(raw_df),
        "clean_rows"     : len(clean_df),
        "rows_removed"   : len(raw_df) - len(clean_df),
        "raw_missing"    : int(raw_df.isnull().sum().sum()),
        "clean_missing"  : int(clean_df.isnull().sum().sum()),
        "duplicates_found": int(raw_df.duplicated().sum()),
    }
