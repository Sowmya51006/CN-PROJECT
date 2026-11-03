"""
prediction.py
-------------
This module handles making predictions using the trained Random Forest model.

It accepts user-provided network measurements, formats them correctly,
and returns the predicted event category along with prediction probabilities.

Only real model output is returned.
No accuracy values are hardcoded or fabricated.
"""

import numpy as np
import pandas as pd

from src.train_model import load_model, get_available_features, FEATURE_COLS


def predict_network_condition(input_values: dict) -> dict:
    """
    Predict the network event category from user-provided measurements.

    Parameters:
        input_values (dict): { feature_name: value } for available features.

    Returns:
        {
          "predicted_class"  : str  — e.g. "Normal" or "HighLatency",
          "probabilities"    : dict — { class_name: probability },
          "top_class_prob"   : float,
          "feature_names"    : list[str],
          "input_used"       : dict,
          "error"            : str or None,
        }
    """
    model, meta = load_model()

    if model is None:
        return {
            "predicted_class" : None,
            "probabilities"   : {},
            "top_class_prob"  : 0.0,
            "feature_names"   : [],
            "input_used"      : {},
            "error"           : "Model not trained yet. Please train the model first.",
        }

    feature_names = meta["feature_names"]
    le            = meta["label_encoder"]
    classes       = meta["classes"]

    # Pre-extract primary metrics if provided to compute sensible derived fallbacks
    rtt_avg = float(input_values.get("rtt_avg_ms", 12.0))
    jitter  = float(input_values.get("jitter_ms", 1.0))
    loss    = float(input_values.get("packet_loss_pct", 0.0))
    gw_rtt  = float(input_values.get("gateway_rtt_avg_ms", 0.5))
    dns_ms  = float(input_values.get("dns_resolve_ms", 15.0))
    wifi_sig= float(input_values.get("wifi_signal_pct", 85.0))

    derived_fallbacks = {
        "rtt_min_ms"             : max(1.0, rtt_avg * 0.8),
        "rtt_max_ms"             : max(rtt_avg + 2.0, rtt_avg * 1.35),
        "rtt_stddev_ms"          : max(0.2, jitter * 0.75),
        "gateway_jitter_ms"      : min(5.0, max(0.1, jitter * 0.2)),
        "gateway_packet_loss_pct": max(0.0, loss * 0.2),
        "tcp_connect_min_ms"     : max(1.0, rtt_avg * 0.9),
        "tcp_connect_median_ms"  : max(1.0, rtt_avg * 1.05),
        "dns_recursive_ms"       : max(1.0, dns_ms * 0.8),
        "speedtest_ping_ms"      : rtt_avg,
        "wifi_rssi_dbm"          : -30.0 - (100.0 - wifi_sig) * 0.6,
        "hour_local"             : 14.0,
        "is_weekend"             : 0.0,
    }

    # Build input row — use provided value first, then derived fallback, then 0.0
    row = []
    for col in feature_names:
        if col in input_values:
            val = input_values[col]
        elif col in derived_fallbacks:
            val = derived_fallbacks[col]
        else:
            val = 0.0

        try:
            row.append(float(val))
        except (ValueError, TypeError):
            row.append(0.0)

    X_input = pd.DataFrame([row], columns=feature_names)

    # Make prediction
    pred_enc  = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0]

    predicted_class = le.inverse_transform([pred_enc])[0]
    prob_dict       = {cls: float(p) for cls, p in zip(classes, pred_prob)}
    top_prob        = float(pred_prob.max())

    return {
        "predicted_class" : predicted_class,
        "probabilities"   : prob_dict,
        "top_class_prob"  : top_prob,
        "feature_names"   : feature_names,
        "input_used"      : {k: input_values.get(k, 0.0) for k in feature_names},
        "error"           : None,
    }


def get_default_input_ranges(df: pd.DataFrame) -> dict:
    """
    Compute sensible min/max/default values from the dataset for each
    feature column — used to set slider ranges in the Streamlit UI.

    FIX: Handles constant-value columns (where min == max) by applying
    a safe fallback range so Streamlit never receives min_value == max_value.

    Returns:
        { col_name: {
            "min": float, "max": float, "default": float,
            "constant": bool  <- True if column had only one unique value
          }
        }
    """
    ranges = {}
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        col_vals = df[col].dropna()
        if col_vals.empty:
            continue

        # Use 1st–99th percentile to avoid extreme outliers skewing sliders
        raw_min  = float(col_vals.quantile(0.01))
        raw_max  = float(col_vals.quantile(0.99))
        col_mean = float(col_vals.mean())

        # Only clamp to 0 when the column's NATURAL range is non-negative.
        # Columns like wifi_rssi_dbm (in dBm) are inherently negative — don't clamp.
        natural_min = float(col_vals.min())
        clamp_to_zero = natural_min >= -1.0  # allow small negative near zero
        safe_min = max(0.0, raw_min) if clamp_to_zero else raw_min
        safe_max = raw_max

        # ── Constant-column fix ─────────────────────────────────────────
        # If q01 == q99 the column has no useful range.
        # Apply a ±50% fallback around the mean (or ±0.5 if mean is ~0).
        is_constant = (abs(safe_max - safe_min) < 1e-9)
        if is_constant:
            centre = col_mean if abs(col_mean) > 1e-9 else 1.0
            safe_min = round(centre * 0.5, 4) if not clamp_to_zero else max(0.0, round(centre * 0.5, 4))
            safe_max = round(centre * 1.5, 4)
            # Final guard
            if abs(safe_max - safe_min) < 1e-9:
                safe_min = centre - 0.5
                safe_max = centre + 0.5
                if clamp_to_zero:
                    safe_min = max(0.0, safe_min)

        # Ensure safe_min < safe_max (absolute safety net)
        if safe_min >= safe_max:
            safe_max = safe_min + 1.0

        # Ensure default is within [min, max]
        safe_default = float(np.clip(col_mean, safe_min, safe_max))

        ranges[col] = {
            "min"     : round(safe_min, 4),
            "max"     : round(safe_max, 4),
            "default" : round(safe_default, 4),
            "constant": is_constant,
        }
    return ranges

