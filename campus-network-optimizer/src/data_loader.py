"""
data_loader.py
--------------
This module handles loading the LocalNet-Dynamics dataset.
It tries to download it automatically from Hugging Face.
If that fails, it falls back to a local CSV file in the dataset/ folder.

Dataset source: https://huggingface.co/datasets/ezharjan/LocalNetDynamics
NOTE: This is a RESIDENTIAL network dataset (not a real campus network).
      It is used here for educational demonstration purposes only.
"""

import os
import pandas as pd
import streamlit as st

# ── Where we cache the downloaded dataset ──────────────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset")
LOCAL_CSV    = os.path.join(DATASET_DIR, "network_data.csv")

# ── Hugging Face parquet URL (converted parquet endpoint) ──────────────────────
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/ezharjan/LocalNetDynamics/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)


def load_dataset(uploaded_file=None) -> pd.DataFrame:
    """
    Load the dataset in this priority order:
      1. Uploaded file from the user (CSV or Parquet).
      2. Cached local CSV in dataset/network_data.csv.
      3. Auto-download from Hugging Face (parquet).

    Returns a pandas DataFrame, or raises RuntimeError with a helpful message.
    """

    # ── 1. User uploaded a file ────────────────────────────────────────────────
    if uploaded_file is not None:
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            raise RuntimeError(f"Could not read the uploaded file: {e}")

    # ── 2. Cached local CSV ────────────────────────────────────────────────────
    if os.path.exists(LOCAL_CSV):
        try:
            df = pd.read_csv(LOCAL_CSV, low_memory=False)
            return df
        except Exception as e:
            raise RuntimeError(f"Found local CSV but could not read it: {e}")

    # ── 3. Auto-download from Hugging Face ─────────────────────────────────────
    try:
        import requests
        response = requests.get(HF_PARQUET_URL, timeout=60, stream=True)
        if response.status_code == 200:
            import io
            df = pd.read_parquet(io.BytesIO(response.content))
            # Save locally so next load is instant
            os.makedirs(DATASET_DIR, exist_ok=True)
            df.to_csv(LOCAL_CSV, index=False)
            return df
        else:
            raise RuntimeError(
                f"Hugging Face returned HTTP {response.status_code}. "
                "Please download the dataset manually (see sidebar instructions)."
            )
    except ImportError:
        raise RuntimeError("requests library not found. Run: pip install requests")
    except Exception as e:
        raise RuntimeError(
            f"Auto-download failed: {e}\n"
            "Please upload the dataset CSV manually using the sidebar uploader."
        )


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return basic metadata about the loaded DataFrame.
    Useful for the Overview Dashboard.
    """
    return {
        "total_records"    : len(df),
        "total_columns"    : len(df.columns),
        "memory_mb"        : round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
        "duplicate_rows"   : int(df.duplicated().sum()),
        "missing_cells"    : int(df.isnull().sum().sum()),
        "columns"          : list(df.columns),
        "dtypes"           : df.dtypes.astype(str).to_dict(),
    }
