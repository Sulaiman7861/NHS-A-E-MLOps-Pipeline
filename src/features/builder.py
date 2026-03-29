import re
import pandas as pd
import numpy as np
from pathlib import Path

# Columns used as anomaly detection features
FEATURE_COLS = [
    "total_attendances",
    "breach_rate",
    "admissions_per_attendance",
    "over_4hr_rate",
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _extract_period_from_filename(filename: str) -> pd.Timestamp:
    """Parse period from filenames like 'March-2025-AE-by-provider-xxx.xls'."""
    match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{4})",
        filename,
        re.IGNORECASE,
    )
    if match:
        month = MONTH_MAP[match.group(1).lower()]
        year = int(match.group(2))
        return pd.Timestamp(year=year, month=month, day=1)
    return pd.NaT


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw parquet data and engineer features for anomaly detection.

    Input:  raw combined DataFrame from parser
    Output: DataFrame with FEATURE_COLS + period, org_code, org_name
    """
    df = df.copy()

    # Drop national aggregate row
    df = df[df["org_code"] != "-"].reset_index(drop=True)

    # Extract period from source filename if not present
    if "period" not in df.columns or df["period"].isna().all():
        df["period"] = df["source_file"].apply(_extract_period_from_filename)

    # --- Derived features ---
    df["breach_rate"] = 1 - df["pct_within_4hr"].clip(0, 1)
    df["admissions_per_attendance"] = (
        df["total_admissions"] / df["total_attendances"].replace(0, np.nan)
    )
    df["over_4hr_rate"] = (
        df["over_4hr_dtoa"] / df["total_attendances"].replace(0, np.nan)
    )

    # Fill remaining NaNs in feature columns with column median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    keep_cols = ["period", "org_code", "org_name", "source_file"] + FEATURE_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)
