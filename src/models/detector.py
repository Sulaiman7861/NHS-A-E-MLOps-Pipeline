import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.features.builder import FEATURE_COLS


def _get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def detect_zscore(
    df: pd.DataFrame,
    threshold: float = 2.5,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Flag anomalies where any feature's z-score exceeds the threshold.

    If reference_df is provided, the scaler is fitted on the reference
    (pre-COVID baseline) and applied to df. This means COVID-era rows
    are scored against what was normal before COVID.
    """
    result = df.copy()
    features = _get_features(df)
    train = reference_df if reference_df is not None else df

    scaler = StandardScaler()
    scaler.fit(train[features].values)
    z = scaler.transform(df[features].values)

    result["zscore_max"] = np.abs(z).max(axis=1)
    result["anomaly"] = result["zscore_max"] > threshold
    result["method"] = "zscore"
    return result


def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Flag anomalies using IsolationForest.

    If reference_df is provided, the model is fitted on the reference
    (pre-COVID baseline) and used to score all of df. This way the model
    learns what "normal" looks like and flags anything that deviates —
    including the COVID period.
    """
    result = df.copy()
    features = _get_features(df)
    train = reference_df if reference_df is not None else df

    scaler = StandardScaler()
    scaler.fit(train[features].values)
    X_train = scaler.transform(train[features].values)
    X_all   = scaler.transform(df[features].values)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_train)

    result["if_score"] = model.decision_function(X_all)
    result["anomaly"]  = model.predict(X_all) == -1
    result["method"]   = "isolation_forest"
    return result


def detect_autoencoder(
    df: pd.DataFrame,
    threshold_percentile: float = 95,
    random_state: int = 42,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Flag anomalies using MLP autoencoder reconstruction error.

    If reference_df is provided, the autoencoder is trained only on
    pre-COVID data so it learns to reconstruct the normal pattern.
    Anything it cannot reconstruct well (high MSE) is flagged —
    this naturally catches COVID-era drift without being told about it.
    """
    result = df.copy()
    features = _get_features(df)
    train = reference_df if reference_df is not None else df

    scaler = StandardScaler()
    scaler.fit(train[features].values)
    X_train = scaler.transform(train[features].values)
    X_all   = scaler.transform(df[features].values)

    hidden = max(2, len(features) // 2)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        max_iter=500,
        random_state=random_state,
    )
    autoencoder.fit(X_train, X_train)

    X_reconstructed = autoencoder.predict(X_all)
    mse = np.mean((X_all - X_reconstructed) ** 2, axis=1)

    # Threshold set from reference reconstruction error, not full dataset
    ref_reconstructed = autoencoder.predict(X_train)
    ref_mse = np.mean((X_train - ref_reconstructed) ** 2, axis=1)
    threshold = np.percentile(ref_mse, threshold_percentile)

    result["reconstruction_error"] = mse
    result["anomaly"] = mse > threshold
    result["method"]  = "autoencoder"
    return result


def run_all(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run all three detectors and concatenate results."""
    return pd.concat([
        detect_zscore(df, reference_df=reference_df),
        detect_isolation_forest(df, reference_df=reference_df),
        detect_autoencoder(df, reference_df=reference_df),
    ], ignore_index=True)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare feature distributions between reference and current period
    using the Kolmogorov-Smirnov test.

    Returns a DataFrame with columns:
        feature, ks_statistic, p_value, drifted
    A feature is considered drifted if p_value < 0.05.
    """
    features = _get_features(reference_df)
    rows = []
    for feat in features:
        ref_vals  = reference_df[feat].dropna().values
        curr_vals = current_df[feat].dropna().values
        if len(ref_vals) < 2 or len(curr_vals) < 2:
            continue
        ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
        rows.append({
            "feature":      feat,
            "ks_statistic": round(ks_stat, 4),
            "p_value":      round(p_val, 4),
            "drifted":      p_val < 0.05,
        })
    return pd.DataFrame(rows)
