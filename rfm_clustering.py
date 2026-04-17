"""
rfm_clustering.py
=================
Core module for Customer Segmentation using RFM methodology + K-Means Clustering.

Author: DA Project — Superstore Analytics
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "raw" / "superstore_raw.csv"

# Segment labels mapped from a combined RFM score quartile approach.
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk Customers",
    3: "Hibernating",
    4: "New Customers",
    5: "Big Spenders",
    6: "Lost",
}

# Display labels with emoji — used inside Streamlit (browser, no encoding issues)
SEGMENT_LABELS_UI = {
    0: "💎 Champions",
    1: "🌟 Loyal Customers",
    2: "⚠️  At-Risk Customers",
    3: "😴 Hibernating",
    4: "🆕 New Customers",
    5: "🤑 Big Spenders",
    6: "📉 Lost",
}

# Hex palette aligned with modern_dark_theme (deep navy + accent gradients)
CLUSTER_PALETTE = [
    "#4FC3F7",  # sky blue
    "#81C784",  # mint green
    "#FFB74D",  # amber
    "#F06292",  # pink
    "#CE93D8",  # lilac
    "#80DEEA",  # cyan
    "#FFCC02",  # gold
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load Superstore raw CSV, handling common encoding issues."""
    df = pd.read_csv(filepath, encoding="latin1", parse_dates=["Order Date", "Ship Date"])
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False)
    return df


# ---------------------------------------------------------------------------
# RFM Feature Engineering
# ---------------------------------------------------------------------------

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary features per Customer ID.

    - Recency  : Days since the last purchase (lower = better).
    - Frequency: Number of unique orders placed.
    - Monetary  : Total Sales revenue generated (USD).
    """
    snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            Customer_Name=("Customer Name", "first"),
            Segment=("Segment", "first"),
            Region=("Region", "first"),
            Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Order ID", "nunique"),
            Monetary=("Sales", "sum"),
        )
        .reset_index()
    )

    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


# ---------------------------------------------------------------------------
# Elbow Method — find optimal K
# ---------------------------------------------------------------------------

def compute_elbow(rfm_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Return inertia (WCSS) and silhouette scores for a range of K values.
    Used to plot the Elbow Curve.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, labels))

    return {
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": silhouettes,
    }


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Scale RFM features and run K-Means. Returns rfm DataFrame enriched with:
      - R_score, F_score, M_score   : Quartile-based 1-4 scores
      - Cluster                     : K-Means cluster label (0-indexed)
      - Cluster_Name                : Human-readable segment label
      - Cluster_Color               : Hex color per cluster
    """
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=15, random_state=42)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(X_scaled)

    # --- Order clusters by Monetary value so label 0 → lowest, N-1 → highest
    cluster_monetary = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_monetary.index)}
    rfm["Cluster"] = rfm["Cluster"].map(rank_map)

    # --- Assign descriptive names (cycling through label dict)
    rfm["Cluster_Name"] = rfm["Cluster"].apply(
        lambda c: SEGMENT_LABELS_UI.get(c, f"Segment {c}")
    )
    rfm["Cluster_Color"] = rfm["Cluster"].apply(
        lambda c: CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
    )

    # --- RFM quartile scores (1=worst, 4=best)
    rfm["R_score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    return rfm, X_scaled, scaler, km


# ---------------------------------------------------------------------------
# Cluster Summary Table
# ---------------------------------------------------------------------------

def cluster_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return a high-level summary table per cluster."""
    summary = (
        rfm.groupby(["Cluster", "Cluster_Name"], sort=True)
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
            Avg_RFM_Score=("RFM_Score", "mean"),
        )
        .reset_index()
    )
    summary["Avg_Recency"] = summary["Avg_Recency"].round(0).astype(int)
    summary["Avg_Frequency"] = summary["Avg_Frequency"].round(1)
    summary["Avg_Monetary"] = summary["Avg_Monetary"].round(0).astype(int)
    summary["Total_Revenue"] = summary["Total_Revenue"].round(0).astype(int)
    summary["Avg_RFM_Score"] = summary["Avg_RFM_Score"].round(1)
    return summary


# ---------------------------------------------------------------------------
# Time-Series Forecasting helper (used in Tab 2)
# ---------------------------------------------------------------------------

def load_monthly_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the pre-aggregated monthly data for time-series forecasting."""
    if filepath is None:
        filepath = Path(__file__).parent / "data" / "processed" / "monthly_aggregation.csv"
    df = pd.read_csv(filepath, parse_dates=["month_year"])
    df = df.sort_values("month_year").reset_index(drop=True)
    return df


def forecast_holt_winters(series: pd.Series, periods: int = 6) -> pd.Series:
    """
    Apply Holt-Winters Exponential Smoothing with additive seasonality.
    Returns a Series of `periods` forecasted values.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(periods)
    return forecast
