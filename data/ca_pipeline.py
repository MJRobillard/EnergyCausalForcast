"""
Data layer for California CAISO spike analysis.

Responsibilities:
  - Load and standardize the merged CA dataset
  - Add lag and rolling weather features
  - Identify spike hours (top 1%, top 5%, monthly/annual peaks)
  - Build matched normal-hour controls for each spike hour
  - Generate weather-perturbed scenario DataFrames for counterfactual analysis

This layer knows nothing about Pyro or the SCM internals. Adding new
data sources (EV load, BTM solar, CEC modifiers) belongs here.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "california"

US_HOLIDAYS = {
    "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-10",
    "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-11",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-11",
    "2025-11-27", "2025-12-25",
}


# ── Loading ───────────────────────────────────────────────────────────────────

def load(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and standardize the merged CA dataset.

    Adds: hour, month, dow, holiday, E_lag24, E_lag168,
          T_max_lag24 (yesterday's temp_max for heat persistence),
          is_summer (Jun–Sep flag).

    Standard columns guaranteed present in output:
        period, demand_mwh, temperature_f, humidity_pct, wind_mph,
        solar_radiation_wm2, temp_max, hour, month, dow, holiday,
        E_lag24, E_lag168, T_max_lag24, is_summer
    """
    path = path or DATA_DIR / "ca_merged.csv"
    df = pd.read_csv(path, parse_dates=["period"])
    df["period"]  = df["period"].dt.floor("h")
    df["hour"]    = df["period"].dt.hour.astype(float)
    df["month"]   = df["period"].dt.month.astype(float)
    df["dow"]     = df["period"].dt.dayofweek.astype(float)   # 0=Mon … 6=Sun
    date_str      = df["period"].dt.strftime("%Y-%m-%d")
    df["holiday"] = date_str.isin(US_HOLIDAYS).astype(float)
    df["E_lag24"]    = df["demand_mwh"].shift(24)
    df["E_lag168"]   = df["demand_mwh"].shift(168)
    df["T_max_lag24"]= df["temp_max"].shift(24)
    df["is_summer"]  = df["month"].isin([6, 7, 8, 9]).astype(float)
    return df.sort_values("period").reset_index(drop=True)


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: Sep 2023 – Aug 2024.  Test: Sep 2024 – Aug 2025."""
    train = df[(df["period"] < "2024-09-01") & df["E_lag168"].notna()].reset_index(drop=True)
    test  = df[df["period"] >= "2024-09-01"].reset_index(drop=True)
    return train, test


# ── Spike-hour identification ─────────────────────────────────────────────────

def label_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean columns identifying spike hours.

    Columns added:
        is_top1pct    — demand >= 99th percentile of this df
        is_top5pct    — demand >= 95th percentile
        is_monthly_peak — highest demand hour in each calendar month
        is_annual_peak  — single highest demand hour in the df
    """
    df = df.copy()
    df["is_top1pct"]  = df["demand_mwh"] >= df["demand_mwh"].quantile(0.99)
    df["is_top5pct"]  = df["demand_mwh"] >= df["demand_mwh"].quantile(0.95)

    monthly_peak_idx = df.groupby(df["period"].dt.to_period("M"))["demand_mwh"].idxmax()
    df["is_monthly_peak"] = False
    df.loc[monthly_peak_idx.values, "is_monthly_peak"] = True

    df["is_annual_peak"] = df["demand_mwh"] == df["demand_mwh"].max()
    return df


# ── Matched normal-hour controls ──────────────────────────────────────────────

def match_normals(
    df: pd.DataFrame,
    spike_mask: pd.Series,
    month_window: int = 1,
    n_matches: int = 20,
) -> pd.DataFrame:
    """
    For each spike hour, find matched "normal" hours as controls.

    Matching criteria (in order of priority):
        1. Same hour of day
        2. Month within ±month_window
        3. Same day-of-week
        4. Not itself a spike hour
        5. Not a holiday (unless the spike is also a holiday)

    Returns a DataFrame of normal hours (may include duplicates if a
    normal hour matches multiple spikes) with a 'matched_spike_idx'
    column indicating which spike hour it controls for.
    """
    spike_rows   = df[spike_mask].copy()
    normal_pool  = df[~spike_mask].copy()

    matched_rows = []
    for idx, spike in spike_rows.iterrows():
        m   = spike["month"]
        h   = spike["hour"]
        dow = spike["dow"]

        candidates = normal_pool[
            (normal_pool["hour"] == h) &
            (normal_pool["month"].between(m - month_window, m + month_window)) &
            (normal_pool["dow"] == dow)
        ]

        # prefer non-holiday matches when spike is non-holiday
        if spike["holiday"] == 0 and len(candidates[candidates["holiday"] == 0]) >= 5:
            candidates = candidates[candidates["holiday"] == 0]

        if len(candidates) == 0:
            # relax dow constraint
            candidates = normal_pool[
                (normal_pool["hour"] == h) &
                (normal_pool["month"].between(m - month_window, m + month_window))
            ]

        sample = candidates.sample(
            n=min(n_matches, len(candidates)), random_state=42
        )
        sample = sample.copy()
        sample["matched_spike_idx"] = idx
        matched_rows.append(sample)

    if not matched_rows:
        return pd.DataFrame()
    return pd.concat(matched_rows, ignore_index=True)


# ── Weather scenario generation ───────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "+2F":             {"temperature_f": +2.0},
    "+5F":             {"temperature_f": +5.0},
    "+10F":            {"temperature_f": +10.0},
    "+5F_summer":      {"temperature_f": +5.0, "summer_only": True},
    "+5F_inland":      {"temp_max": +5.0, "T_max_lag24": +5.0},
    "+10F_heatwave":   {"temperature_f": +10.0, "temp_max": +10.0,
                        "T_max_lag24": +10.0, "summer_only": True},
    "p90_temp":        {"set_temp_pct": 90},
    "p95_temp":        {"set_temp_pct": 95},
    "normal_weather":  {"set_temp_pct": 50},   # replace temp with median
}


def make_scenario(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """
    Return a copy of df with weather perturbed according to scenario spec.

    Scenario keys:
        temperature_f   : additive shift in °F (optionally summer_only=True)
        temp_max        : additive shift to temp_max
        T_max_lag24     : additive shift to T_max_lag24
        summer_only     : if True, apply shifts only to Jun–Sep rows
        set_temp_pct    : replace temperature_f with the Nth percentile value
                          (e.g. 50 → replace every hour with median temperature)
    """
    out = df.copy()

    if "set_temp_pct" in scenario:
        pct_val = float(np.percentile(df["temperature_f"].dropna(), scenario["set_temp_pct"]))
        out["temperature_f"] = pct_val
        return out

    summer_mask = out["is_summer"].astype(bool) if "summer_only" in scenario and scenario["summer_only"] \
                  else pd.Series(True, index=out.index)

    for col in ("temperature_f", "temp_max", "T_max_lag24"):
        if col in scenario:
            out.loc[summer_mask, col] = out.loc[summer_mask, col] + scenario[col]

    return out
