"""
Peak event causal analysis pipeline.

Design philosophy: train the SCM on "normal" demand only (peaks excised),
then treat each peak event as a case study with a clean counterfactual
baseline from the excised model.

Responsibilities:
  - Load and merge multi-year CAISO load + regional weather data
  - Build heat trajectory features (rolling regional maxima, wet bulb,
    marine layer flag, consecutive hot days)
  - Identify and cluster spike hours into named peak events
  - Excise peaks from the training set without distorting normal-hour fits
  - Build synthetic control baselines for each event

Data sources expected in DATA_DIR / 'california':
  ca_load.csv                    — hourly CAISO demand, columns: period, demand_mwh
  wx_<region>.parquet            — hourly weather per region (see REGIONS)
  ca_merged.csv                  — pre-merged composite (used as fallback)

To extend historical coverage, add earlier ca_load.csv rows and matching
wx_<region>.parquet files. The pipeline merges on period automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "california"

REGIONS = ["bay_area", "fresno", "los_angeles", "riverside", "sacramento", "san_diego"]

# Inland regions used for marine layer detection (coast vs inland spread)
INLAND_REGIONS = ["riverside", "fresno", "sacramento"]
COASTAL_REGIONS = ["bay_area", "los_angeles", "san_diego"]

US_HOLIDAYS = {
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28",
    "2018-07-04", "2018-09-03", "2018-11-12", "2018-11-22", "2018-12-25",
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-05-27",
    "2019-07-04", "2019-09-02", "2019-11-11", "2019-11-28", "2019-12-25",
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-05-25",
    "2020-07-04", "2020-09-07", "2020-11-11", "2020-11-26", "2020-12-25",
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-05-31",
    "2021-07-05", "2021-09-06", "2021-11-11", "2021-11-25", "2021-12-24",
    "2022-01-17", "2022-02-21", "2022-05-30",
    "2022-06-19", "2022-07-04", "2022-09-05", "2022-11-11",
    "2022-11-24", "2022-12-26",
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

def load_extended(data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Load all available CAISO load + regional weather data and merge.

    Tries to build from ca_load.csv + wx_<region>.parquet files first.
    Falls back to ca_merged.csv if regional parquets are unavailable.

    Returns a DataFrame with columns:
        period, demand_mwh, temperature_f, humidity_pct, wind_mph,
        solar_radiation_wm2, temp_max,
        temp_<region>    — per-region hourly temperature
        hour, month, dow, holiday,
        E_lag24, E_lag168, T_max_lag24, is_summer
    plus all heat trajectory columns added by build_heat_trajectory().
    """
    d = Path(data_dir or DATA_DIR)

    # Load demand — merge historical + current if both exist
    load_parts = []
    for fname in ("ca_load_historical.csv", "ca_load.csv"):
        p = d / fname
        if p.exists():
            part = pd.read_csv(p, parse_dates=["period"])
            part["period"] = part["period"].dt.floor("h")
            load_parts.append(part)
    if not load_parts:
        raise FileNotFoundError(f"No ca_load*.csv found in {d}")
    df_load = (pd.concat(load_parts)
               .drop_duplicates("period")
               .sort_values("period")
               .reset_index(drop=True))

    # Load weather — merge historical + current parquets per region
    region_dfs: dict[str, pd.DataFrame] = {}
    for region in REGIONS:
        parts = []
        for suffix in ("_historical.parquet", ".parquet"):
            path = d / f"wx_{region}{suffix}"
            if path.exists():
                rdf = pd.read_parquet(path)
                rdf["period"] = pd.to_datetime(rdf["period"]).dt.floor("h")
                parts.append(rdf)
        if parts:
            merged = (pd.concat(parts)
                      .drop_duplicates("period")
                      .sort_values("period")
                      .set_index("period"))
            region_dfs[region] = merged

    if not region_dfs:
        # Fallback: use pre-merged composite
        merged_path = d / "ca_merged.csv"
        if not merged_path.exists():
            raise FileNotFoundError("Neither wx_*.parquet nor ca_merged.csv found.")
        df = pd.read_csv(merged_path, parse_dates=["period"])
        df["period"] = df["period"].dt.floor("h")
        df = _add_calendar(df)
        df = _add_lags(df)
        return df.sort_values("period").reset_index(drop=True)

    # Composite weather: load-weighted average across regions
    # (replicates ca_pipeline merge logic; composite columns get no suffix)
    all_wx = pd.concat(
        [rdf.add_suffix(f"_{region}") for region, rdf in region_dfs.items()],
        axis=1,
    )

    # Simple equal-weight composite for composite columns
    temp_cols = [f"temperature_f_{r}" for r in region_dfs]
    hum_cols  = [f"humidity_pct_{r}"  for r in region_dfs]
    wind_cols = [f"wind_mph_{r}"      for r in region_dfs]
    rad_cols  = [f"solar_radiation_wm2_{r}" for r in region_dfs]

    all_wx["temperature_f"]       = all_wx[temp_cols].mean(axis=1)
    all_wx["humidity_pct"]        = all_wx[hum_cols].mean(axis=1)
    all_wx["wind_mph"]            = all_wx[wind_cols].mean(axis=1)
    all_wx["solar_radiation_wm2"] = all_wx[rad_cols].mean(axis=1)
    all_wx["temp_max"]            = all_wx[temp_cols].max(axis=1)

    df = df_load.join(all_wx, on="period", how="inner")

    # Keep individual region temps for trajectory features (re-index to joined df)
    for region in region_dfs:
        col = f"temperature_f_{region}"
        if col in df.columns:
            df[f"temp_{region}"] = df[col]

    df = _add_calendar(df)
    df = _add_lags(df)
    return df.sort_values("period").reset_index(drop=True)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]    = df["period"].dt.hour.astype(float)
    df["month"]   = df["period"].dt.month.astype(float)
    df["dow"]     = df["period"].dt.dayofweek.astype(float)
    date_str      = df["period"].dt.strftime("%Y-%m-%d")
    df["holiday"] = date_str.isin(US_HOLIDAYS).astype(float)
    df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(float)
    return df


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["E_lag24"]     = df["demand_mwh"].shift(24)
    df["E_lag168"]    = df["demand_mwh"].shift(168)
    df["T_max_lag24"] = df["temp_max"].shift(24)
    return df


# ── Heat trajectory features ──────────────────────────────────────────────────

def build_heat_trajectory(df: pd.DataFrame, inland_threshold_f: float = 95.0,
                          marine_spread_f: float = 15.0) -> pd.DataFrame:
    """
    Add multi-day heat trajectory features to a DataFrame.

    New columns:
        wb_temp           — wet bulb temperature (Stull approximation, °F)
        roll_max_48h      — max of temp_max over trailing 48 hours
        roll_max_72h      — max of temp_max over trailing 72 hours
        days_above_95f    — consecutive calendar days where daily temp_max
                            exceeded inland_threshold_f up to each hour
        marine_layer_flag — 1 if inland–coastal temp spread > marine_spread_f
                            (proxy for marine layer collapse)
        heat_index        — NWS simplified heat index (°F), requires T > 80°F
    """
    df = df.copy().sort_values("period").reset_index(drop=True)

    T  = df["temperature_f"]
    RH = df["humidity_pct"]

    # Wet bulb (Stull 2011 approximation, valid for T 5–50°C, RH 5–99%)
    T_c  = (T - 32) * 5 / 9
    wb_c = (T_c * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
            + np.arctan(T_c + RH)
            - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
            - 4.686035)
    df["wb_temp"] = wb_c * 9 / 5 + 32

    # Rolling regional max (trailing, so no lookahead)
    if "temp_max" in df.columns:
        df["roll_max_48h"] = df["temp_max"].rolling(48, min_periods=1).max()
        df["roll_max_72h"] = df["temp_max"].rolling(72, min_periods=1).max()
    else:
        df["roll_max_48h"] = np.nan
        df["roll_max_72h"] = np.nan

    # Consecutive days above threshold — compute per calendar day then forward-fill
    if "temp_max" in df.columns:
        daily_max = (df.groupby(df["period"].dt.date)["temp_max"]
                     .max()
                     .rename("daily_max"))
        daily_max_df = daily_max.reset_index()
        daily_max_df.columns = ["date", "daily_max"]
        daily_max_df["hot_day"] = (daily_max_df["daily_max"] >= inland_threshold_f).astype(int)

        # Count consecutive hot days ending on each date
        consecutive = []
        count = 0
        for hot in daily_max_df["hot_day"]:
            count = count + 1 if hot else 0
            consecutive.append(count)
        daily_max_df["days_above_95f"] = consecutive
        daily_max_df["date"] = pd.to_datetime(daily_max_df["date"])

        df["date"] = df["period"].dt.normalize()
        df = df.merge(daily_max_df[["date", "days_above_95f"]], on="date", how="left")
        df = df.drop(columns=["date"])
    else:
        df["days_above_95f"] = np.nan

    # Marine layer flag: large inland–coastal spread signals marine layer collapse
    inland_cols  = [f"temp_{r}" for r in INLAND_REGIONS  if f"temp_{r}" in df.columns]
    coastal_cols = [f"temp_{r}" for r in COASTAL_REGIONS if f"temp_{r}" in df.columns]
    if inland_cols and coastal_cols:
        df["marine_layer_flag"] = (
            (df[inland_cols].mean(axis=1) - df[coastal_cols].mean(axis=1))
            >= marine_spread_f
        ).astype(float)
    else:
        df["marine_layer_flag"] = np.nan

    # Simplified NWS heat index (valid when T >= 80°F)
    mask = T >= 80
    hi = np.where(
        mask,
        (-42.379 + 2.04901523 * T + 10.14333127 * RH
         - 0.22475541 * T * RH - 6.83783e-3 * T**2
         - 5.481717e-2 * RH**2 + 1.22874e-3 * T**2 * RH
         + 8.5282e-4 * T * RH**2 - 1.99e-6 * T**2 * RH**2),
        T,
    )
    df["heat_index"] = hi

    return df


# ── Peak event identification ─────────────────────────────────────────────────

def identify_peak_events(
    df: pd.DataFrame,
    threshold_pct: float = 99.0,
    gap_hours: int = 24,
    min_duration_hours: int = 2,
) -> pd.DataFrame:
    """
    Cluster spike hours into named peak events.

    Spike hours within gap_hours of each other belong to the same event.
    Events shorter than min_duration_hours are dropped (single-hour anomalies).

    Returns a DataFrame with one row per event:
        event_id, event_name, start, end, duration_hours,
        peak_mw, peak_hour, mean_excess_mw,
        preceding_72h_max_temp, preceding_72h_mean_temp,
        days_above_95f_at_peak, marine_layer_flag_at_peak
    """
    threshold = df["demand_mwh"].quantile(threshold_pct / 100)
    spike_mask = df["demand_mwh"] >= threshold

    # Cluster consecutive (within gap) spike hours into events
    spike_idx = df.index[spike_mask].tolist()
    if not spike_idx:
        return pd.DataFrame()

    clusters: list[list[int]] = []
    current = [spike_idx[0]]
    for idx in spike_idx[1:]:
        if idx - current[-1] <= gap_hours:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    rows = []
    for i, cluster in enumerate(clusters):
        if len(cluster) < min_duration_hours:
            continue
        sub = df.loc[cluster]
        peak_row = sub.loc[sub["demand_mwh"].idxmax()]
        peak_hour = peak_row["period"]

        # Preceding 72h window
        window_start = peak_hour - pd.Timedelta(hours=72)
        preceding = df[(df["period"] >= window_start) & (df["period"] < peak_hour)]

        event_date = peak_hour.date()
        month_name = peak_hour.strftime("%b")
        year       = peak_hour.year
        name       = f"{month_name} {event_date.day} {year}"

        rows.append({
            "event_id":               i,
            "event_name":             name,
            "start":                  sub["period"].min(),
            "end":                    sub["period"].max(),
            "duration_hours":         len(cluster),
            "peak_mw":                peak_row["demand_mwh"],
            "peak_hour":              peak_hour,
            "mean_demand_mw":         sub["demand_mwh"].mean(),
            "preceding_72h_max_temp": preceding["temperature_f"].max() if len(preceding) else np.nan,
            "preceding_72h_mean_temp":preceding["temperature_f"].mean() if len(preceding) else np.nan,
            "preceding_72h_max_temp_max": preceding["temp_max"].max() if "temp_max" in preceding.columns and len(preceding) else np.nan,
            "days_above_95f_at_peak": peak_row.get("days_above_95f", np.nan),
            "marine_layer_at_peak":   peak_row.get("marine_layer_flag", np.nan),
            "wb_temp_at_peak":        peak_row.get("wb_temp", np.nan),
        })

    events = pd.DataFrame(rows).sort_values("peak_mw", ascending=False).reset_index(drop=True)
    return events


def get_event_hours(df: pd.DataFrame, events: pd.DataFrame,
                    padding_hours: int = 6) -> dict[str, pd.DataFrame]:
    """
    Return a dict mapping event_name -> DataFrame of hours for that event
    (including padding_hours before/after for context).
    """
    result = {}
    for _, ev in events.iterrows():
        mask = (
            (df["period"] >= ev["start"] - pd.Timedelta(hours=padding_hours)) &
            (df["period"] <= ev["end"]   + pd.Timedelta(hours=padding_hours))
        )
        result[ev["event_name"]] = df[mask].reset_index(drop=True)
    return result


# ── Peak excision ─────────────────────────────────────────────────────────────

def excise_peaks(
    df: pd.DataFrame,
    threshold_pct: float = 97.0,
    events: Optional[pd.DataFrame] = None,
    event_buffer_hours: int = 6,
) -> pd.DataFrame:
    """
    Remove peak hours from a DataFrame so the SCM trains on normal demand only.

    Two excision modes (applied together):
      1. Quantile threshold: remove any hour above threshold_pct
      2. Event buffer: if events is provided, also remove event_buffer_hours
         before and after each event start/end (prevents edge effects)

    Returns the excised DataFrame and prints a summary.
    """
    mask = df["demand_mwh"] < df["demand_mwh"].quantile(threshold_pct / 100)

    if events is not None and len(events):
        event_mask = pd.Series(False, index=df.index)
        for _, ev in events.iterrows():
            buf = pd.Timedelta(hours=event_buffer_hours)
            event_mask |= (
                (df["period"] >= ev["start"] - buf) &
                (df["period"] <= ev["end"]   + buf)
            )
        mask = mask & ~event_mask

    n_removed = (~mask).sum()
    pct_removed = 100 * n_removed / len(df)
    print(f"Excised {n_removed:,} hours ({pct_removed:.1f}%) — "
          f"{mask.sum():,} hours remain for training.")
    return df[mask].reset_index(drop=True)


def train_test_split_clean(
    df: pd.DataFrame,
    test_start: str = "2025-01-01",
    min_lag_rows: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split excised DataFrame into train and test sets.

    Drops rows where lag features are NaN (first 168 rows after any gap).
    Test set retains all hours including peaks for evaluation.
    """
    lag_ok = df["E_lag24"].notna() & df["E_lag168"].notna()
    df_clean = df[lag_ok].reset_index(drop=True)

    train = df_clean[df_clean["period"] < test_start].reset_index(drop=True)
    test  = df_clean[df_clean["period"] >= test_start].reset_index(drop=True)
    return train, test


# ── Synthetic control baseline ────────────────────────────────────────────────

def synthetic_control(
    event_df: pd.DataFrame,
    donor_pool: pd.DataFrame,
    n_donors: int = 50,
    month_window: int = 1,
    match_cols: tuple[str, ...] = ("roll_max_48h",),
) -> pd.DataFrame:
    """
    Build a synthetic control baseline for a peak event.

    For each hour in event_df, find n_donors "similar normal hours" from
    donor_pool (excised training data) and return their mean as a baseline.

    Matching criteria:
      - Same hour of day
      - Month within ±month_window
      - Same day-of-week
      - Not a holiday
      - Closest match on match_cols (e.g. roll_max_48h quantile bin)

    Returns a DataFrame with the same structure as event_df but with
    demand_mwh and weather columns replaced by the donor mean — i.e.
    "what a normal day with a similar preceding heat trajectory looks like."
    """
    result_rows = []

    for _, hour_row in event_df.iterrows():
        h   = hour_row["hour"]
        m   = hour_row["month"]
        dow = hour_row["dow"]

        candidates = donor_pool[
            (donor_pool["hour"] == h) &
            (donor_pool["month"].between(m - month_window, m + month_window)) &
            (donor_pool["dow"] == dow) &
            (donor_pool["holiday"] == 0)
        ].copy()

        if len(candidates) < 5:
            # Relax dow constraint
            candidates = donor_pool[
                (donor_pool["hour"] == h) &
                (donor_pool["month"].between(m - month_window, m + month_window))
            ].copy()

        # Score candidates by distance on match_cols
        if match_cols and len(candidates) > n_donors:
            dists = np.zeros(len(candidates))
            for col in match_cols:
                if col in candidates.columns and col in hour_row.index:
                    col_range = candidates[col].max() - candidates[col].min()
                    if col_range > 0:
                        dists += ((candidates[col] - hour_row[col]) / col_range) ** 2
            candidates = candidates.iloc[np.argsort(dists)[:n_donors]]

        mean_row = candidates.mean(numeric_only=True)
        mean_row["period"] = hour_row["period"]
        result_rows.append(mean_row)

    baseline = pd.DataFrame(result_rows).reset_index(drop=True)
    baseline["period"] = event_df["period"].values
    return baseline
