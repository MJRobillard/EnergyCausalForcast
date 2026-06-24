"""
Load CEC 2025 hourly forecast scenarios and extract annual-peak rows.

Exposes:
    load_scenario(path, name)   -> hourly DataFrame with parsed 'period' column
    build_peak_tables(scenarios) -> dict[scenario_name, annual-peak DataFrame]
    load_all_scenarios()         -> (hourly_dict, peak_dict) for all configured scenarios
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

from pipeline.config import SCENARIO_FILES

COMP_COLS = [
    "MANAGED_NET_LOAD",
    "UNADJUSTED_CONSUMPTION",
    "DATA_CENTER",
    "AAFS",
    "LIGHT_EV",
    "AATE_LDV",
    "MEDIUM_HEAVY_EV",
    "BTM_PV",
    "AAEE",
]


def load_scenario(path: str | Path, name: str) -> pd.DataFrame:
    """
    Read a CEC hourly Excel file and return a tidy DataFrame.

    The CEC format has columns YEAR, MONTH, DAY, HOUR (1-based), plus
    load component columns. We parse a proper datetime 'period' column
    and attach the scenario name.
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Data", header=0)
    df["period"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"], hour=df["HOUR"] - 1)
    )
    df["scenario"] = name

    # Ensure all expected component columns exist (older files may lack some)
    for col in COMP_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


def build_peak_tables(
    scenarios: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    For each scenario, extract the single annual-peak-hour row per year.

    Peak is defined as the hour with maximum MANAGED_NET_LOAD in each
    calendar year.  Returns dict of DataFrames indexed by YEAR with
    the COMP_COLS as columns.
    """
    peak_tables: dict[str, pd.DataFrame] = {}
    for name, df in scenarios.items():
        idx = df.groupby("YEAR")["MANAGED_NET_LOAD"].idxmax()
        keep_cols = ["YEAR"] + [c for c in COMP_COLS if c in df.columns]
        peak_tables[name] = df.loc[idx, keep_cols].set_index("YEAR")
    return peak_tables


def load_all_scenarios(
    scenario_files: dict[str, Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load every configured CEC scenario.

    Returns
    -------
    hourly_dict : dict[name -> hourly DataFrame]
    peak_dict   : dict[name -> annual-peak DataFrame]
    """
    files = scenario_files or SCENARIO_FILES
    hourly: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        hourly[name] = load_scenario(path, name)

    peaks = build_peak_tables(hourly)
    return hourly, peaks
