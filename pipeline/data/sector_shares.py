"""
Derive Residential / Commercial / Industrial sector shares of total
CAISO electricity consumption and project them forward to 2050.

Data source: CEC AGG_CONSUMPTION_ELEC_COUNTY_TBL_MONTHLY.xlsx
  Columns: YEAR, MONTH, COUNTY_NUM, COUNTY_NAME, SECTOR, RNR, GWH

The raw file covers the whole state; we use it as a proxy for CAISO
territory (which represents ~80% of California load).

Sector mapping
--------------
RES  <- 'Residential'
COM  <- 'Commercial' + 'Streetlighting'
IND  <- 'Industrial' + 'Mining' + 'Agriculture and Water Pumping'
         + 'Transportation, Communications, & Utilities'

Forward projection
------------------
We fit a linear trend to 2015-2024 annual shares and extrapolate to 2050,
capping total drift at SECTOR_SHARE_MAX_DRIFT_PP percentage points from
the 2024 anchor to avoid runaway extrapolation.  Shares are renormalised
to sum to 1.0 for each projection year.

Exposes
-------
load_historical_shares()    -> DataFrame [year × sector] with raw GWh shares
project_sector_shares(years) -> DataFrame [year × {RES,COM,IND}] fractions
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import (
    SECTOR_CONSUMPTION_FILE,
    SECTOR_SHARE_DRIFT_PP_YR,
    SECTOR_SHARE_ANCHOR_YEAR,
    SECTOR_SHARE_MAX_DRIFT_PP,
)

_SECTOR_MAP: dict[str, str] = {
    "Residential":                              "RES",
    "Commercial":                               "COM",
    "Streetlighting":                           "COM",
    "Industrial":                               "IND",
    "Mining":                                   "IND",
    "Agriculture and Water Pumping":            "IND",
    "Transportation, Communications, & Utilities": "IND",
}


def load_historical_shares(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Return annual sector shares as fractions summing to 1.0.

    Returns DataFrame indexed by YEAR with columns [RES, COM, IND].
    """
    path = Path(path or SECTOR_CONSUMPTION_FILE)
    raw = pd.read_excel(path, sheet_name=0)
    raw["sector3"] = raw["SECTOR"].map(_SECTOR_MAP)
    raw = raw.dropna(subset=["sector3"])

    annual = (
        raw.groupby(["YEAR", "sector3"])["GWH"]
        .sum()
        .unstack(fill_value=0.0)
    )
    shares = annual.div(annual.sum(axis=1), axis=0)
    return shares[["RES", "COM", "IND"]]


def project_sector_shares(
    years: list[int] | np.ndarray,
    anchor_shares: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Project sector shares forward for each year in `years`.

    Parameters
    ----------
    years         : list of projection years (e.g. range(2025, 2051))
    anchor_shares : pd.Series with index [RES, COM, IND] at anchor year;
                    if None, loaded from historical data at SECTOR_SHARE_ANCHOR_YEAR

    Returns
    -------
    DataFrame indexed by year, columns [RES, COM, IND], fractions summing to 1.
    """
    if anchor_shares is None:
        hist = load_historical_shares()
        anchor_shares = hist.loc[SECTOR_SHARE_ANCHOR_YEAR]

    rows = []
    for yr in years:
        drift = {
            s: SECTOR_SHARE_DRIFT_PP_YR[s] * (yr - SECTOR_SHARE_ANCHOR_YEAR) / 100.0
            for s in ["RES", "COM", "IND"]
        }
        # Apply drift, capped at max deviation from anchor
        raw_share = {
            s: anchor_shares[s] + np.clip(
                drift[s],
                -SECTOR_SHARE_MAX_DRIFT_PP / 100.0,
                SECTOR_SHARE_MAX_DRIFT_PP / 100.0,
            )
            for s in ["RES", "COM", "IND"]
        }
        total = sum(raw_share.values())
        rows.append({s: raw_share[s] / total for s in ["RES", "COM", "IND"]})

    return pd.DataFrame(rows, index=years, columns=["RES", "COM", "IND"])


def anchor_shares_2025() -> pd.Series:
    """Convenience: return the 2024 anchor shares (used as 2025 calibration)."""
    hist = load_historical_shares()
    return hist.loc[SECTOR_SHARE_ANCHOR_YEAR]
