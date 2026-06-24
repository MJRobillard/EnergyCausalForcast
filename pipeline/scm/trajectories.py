"""
Compute sectoral δMNL trajectories over 2025–2050.

Iterates over years in a peak table, pulls the sector shares for each year,
and calls compute_sector_uplift() to get a SectorUpliftRow per year.

Exposes
-------
compute_trajectory(peak_tbl, cal, sector_shares_df, scenario, dT_hub)
    -> pd.DataFrame of SectorUpliftRow fields, indexed by year
"""

from __future__ import annotations
import pandas as pd

from pipeline.scm.calibration import SCMCalibration
from pipeline.scm.sector_model import SectorUpliftRow, compute_sector_uplift


def compute_trajectory(
    peak_tbl: pd.DataFrame,
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    scenario: str = "",
    dT_hub: float | None = None,
) -> pd.DataFrame:
    """
    Produce a year-indexed DataFrame of sectoral causal uplifts.

    Parameters
    ----------
    peak_tbl         : annual-peak DataFrame indexed by YEAR (from cec_loader)
    cal              : SCMCalibration from calibrate()
    sector_shares_df : DataFrame indexed by year, columns [RES, COM, IND]
    scenario         : scenario name tag
    dT_hub           : hub ΔT to use; if None, inferred from calibration

    Returns
    -------
    DataFrame with columns mirroring SectorUpliftRow fields, indexed by year.
    """
    rows: list[SectorUpliftRow] = []

    for yr in peak_tbl.index:
        peak_row = peak_tbl.loc[yr]

        if yr in sector_shares_df.index:
            shares = sector_shares_df.loc[yr].to_dict()
        else:
            # Fall back to the nearest available projected year
            nearest = sector_shares_df.index[
                (sector_shares_df.index - yr).abs().argmin()
            ]
            shares = sector_shares_df.loc[nearest].to_dict()

        row = compute_sector_uplift(
            yr=yr,
            peak_row=peak_row,
            cal=cal,
            sector_shares_yr=shares,
            scenario=scenario,
            dT_hub=dT_hub,
        )
        rows.append(row)

    df = _rows_to_df(rows)
    return df


def _rows_to_df(rows: list[SectorUpliftRow]) -> pd.DataFrame:
    """Flatten SectorUpliftRow dataclasses to a tidy DataFrame."""
    records = []
    for r in rows:
        records.append({
            "year":           r.year,
            "scenario":       r.scenario,
            "d_hvac_res":     r.d_hvac_res,
            "d_hvac_com":     r.d_hvac_com,
            "d_hvac_ind":     r.d_hvac_ind,
            "d_hvac_total":   r.d_hvac_total,
            "d_aafs_res":     r.d_aafs_res,
            "d_aafs_com":     r.d_aafs_com,
            "d_aafs_total":   r.d_aafs_total,
            "d_ev_res":       r.d_ev_res,
            "d_ev_com":       r.d_ev_com,
            "d_ev_ind":       r.d_ev_ind,
            "d_ev_total":     r.d_ev_total,
            "d_dc":           r.d_dc,
            "d_residual":     r.d_residual,
            "d_total":        r.d_total,
            "d_static":       r.d_static,
            "mnl_cec":        r.mnl_cec,
            "mnl_hw":         r.mnl_hw,
            "vs_static":      r.d_total - r.d_static,
            "capacity_clipped":  r.capacity_clipped,
            "clipped_sectors":   "|".join(r.clipped_sectors) if r.clipped_sectors else "",
            "clip_amount_mw":    r.clip_amount_mw,
        })
    return pd.DataFrame(records).set_index("year")


def compute_all_trajectories(
    peak_tables: dict[str, pd.DataFrame],
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    dT_hub: float | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run compute_trajectory for every scenario in peak_tables.

    Returns dict[scenario_name -> trajectory DataFrame].
    """
    return {
        name: compute_trajectory(
            peak_tbl=tbl,
            cal=cal,
            sector_shares_df=sector_shares_df,
            scenario=name,
            dT_hub=dT_hub,
        )
        for name, tbl in peak_tables.items()
    }
