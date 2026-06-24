"""
Generic counterfactual engine.

A counterfactual is defined as a set of overrides applied to the
SCMCalibration or to specific fleet columns before computing the trajectory.
Each override is a callable that mutates a copy of the calibration or
transforms a peak-row column value.

CounterfactualSpec   dataclass — defines a single CF experiment
run_counterfactual() — executes the CF and returns a trajectory DataFrame
"""

from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from pipeline.scm.calibration import SCMCalibration
from pipeline.scm.heatwave_uplift import UpliftContext, UpliftMethod, compute_uplift
from pipeline.scm.trajectories import _rows_to_df


@dataclass
class CounterfactualSpec:
    """
    Specification for a single policy counterfactual.

    Parameters
    ----------
    label         : human-readable name for the CF
    description   : one-sentence policy description (used in reports)
    cal_overrides : dict of SCMCalibration field name → new value
                    (e.g. {"beta_ev": 0.0} for perfect EV DR)
    aafs_scale_fn : callable(year) -> float multiplier on the AAFS fleet
    dc_adj_fn     : callable(year, dc_mw) -> adjusted DC grid draw
    ev_adj_fn     : callable(year, lev_mw, ldv_mw) -> (lev_mw, ldv_mw)
    heavy_ev_adj_fn : callable(year, heavy_mw) -> adjusted medium/heavy EV draw
    f_dc_scale    : multiplier on DC thermal factor (PUE / cooling efficiency)
    beta_ev_scale : multiplier on EV heat sensitivity (demand response)
    beta_heavy_ev_scale : multiplier on medium/heavy EV sensitivity
    sector_beta_overrides : dict {sector: new_beta}
                    e.g. {"RES": cal.beta["RES"] * 0.6} for pre-cool
    """
    label:       str
    description: str = ""

    cal_overrides:           dict[str, object]                   = field(default_factory=dict)
    aafs_scale_fn:           Callable[[int], float] | None       = None
    dc_adj_fn:               Callable[[int, float], float] | None = None
    ev_adj_fn:               Callable[[int, float, float], tuple[float, float]] | None = None
    heavy_ev_adj_fn:         Callable[[int, float], float] | None = None
    f_dc_scale:              float = 1.0
    beta_ev_scale:           float = 1.0
    beta_heavy_ev_scale:     float = 1.0
    sector_beta_overrides:   dict[str, float]                    = field(default_factory=dict)


def run_counterfactual(
    spec: CounterfactualSpec,
    peak_tbl: pd.DataFrame,
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    baseline_traj: pd.DataFrame,
    dT_hub: float | None = None,
    uplift_method: UpliftMethod = UpliftMethod.YEAR_NATIVE,
    uplift_ctx: UpliftContext | None = None,
    dynamic_lag: bool = True,
) -> pd.DataFrame:
    """
    Execute a counterfactual and return a trajectory DataFrame with an
    additional 'vs_baseline' column showing the MW delta vs the baseline.

    The function applies overrides in this order:
      1. Scalar SCMCalibration field overrides (cal_overrides)
      2. Sector-specific beta overrides (sector_beta_overrides)
      3. Per-year fleet adjustments (aafs, dc, ev, heavy_ev)
      4. Thermal / sensitivity scalars (f_dc_scale, beta_ev_scale, …)
    """
    pcal = copy.deepcopy(cal)

    if spec.cal_overrides:
        pcal = dataclasses.replace(pcal, **spec.cal_overrides)

    if spec.sector_beta_overrides:
        new_beta = {**pcal.beta, **spec.sector_beta_overrides}
        pcal = dataclasses.replace(pcal, beta=new_beta)

    sector_hvac_mult: dict[str, float] = {}
    for s, new_b in spec.sector_beta_overrides.items():
        old_b = cal.beta.get(s, 0.0)
        sector_hvac_mult[s] = (new_b / old_b) if old_b > 0 else 0.0

    rows = []
    for yr in peak_tbl.index:
        if yr in sector_shares_df.index:
            shares = sector_shares_df.loc[yr].to_dict()
        else:
            nearest = sector_shares_df.index[
                (sector_shares_df.index - yr).abs().argmin()
            ]
            shares = sector_shares_df.loc[nearest].to_dict()

        peak_row = peak_tbl.loc[yr].astype(float).copy()

        if spec.aafs_scale_fn is not None:
            peak_row["AAFS"] = float(peak_row["AAFS"]) * spec.aafs_scale_fn(yr)

        if spec.dc_adj_fn is not None:
            peak_row["DATA_CENTER"] = spec.dc_adj_fn(yr, float(peak_row["DATA_CENTER"]))

        if spec.ev_adj_fn is not None:
            lev, ldv = spec.ev_adj_fn(
                yr, float(peak_row["LIGHT_EV"]), float(peak_row["AATE_LDV"]),
            )
            peak_row["LIGHT_EV"] = lev
            peak_row["AATE_LDV"] = ldv

        if spec.heavy_ev_adj_fn is not None and "MEDIUM_HEAVY_EV" in peak_row.index:
            peak_row["MEDIUM_HEAVY_EV"] = spec.heavy_ev_adj_fn(
                yr, float(peak_row["MEDIUM_HEAVY_EV"]),
            )

        row = compute_uplift(
            yr=int(yr),
            peak_row=peak_row,
            cal=pcal,
            sector_shares_yr=shares,
            method=uplift_method,
            ctx=uplift_ctx,
            scenario=spec.label,
            dT_hub=dT_hub,
            dynamic_lag=dynamic_lag,
            f_dc_scale=spec.f_dc_scale,
            beta_ev_scale=spec.beta_ev_scale,
            beta_heavy_ev_scale=spec.beta_heavy_ev_scale,
            sector_hvac_mult=sector_hvac_mult or None,
        )
        rows.append(row)

    cf_df = _rows_to_df(rows)
    cf_df["stress_total"] = cf_df["mnl_cec"] + cf_df["d_total"]

    if baseline_traj is not None and "d_total" in baseline_traj.columns:
        cf_df["vs_baseline"] = cf_df["d_total"] - baseline_traj["d_total"].reindex(cf_df.index)
        cf_df["stress_vs_baseline"] = (
            cf_df["mnl_hw"] - baseline_traj["mnl_hw"].reindex(cf_df.index)
        )
    else:
        cf_df["vs_baseline"] = float("nan")
        cf_df["stress_vs_baseline"] = float("nan")

    return cf_df
