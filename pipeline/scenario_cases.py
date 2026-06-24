"""
Best / average / worst-case scenario engine for CEC + SCM heatwave analysis.

Sampling modes
──────────────
co_occurring   : all components from the annual MNL peak hour (single row)
split          : dispatch loads (MNL, DC, EV, AAFS) from MNL peak;
                 fleet exposure (UNADJ) from annual max core-load hour
component_max  : each component at its annual maximum (theoretical envelope)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from pipeline.data.cec_loader import COMP_COLS
from pipeline.scm.calibration import SCMCalibration
from pipeline.scm.heatwave_uplift import (
    UpliftContext,
    UpliftMethod,
    build_uplift_context,
    compute_uplift,
)
from pipeline.scm.sector_model import SectorUpliftRow
from pipeline.scm.trajectories import _rows_to_df

META_COLS = ["MONTH", "HOUR", "CLIMATE_CHANGE"]

HIST_CATASTROPHIC_MW = 51_104  # Sep 7 2022 CAISO peak


class SamplingMode(str, Enum):
    CO_OCCURRING = "co_occurring"
    SPLIT = "split"
    COMPONENT_MAX = "component_max"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    cec_scenario: str
    sampling: SamplingMode
    dynamic_lag: bool = False
    description: str = ""


CASE_BEST = CaseSpec(
    key="best",
    label="Best (Planning, co-occurring peak)",
    cec_scenario="Planning",
    sampling=SamplingMode.CO_OCCURRING,
    dynamic_lag=False,
    description="Reference CEC path; all components at annual MNL peak hour.",
)

CASE_AVERAGE = CaseSpec(
    key="average",
    label="Average (Planning–LocalRel midpoint)",
    cec_scenario="_blend_midpoint_",
    sampling=SamplingMode.CO_OCCURRING,
    dynamic_lag=False,
    description="50/50 blend of Planning and LocalRel stress-hour components.",
)

CASE_WORST = CaseSpec(
    key="worst",
    label="Worst (LocalRel, split peak)",
    cec_scenario="LocalRel",
    sampling=SamplingMode.SPLIT,
    dynamic_lag=True,
    description=(
        "Committed loads on MNL peak hour; fleet HVAC/lag scaled from "
        "annual max core load (UNADJ − DC − AAFS)."
    ),
)

CASE_ENVELOPE = CaseSpec(
    key="envelope",
    label="Envelope (LocalRel, component maxima)",
    cec_scenario="LocalRel",
    sampling=SamplingMode.COMPONENT_MAX,
    dynamic_lag=True,
    description="Theoretical upper bound — components never co-occur in one hour.",
)

ALL_CASES = [CASE_BEST, CASE_AVERAGE, CASE_WORST, CASE_ENVELOPE]


def core_load(row: pd.Series) -> float:
    return (
        float(row["UNADJUSTED_CONSUMPTION"])
        - float(row["DATA_CENTER"])
        - float(row["AAFS"])
    )


def build_stress_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("YEAR")["MANAGED_NET_LOAD"].idxmax()
    keep = ["YEAR"] + [c for c in COMP_COLS if c in df.columns]
    keep += [c for c in META_COLS if c in df.columns]
    out = df.loc[idx, keep].set_index("YEAR")
    out["CORE_LOAD"] = out.apply(core_load, axis=1)
    return out


def build_fleet_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["CORE_LOAD"] = tmp.apply(core_load, axis=1)
    idx = tmp.groupby("YEAR")["CORE_LOAD"].idxmax()
    keep = ["YEAR"] + [c for c in COMP_COLS if c in df.columns]
    keep += [c for c in META_COLS if c in df.columns]
    out = df.loc[idx, keep].set_index("YEAR")
    out["CORE_LOAD"] = out.apply(core_load, axis=1)
    return out


def build_component_max_table(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["YEAR"].unique())
    rows = []
    for yr in years:
        sub = df[df["YEAR"] == yr]
        row: dict = {"YEAR": yr}
        for col in COMP_COLS:
            if col in sub.columns:
                row[col] = sub[col].max()
        row["MANAGED_NET_LOAD"] = sub.loc[sub["MANAGED_NET_LOAD"].idxmax(), "MANAGED_NET_LOAD"]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("YEAR")
    out["CORE_LOAD"] = out.apply(core_load, axis=1)
    return out


def merge_case_row(
    stress_tbl: pd.DataFrame,
    fleet_tbl: pd.DataFrame,
    compmax_tbl: pd.DataFrame,
    yr: int,
    mode: SamplingMode,
) -> pd.Series:
    if mode == SamplingMode.CO_OCCURRING:
        row = stress_tbl.loc[yr].copy()
        for c in row.index:
            if c in META_COLS:
                continue
            row[c] = float(row[c])
        return row

    if mode == SamplingMode.SPLIT:
        stress = stress_tbl.loc[yr].copy()
        fleet = fleet_tbl.loc[yr]
        for c in stress.index:
            if c in META_COLS:
                continue
            stress[c] = float(stress[c])
        stress["UNADJUSTED_CONSUMPTION"] = float(fleet["UNADJUSTED_CONSUMPTION"])
        stress["CORE_LOAD"] = core_load(stress)
        return stress

    row = compmax_tbl.loc[yr].copy()
    for c in row.index:
        if c in META_COLS:
            continue
        row[c] = float(row[c])
    return row


def build_sampling_diagnostics(hourly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare stress vs fleet hour for peak composition shifts."""
    records = []
    for name, df in hourly.items():
        stress_tbl = build_stress_table(df)
        core_2025 = core_load(stress_tbl.loc[2025]) if 2025 in stress_tbl.index else None

        stress_idx = df.groupby("YEAR")["MANAGED_NET_LOAD"].idxmax()
        tmp = df.copy()
        tmp["CORE_LOAD"] = tmp.apply(core_load, axis=1)
        fleet_idx = tmp.groupby("YEAR")["CORE_LOAD"].idxmax()

        for yr in sorted(df["YEAR"].unique()):
            si = stress_idx[yr]
            fi = fleet_idx[yr]
            sr = df.loc[si]
            fr = df.loc[fi]
            sc = core_load(sr)
            records.append({
                "scenario": name,
                "year": int(yr),
                "stress_mnl": float(sr["MANAGED_NET_LOAD"]),
                "stress_unadj": float(sr["UNADJUSTED_CONSUMPTION"]),
                "stress_core": sc,
                "stress_dc": float(sr["DATA_CENTER"]),
                "stress_month": int(sr["MONTH"]),
                "stress_hour": int(sr["HOUR"]),
                "fleet_unadj": float(fr["UNADJUSTED_CONSUMPTION"]),
                "fleet_core": core_load(fr),
                "fleet_month": int(fr["MONTH"]),
                "fleet_hour": int(fr["HOUR"]),
                "same_hour": bool(si == fi),
                "core_below_2025_stress": bool(core_2025 is not None and sc < core_2025),
            })
    return pd.DataFrame(records)


def _apply_dynamic_lag(
    row: SectorUpliftRow,
    cal: SCMCalibration,
    fleet_unadj: float,
) -> SectorUpliftRow:
    if cal.unadj_2025 <= 0:
        return row
    growth = fleet_unadj / cal.unadj_2025
    new_lag = cal.d_residual * growth
    delta = new_lag - row.d_residual
    return dataclasses.replace(
        row,
        d_residual=new_lag,
        d_total=row.d_total + delta,
        mnl_hw=row.mnl_hw + delta,
    )


def build_midpoint_stress_table(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    """50/50 blend of stress-hour components between two scenarios."""
    ta = build_stress_table(df_a)
    tb = build_stress_table(df_b)
    years = ta.index.intersection(tb.index)
    rows = []
    for yr in years:
        blended = (ta.loc[yr].astype(float) + tb.loc[yr].astype(float)) / 2.0
        rows.append(blended)
    out = pd.DataFrame(rows, index=years)
    out["CORE_LOAD"] = out.apply(core_load, axis=1)
    return out


def resolve_hourly_for_case(
    case: CaseSpec,
    hourly: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Return (effective_hourly_or_proxy_df, optional_midpoint_stress_tbl).

    For blend midpoint case, returns Planning df but caller uses midpoint table.
    """
    if case.cec_scenario == "_blend_midpoint_":
        mid = build_midpoint_stress_table(hourly["Planning"], hourly["LocalRel"])
        return hourly["Planning"], mid
    return hourly[case.cec_scenario], None


def compute_case_trajectory(
    case: CaseSpec,
    hourly_df: pd.DataFrame,
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    dT_hub: float | None = None,
    stress_override: pd.DataFrame | None = None,
    uplift_method: UpliftMethod = UpliftMethod.YEAR_NATIVE,
    uplift_ctx: UpliftContext | None = None,
) -> pd.DataFrame:
    stress_tbl = stress_override if stress_override is not None else build_stress_table(hourly_df)
    fleet_tbl = build_fleet_table(hourly_df)
    compmax_tbl = build_component_max_table(hourly_df)

    rows: list[SectorUpliftRow] = []
    for yr in stress_tbl.index:
        peak_row = merge_case_row(
            stress_tbl, fleet_tbl, compmax_tbl, yr, case.sampling
        )

        if yr in sector_shares_df.index:
            shares = sector_shares_df.loc[yr].to_dict()
        else:
            nearest = sector_shares_df.index[
                (sector_shares_df.index - yr).abs().argmin()
            ]
            shares = sector_shares_df.loc[nearest].to_dict()

        use_dynamic_lag = (
            case.dynamic_lag
            or uplift_method == UpliftMethod.YEAR_NATIVE
        )
        row = compute_uplift(
            yr=int(yr),
            peak_row=peak_row,
            cal=cal,
            sector_shares_yr=shares,
            method=uplift_method,
            ctx=uplift_ctx,
            scenario=case.key,
            dT_hub=dT_hub,
            dynamic_lag=use_dynamic_lag,
        )

        if case.dynamic_lag and uplift_method == UpliftMethod.FLEET_SCALED:
            fleet_unadj = float(fleet_tbl.loc[yr, "UNADJUSTED_CONSUMPTION"])
            row = _apply_dynamic_lag(row, cal, fleet_unadj)

        rows.append(row)

    traj = _rows_to_df(rows)
    traj["case"] = case.key
    traj["case_label"] = case.label
    traj["stress_total"] = traj["mnl_cec"] + traj["d_total"]
    traj["headroom_mw"] = HIST_CATASTROPHIC_MW - traj["stress_total"]
    traj["headroom_vs_mnl_only"] = HIST_CATASTROPHIC_MW - traj["mnl_cec"]
    return traj


def apply_dynamic_lag_to_trajectory(
    traj: pd.DataFrame,
    cal: SCMCalibration,
    fleet_tbl: pd.DataFrame,
) -> pd.DataFrame:
    """Post-process a trajectory to scale residual/lag with fleet UNADJ growth."""
    out = traj.copy()
    for yr in out.index:
        if yr not in fleet_tbl.index:
            continue
        fleet_unadj = float(fleet_tbl.loc[yr, "UNADJUSTED_CONSUMPTION"])
        growth = fleet_unadj / cal.unadj_2025 if cal.unadj_2025 > 0 else 1.0
        new_lag = cal.d_residual * growth
        delta = new_lag - float(out.loc[yr, "d_residual"])
        out.loc[yr, "d_residual"] = new_lag
        out.loc[yr, "d_total"] = float(out.loc[yr, "d_total"]) + delta
        out.loc[yr, "mnl_hw"] = float(out.loc[yr, "mnl_hw"]) + delta
        out.loc[yr, "stress_total"] = float(out.loc[yr, "mnl_cec"]) + float(out.loc[yr, "d_total"])
        out.loc[yr, "headroom_mw"] = HIST_CATASTROPHIC_MW - float(out.loc[yr, "stress_total"])
    return out


def build_case_peak_table(
    hourly_df: pd.DataFrame,
    case: CaseSpec,
) -> pd.DataFrame:
    """Annual peak table under a CaseSpec's sampling rules."""
    stress_tbl = build_stress_table(hourly_df)
    fleet_tbl = build_fleet_table(hourly_df)
    compmax_tbl = build_component_max_table(hourly_df)
    rows = []
    for yr in stress_tbl.index:
        row = merge_case_row(stress_tbl, fleet_tbl, compmax_tbl, yr, case.sampling)
        row = row.copy()
        row["YEAR"] = yr
        rows.append(row)
    out = pd.DataFrame(rows).set_index("YEAR")
    return out


def compute_all_cases(
    hourly: dict[str, pd.DataFrame],
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    cases: list[CaseSpec] | None = None,
    dT_hub: float | None = None,
    uplift_method: UpliftMethod = UpliftMethod.YEAR_NATIVE,
    uplift_ctx: UpliftContext | None = None,
) -> dict[str, pd.DataFrame]:
    cases = cases or ALL_CASES
    out: dict[str, pd.DataFrame] = {}
    for case in cases:
        hourly_df, stress_override = resolve_hourly_for_case(case, hourly)
        out[case.key] = compute_case_trajectory(
            case, hourly_df, cal, sector_shares_df,
            dT_hub=dT_hub, stress_override=stress_override,
            uplift_method=uplift_method, uplift_ctx=uplift_ctx,
        )
    return out


def compute_method_comparison(
    hourly: dict[str, pd.DataFrame],
    cal: SCMCalibration,
    sector_shares_df: pd.DataFrame,
    case: CaseSpec,
    uplift_ctx: UpliftContext,
    dT_hub: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Run one case under static, fleet-scaled, and year-native uplift."""
    hourly_df, stress_override = resolve_hourly_for_case(case, hourly)
    out: dict[str, pd.DataFrame] = {}
    for method in UpliftMethod:
        out[method.value] = compute_case_trajectory(
            case, hourly_df, cal, sector_shares_df,
            dT_hub=dT_hub, stress_override=stress_override,
            uplift_method=method, uplift_ctx=uplift_ctx,
        )
    return out
