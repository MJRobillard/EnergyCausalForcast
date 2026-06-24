"""
Year-native heatwave uplift engine.

Answers: "What would a 2022-severity heatwave do in year Y given Y's fleet
and climatology?" rather than "What if we replay 2022 weather on Y's fleet?"

Methods
-------
STATIC_2022    — flat 4,625 MW benchmark (not a causal projection)
FLEET_SCALED   — β × fleet(Y) × ΔT_2022 + fixed/scaled lag (legacy pipeline)
YEAR_NATIVE    — structural SCM counterfactual at severity-matched weather
                 on year-specific climatology + fleet-scaled persistence lag
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from pipeline.config import (
    K_COOL,
    K_COOL_SQ,
    K_MAX,
    T_COOL_BASE,
    T_MAX_BASE,
    COOLING_SHARE,
    THERMAL_SLOPE,
    T_DC_THRESHOLD,
    DC_COASTAL_SHARE,
    SECTOR_HEAT_WEIGHTS,
    AAFS_SECTOR_SPLIT,
    LIGHT_EV_SECTOR_SPLIT,
    AATE_LDV_SECTOR_SPLIT,
    OBSERVED_2022_UPLIFT_MW,
    EV_BETA_RATIO,
)
from pipeline.data.weather import HeatwavePeak
from pipeline.scm.calibration import SCMCalibration
from pipeline.scm.sector_model import SectorUpliftRow, compute_sector_uplift

# Medium/heavy-duty fleet heat sensitivity (depot + on-route charging at peak)
HEAVY_EV_BETA_RATIO = 0.55
HEAVY_EV_SECTOR_SPLIT: dict[str, float] = {
    "RES": 0.05,
    "COM": 0.35,
    "IND": 0.60,
}

# Fraction of CEC CLIMATE_CHANGE MW mapped to hub warming (avoids double-count
# with UNADJ growth already embedding much of the climate signal).
CLIMATE_WARMING_ATTRIBUTION = 0.25


class UpliftMethod(str, Enum):
    STATIC_2022 = "static_2022"
    FLEET_SCALED = "fleet_scaled"
    YEAR_NATIVE = "year_native"


@dataclass
class YearWeatherProfile:
    """Severity-matched heatwave weather for a projection year."""
    year: int
    month: int
    hour: int
    T_hub_clim: float
    T_rv_clim: float
    T_hub_event: float
    T_rv_event: float
    dT_hub: float
    dT_rv: float
    dT_dc_effective: float
    climate_warming_offset: float


@dataclass
class UpliftContext:
    """Shared weather anchor for a forward scenario run."""
    hw_anchor: HeatwavePeak
    hub_clim: pd.Series
    rv_clim: pd.Series
    climate_change_2025: float = 0.0


def _hvac_structural(T_hub_event: float, T_hub_clim: float,
                     T_rv_event: float, T_rv_clim: float) -> tuple[float, float, float, float]:
    d_linear = K_COOL * (
        max(T_hub_event - T_COOL_BASE, 0.0) - max(T_hub_clim - T_COOL_BASE, 0.0)
    )
    d_nl = K_COOL_SQ * (
        max(T_hub_event - 80.0, 0.0) ** 2 - max(T_hub_clim - 80.0, 0.0) ** 2
    )
    d_max = K_MAX * (
        max(T_rv_event - T_MAX_BASE, 0.0) - max(T_rv_clim - T_MAX_BASE, 0.0)
    )
    return d_linear, d_nl, d_max, d_linear + d_nl + d_max


def _dc_thermal_factor(T_hub_event: float, T_hub_clim: float,
                       T_rv_event: float, T_rv_clim: float) -> tuple[float, float]:
    dT_hub_dc = (
        max(T_hub_event - T_DC_THRESHOLD, 0.0)
        - max(T_hub_clim - T_DC_THRESHOLD, 0.0)
    )
    dT_rv_dc = (
        max(T_rv_event - T_DC_THRESHOLD, 0.0)
        - max(T_rv_clim - T_DC_THRESHOLD, 0.0)
    )
    dT_eff = DC_COASTAL_SHARE * dT_hub_dc + (1.0 - DC_COASTAL_SHARE) * dT_rv_dc
    f = COOLING_SHARE * THERMAL_SLOPE * dT_eff
    return f, dT_eff


def _climate_warming_offset(
    yr: int,
    peak_row: pd.Series,
    ctx: UpliftContext,
) -> float:
    cc = float(peak_row.get("CLIMATE_CHANGE", ctx.climate_change_2025))
    d_cc = max(cc - ctx.climate_change_2025, 0.0)
    return (d_cc / K_COOL) * CLIMATE_WARMING_ATTRIBUTION if K_COOL > 0 else 0.0


def build_year_weather_profile(
    yr: int,
    peak_row: pd.Series,
    ctx: UpliftContext,
) -> YearWeatherProfile:
    hw = ctx.hw_anchor
    month = int(peak_row.get("MONTH", hw.month))
    if "period" in peak_row.index and pd.notna(peak_row["period"]):
        hour = int(pd.Timestamp(peak_row["period"]).hour)
    else:
        # CEC HOUR column is 1-based (1–24)
        hour = int(peak_row.get("HOUR", hw.hour + 1)) - 1

    key = (month, hour)
    if key not in ctx.hub_clim.index:
        key = (hw.month, hw.hour)

    warm = _climate_warming_offset(yr, peak_row, ctx)
    T_hub_clim = float(ctx.hub_clim.loc[key]) + warm
    T_rv_clim = float(ctx.rv_clim.loc[key]) + warm

    # Same anomaly severity as 2022 anchor, applied to year-Y climatology
    T_hub_event = T_hub_clim + hw.dT_hub
    T_rv_event = T_rv_clim + hw.dT_rv
    _, dT_dc_eff = _dc_thermal_factor(T_hub_event, T_hub_clim, T_rv_event, T_rv_clim)

    return YearWeatherProfile(
        year=yr,
        month=month,
        hour=hour,
        T_hub_clim=T_hub_clim,
        T_rv_clim=T_rv_clim,
        T_hub_event=T_hub_event,
        T_rv_event=T_rv_event,
        dT_hub=T_hub_event - T_hub_clim,
        dT_rv=T_rv_event - T_rv_clim,
        dT_dc_effective=dT_dc_eff,
        climate_warming_offset=warm,
    )


def _sector_hvac_from_struct(
    hvac_struct: float,
    unadj: float,
    unadj_2025: float,
    sector_shares: dict[str, float],
) -> dict[str, float]:
    """Allocate pooled structural HVAC to sectors by heat weight × fleet."""
    if unadj_2025 <= 0 or hvac_struct <= 0:
        return {s: 0.0 for s in ["RES", "COM", "IND"]}

    w = SECTOR_HEAT_WEIGHTS
    weighted_total = sum(sector_shares[s] * w[s] for s in ["RES", "COM", "IND"])
    fleet_scale = unadj / unadj_2025

    out: dict[str, float] = {}
    for s in ["RES", "COM", "IND"]:
        share = sector_shares[s] * w[s] / weighted_total if weighted_total > 0 else 0.0
        out[s] = hvac_struct * share * fleet_scale
    return out


def _apply_capacity_ceiling(
    d_hvac: dict[str, float],
    unadj_s: dict[str, float],
    cal: SCMCalibration,
) -> tuple[dict[str, float], bool, list[str], float]:
    clipped = False
    sectors: list[str] = []
    clip_amt = 0.0
    out: dict[str, float] = {}
    for s in ["RES", "COM", "IND"]:
        ceiling = unadj_s[s] * cal.nameplate_cooling_fraction[s]
        raw = d_hvac[s]
        if raw > ceiling:
            clipped = True
            sectors.append(s)
            clip_amt += raw - ceiling
            out[s] = ceiling
        else:
            out[s] = raw
    return out, clipped, sectors, clip_amt


def _dynamic_lag(
    cal: SCMCalibration,
    unadj: float,
    weather: YearWeatherProfile,
    hw_anchor: HeatwavePeak,
) -> float:
    fleet_growth = unadj / cal.unadj_2025 if cal.unadj_2025 > 0 else 1.0
    persist_ratio = weather.dT_rv / hw_anchor.dT_rv if hw_anchor.dT_rv > 0 else 1.0
    return cal.d_residual * fleet_growth * persist_ratio


def compute_year_native_uplift(
    yr: int,
    peak_row: pd.Series,
    cal: SCMCalibration,
    sector_shares_yr: dict[str, float],
    ctx: UpliftContext,
    scenario: str = "",
    f_dc_scale: float = 1.0,
    beta_ev_scale: float = 1.0,
    beta_heavy_ev_scale: float = 1.0,
    sector_hvac_mult: dict[str, float] | None = None,
    dynamic_lag: bool = True,
) -> SectorUpliftRow:
    weather = build_year_weather_profile(yr, peak_row, ctx)

    unadj = float(peak_row["UNADJUSTED_CONSUMPTION"])
    aafs = float(peak_row["AAFS"])
    lev = float(peak_row["LIGHT_EV"])
    ldv = float(peak_row["AATE_LDV"])
    heavy = float(peak_row.get("MEDIUM_HEAVY_EV", 0.0))
    dc = float(peak_row["DATA_CENTER"])
    mnl = float(peak_row["MANAGED_NET_LOAD"])

    unadj_s = {s: unadj * sector_shares_yr[s] for s in ["RES", "COM", "IND"]}

    _, _, _, hvac_struct = _hvac_structural(
        weather.T_hub_event, weather.T_hub_clim,
        weather.T_rv_event, weather.T_rv_clim,
    )
    d_hvac_raw = _sector_hvac_from_struct(
        hvac_struct, unadj, cal.unadj_2025, sector_shares_yr,
    )
    mult = sector_hvac_mult or {}
    for s in ["RES", "COM", "IND"]:
        d_hvac_raw[s] *= mult.get(s, 1.0)
    d_hvac, clipped, clipped_sectors, clip_amount = _apply_capacity_ceiling(
        d_hvac_raw, unadj_s, cal,
    )

    dT = weather.dT_hub
    beta_ev = cal.beta_ev * beta_ev_scale
    beta_heavy = cal.beta_aafs * HEAVY_EV_BETA_RATIO * beta_heavy_ev_scale

    d_aafs_res = aafs * AAFS_SECTOR_SPLIT["RES"] * cal.beta_aafs * dT
    d_aafs_com = aafs * AAFS_SECTOR_SPLIT["COM"] * cal.beta_aafs * dT

    ev_res = lev * LIGHT_EV_SECTOR_SPLIT["RES"] + ldv * AATE_LDV_SECTOR_SPLIT["RES"]
    ev_com = lev * LIGHT_EV_SECTOR_SPLIT["COM"] + ldv * AATE_LDV_SECTOR_SPLIT["COM"]
    ev_ind = lev * LIGHT_EV_SECTOR_SPLIT["IND"] + ldv * AATE_LDV_SECTOR_SPLIT["IND"]
    d_ev_res = ev_res * beta_ev * dT
    d_ev_com = ev_com * beta_ev * dT
    d_ev_ind = ev_ind * beta_ev * dT

    heavy_res = heavy * HEAVY_EV_SECTOR_SPLIT["RES"]
    heavy_com = heavy * HEAVY_EV_SECTOR_SPLIT["COM"]
    heavy_ind = heavy * HEAVY_EV_SECTOR_SPLIT["IND"]
    d_heavy_res = heavy_res * beta_heavy * dT
    d_heavy_com = heavy_com * beta_heavy * dT
    d_heavy_ind = heavy_ind * beta_heavy * dT

    f_dc, _ = _dc_thermal_factor(
        weather.T_hub_event, weather.T_hub_clim,
        weather.T_rv_event, weather.T_rv_clim,
    )
    d_dc = dc * f_dc * f_dc_scale

    d_lag = _dynamic_lag(cal, unadj, weather, ctx.hw_anchor) if dynamic_lag else cal.d_residual

    d_total = (
        sum(d_hvac.values())
        + d_aafs_res + d_aafs_com
        + d_ev_res + d_ev_com + d_ev_ind
        + d_heavy_res + d_heavy_com + d_heavy_ind
        + d_dc
        + d_lag
    )

    # Fold heavy EV into sector EV columns for downstream compatibility
    return SectorUpliftRow(
        year=yr,
        scenario=scenario,
        d_hvac_res=d_hvac["RES"],
        d_hvac_com=d_hvac["COM"],
        d_hvac_ind=d_hvac["IND"],
        d_aafs_res=d_aafs_res,
        d_aafs_com=d_aafs_com,
        d_ev_res=d_ev_res + d_heavy_res,
        d_ev_com=d_ev_com + d_heavy_com,
        d_ev_ind=d_ev_ind + d_heavy_ind,
        d_dc=d_dc,
        d_residual=d_lag,
        d_total=d_total,
        d_static=float(OBSERVED_2022_UPLIFT_MW),
        mnl_cec=mnl,
        mnl_hw=mnl + d_total,
        capacity_clipped=clipped,
        clipped_sectors=clipped_sectors,
        clip_amount_mw=clip_amount,
    )


def compute_static_uplift(
    yr: int,
    peak_row: pd.Series,
    scenario: str = "",
) -> SectorUpliftRow:
    mnl = float(peak_row["MANAGED_NET_LOAD"])
    d = float(OBSERVED_2022_UPLIFT_MW)
    return SectorUpliftRow(
        year=yr, scenario=scenario,
        d_hvac_res=0, d_hvac_com=0, d_hvac_ind=0,
        d_aafs_res=0, d_aafs_com=0,
        d_ev_res=0, d_ev_com=0, d_ev_ind=0,
        d_dc=0, d_residual=d, d_total=d, d_static=d,
        mnl_cec=mnl, mnl_hw=mnl + d,
    )


def compute_uplift(
    yr: int,
    peak_row: pd.Series,
    cal: SCMCalibration,
    sector_shares_yr: dict[str, float],
    method: UpliftMethod = UpliftMethod.YEAR_NATIVE,
    ctx: UpliftContext | None = None,
    scenario: str = "",
    dT_hub: float | None = None,
    dynamic_lag: bool = False,
    f_dc_scale: float = 1.0,
    beta_ev_scale: float = 1.0,
    beta_heavy_ev_scale: float = 1.0,
    sector_hvac_mult: dict[str, float] | None = None,
) -> SectorUpliftRow:
    """Dispatch uplift computation by method."""
    if method == UpliftMethod.STATIC_2022:
        return compute_static_uplift(yr, peak_row, scenario)

    if method == UpliftMethod.FLEET_SCALED:
        return compute_sector_uplift(
            yr=yr, peak_row=peak_row, cal=cal,
            sector_shares_yr=sector_shares_yr,
            scenario=scenario, dT_hub=dT_hub,
        )

    if ctx is None:
        raise ValueError("YEAR_NATIVE uplift requires UpliftContext")

    return compute_year_native_uplift(
        yr=yr, peak_row=peak_row, cal=cal,
        sector_shares_yr=sector_shares_yr, ctx=ctx,
        scenario=scenario,         f_dc_scale=f_dc_scale,
        beta_ev_scale=beta_ev_scale,
        beta_heavy_ev_scale=beta_heavy_ev_scale,
        sector_hvac_mult=sector_hvac_mult,
        dynamic_lag=dynamic_lag,
    )


def build_uplift_context(
    hw_anchor: HeatwavePeak,
    hub_clim: pd.Series,
    rv_clim: pd.Series,
    climate_change_2025: float = 0.0,
) -> UpliftContext:
    return UpliftContext(
        hw_anchor=hw_anchor,
        hub_clim=hub_clim,
        rv_clim=rv_clim,
        climate_change_2025=climate_change_2025,
    )
