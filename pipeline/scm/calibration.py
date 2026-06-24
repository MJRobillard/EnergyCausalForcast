"""
SCM calibration layer.

Computes:
  - Structural HVAC uplift at the 2025 CAISO scale (pooled k_cool/k_max)
  - Sector-disaggregated β coefficients (RES/COM/IND)
  - DC thermal factor
  - Fixed residual (lag/behavioral/humidity), calibrated so all components
    sum exactly to OBSERVED_2022_UPLIFT_MW at 2025

All outputs are pure floats / small dicts — no DataFrames.  This keeps
the calibration layer testable and independent of the data loading layer.

Exposes
-------
SCMCalibration   dataclass   — all calibrated parameters
calibrate(peak_2025, hw_peak, sector_shares_2025) -> SCMCalibration
"""

from __future__ import annotations
from dataclasses import dataclass, field

from pipeline.config import (
    K_COOL, K_COOL_SQ, K_MAX,
    T_COOL_BASE, T_MAX_BASE,
    COOLING_SHARE, THERMAL_SLOPE, T_DC_THRESHOLD, DC_COASTAL_SHARE,
    EV_BETA_RATIO,
    SECTOR_HEAT_WEIGHTS,
    AAFS_SECTOR_SPLIT, LIGHT_EV_SECTOR_SPLIT, AATE_LDV_SECTOR_SPLIT,
    OBSERVED_2022_UPLIFT_MW,
)
from pipeline.data.weather import HeatwavePeak


@dataclass
class SCMCalibration:
    """
    Complete set of calibrated SCM parameters for the sectoral pipeline.

    All MW figures refer to CAISO peak-hour demand at the 2025 anchor.
    """
    # ── Pooled HVAC structural terms ─────────────────────────────────────
    d_hvac_linear:   float     # k_cool × (T_hub_2022 - T_cool_base)
    d_hvac_nl:       float     # k_cool_sq × quadratic term (negative)
    d_hvac_max:      float     # k_max × (T_rv_2022 - T_max_base)
    hvac_structural_2025: float  # sum of above three at 2025 UNADJ scale

    # ── Sector-specific β coefficients (per MW of sectoral UNADJ, per °F) ─
    beta: dict[str, float]     # {RES, COM, IND}

    # ── Pooled fleet betas (for AAFS/EV which are CEC-defined additive) ──
    beta_aafs: float           # fractional heat response for electrified buildings
    beta_ev:   float           # fractional heat response for EV fleet

    # ── DC thermal factor ─────────────────────────────────────────────────
    f_dc_thermal:    float     # cooling_share × thermal_slope × ΔT_effective
    dT_dc_effective: float     # blended hub/Riverside ΔT above DC threshold

    # ── 2025 component MW values (for calibration audit) ─────────────────
    d_aafs_2025:  float
    d_ev_2025:    float
    d_dc_2025:    float
    d_residual:   float        # fixed lag/behavioral/humidity residual

    # ── UNADJ anchor ──────────────────────────────────────────────────────
    unadj_2025:   float        # total UNADJUSTED_CONSUMPTION at 2025 anchor

    # ── Sector MW anchors ─────────────────────────────────────────────────
    unadj_sector_2025: dict[str, float]  # {RES, COM, IND} in MW

    # ── Actual heatwave ΔT_hub (stored to avoid re-inference errors) ─────
    # Always use the weather-derived value, never infer from β ratios.
    dT_hub: float = 0.0

    # ── Fleet capacity ceilings (Validation 2) ────────────────────────────
    # Cooling nameplate capacity per MW of UNADJ load, by sector.
    # Residential: ~0.40 cooling fraction at peak, commercial ~0.30, IND ~0.10
    nameplate_cooling_fraction: dict[str, float] = field(default_factory=lambda: {
        "RES": 0.40,
        "COM": 0.30,
        "IND": 0.10,
    })

    # ── Sector-specific AAFS / EV splits ─────────────────────────────────
    aafs_split:       dict[str, float] = field(default_factory=lambda: AAFS_SECTOR_SPLIT.copy())
    light_ev_split:   dict[str, float] = field(default_factory=lambda: LIGHT_EV_SECTOR_SPLIT.copy())
    aate_ldv_split:   dict[str, float] = field(default_factory=lambda: AATE_LDV_SECTOR_SPLIT.copy())


def _hvac_uplift_components(hw: HeatwavePeak) -> tuple[float, float, float]:
    """Return (d_linear, d_nl, d_max) SCM terms at 2025 pooled scale."""
    d_linear = K_COOL * (
        max(hw.T_hub - T_COOL_BASE, 0.0) - max(hw.T_hub_clim - T_COOL_BASE, 0.0)
    )
    d_nl = K_COOL_SQ * (
        max(hw.T_hub - 80.0, 0.0) ** 2 - max(hw.T_hub_clim - 80.0, 0.0) ** 2
    )
    d_max = K_MAX * (
        max(hw.T_rv - T_MAX_BASE, 0.0) - max(hw.T_rv_clim - T_MAX_BASE, 0.0)
    )
    return d_linear, d_nl, d_max


def _dc_thermal_factor(hw: HeatwavePeak) -> tuple[float, float]:
    """Return (f_dc_thermal, dT_dc_effective)."""
    dT_hub_dc = max(hw.T_hub - T_DC_THRESHOLD, 0.0) - max(hw.T_hub_clim - T_DC_THRESHOLD, 0.0)
    dT_rv_dc  = max(hw.T_rv  - T_DC_THRESHOLD, 0.0) - max(hw.T_rv_clim  - T_DC_THRESHOLD, 0.0)
    dT_eff    = DC_COASTAL_SHARE * dT_hub_dc + (1.0 - DC_COASTAL_SHARE) * dT_rv_dc
    f         = COOLING_SHARE * THERMAL_SLOPE * dT_eff
    return f, dT_eff


def _sector_betas(
    unadj_2025: float,
    sector_shares: dict[str, float],
    hvac_structural_2025: float,
    dT_hub: float,
) -> dict[str, float]:
    """
    Compute sector-specific β coefficients (MW per MW of sectoral UNADJ per °F).

    β_s is sized so that Σ_s (β_s × UNADJ_s(2025) × dT_hub) = hvac_structural_2025.
    This ensures the sector model exactly reconstructs the pooled HVAC anchor
    at the 2025 calibration year (required by Validation 1).

    The literature heat-sensitivity weights (SECTOR_HEAT_WEIGHTS) determine the
    relative split between sectors; the overall magnitude is set by the anchor.

        weighted_norm_s = w_s × share_s / Σ(w_s' × share_s')
        hvac_s_2025     = weighted_norm_s × hvac_structural_2025
        β_s             = hvac_s_2025 / (UNADJ_s_2025 × dT_hub)
    """
    if dT_hub == 0:
        raise ValueError("dT_hub must be non-zero for sector beta calibration.")

    w = SECTOR_HEAT_WEIGHTS
    weighted_total = sum(sector_shares[s] * w[s] for s in ["RES", "COM", "IND"])

    betas: dict[str, float] = {}
    for s in ["RES", "COM", "IND"]:
        hvac_s_2025 = hvac_structural_2025 * (sector_shares[s] * w[s] / weighted_total)
        unadj_s_2025 = unadj_2025 * sector_shares[s]
        betas[s] = hvac_s_2025 / (unadj_s_2025 * dT_hub) if unadj_s_2025 > 0 else 0.0

    return betas


def calibrate(
    peak_2025: "pd.Series",  # noqa: F821 — row from peak_tables
    hw: HeatwavePeak,
    sector_shares_2025: dict[str, float],
) -> SCMCalibration:
    """
    Produce a fully-calibrated SCMCalibration object.

    Parameters
    ----------
    peak_2025         : row from peak_tables[scenario].loc[2025]
                        expects keys: UNADJUSTED_CONSUMPTION, AAFS,
                        LIGHT_EV, AATE_LDV, DATA_CENTER
    hw                : HeatwavePeak for the 2022 event peak hour
    sector_shares_2025: {RES, COM, IND} fractions summing to 1
    """
    unadj_2025 = float(peak_2025["UNADJUSTED_CONSUMPTION"])
    aafs_2025  = float(peak_2025["AAFS"])
    ev_2025    = float(peak_2025["LIGHT_EV"]) + float(peak_2025["AATE_LDV"])
    dc_2025    = float(peak_2025["DATA_CENTER"])

    d_lin, d_nl, d_max = _hvac_uplift_components(hw)
    hvac_struct = d_lin + d_nl + d_max

    f_dc, dT_dc_eff = _dc_thermal_factor(hw)

    beta = _sector_betas(unadj_2025, sector_shares_2025, hvac_struct, hw.dT_hub)
    beta_aafs = K_COOL / unadj_2025
    beta_ev   = EV_BETA_RATIO * beta_aafs

    d_aafs = aafs_2025 * beta_aafs * hw.dT_hub
    d_ev   = ev_2025   * beta_ev   * hw.dT_hub
    d_dc   = dc_2025   * f_dc

    d_residual = OBSERVED_2022_UPLIFT_MW - hvac_struct - d_aafs - d_ev - d_dc

    unadj_sector = {s: unadj_2025 * sector_shares_2025[s] for s in ["RES", "COM", "IND"]}

    return SCMCalibration(
        d_hvac_linear=d_lin,
        d_hvac_nl=d_nl,
        d_hvac_max=d_max,
        hvac_structural_2025=hvac_struct,
        beta=beta,
        beta_aafs=beta_aafs,
        beta_ev=beta_ev,
        f_dc_thermal=f_dc,
        dT_dc_effective=dT_dc_eff,
        d_aafs_2025=d_aafs,
        d_ev_2025=d_ev,
        d_dc_2025=d_dc,
        d_residual=d_residual,
        unadj_2025=unadj_2025,
        unadj_sector_2025=unadj_sector,
        dT_hub=hw.dT_hub,
    )
