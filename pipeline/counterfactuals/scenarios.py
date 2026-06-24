"""
Policy counterfactual scenario definitions.

Inherits the 4 original CFs from the notebook and adds 3 new
sector-specific interventions.

Original CFs
────────────
CF1   2× accelerated heat pump adoption (AAFS) — increases risk
CF2a  EV curtailment — β_ev = 0 (Flex Alert 100% compliance)
CF2b  V2G active discharge — β_ev < 0 (EVs feed homes during peak)
CF3   DC emergency curtailment — cap grid draw at 70% or 2030 level

New sector-specific CFs
───────────────────────
CF_RES_PRECOOL  Residential smart thermostat pre-cooling
                do(β_res = β_res × 0.60): pre-cooling 2h before peak
                shifts residential HVAC demand off the critical hour,
                effectively reducing the peak-hour heat sensitivity by 40%.

CF_IND_INTERRUPT  Industrial interruptible service
                  do(β_ind = 0): CAISO interruptible-service contracts
                  eliminate industrial heatwave contribution entirely.
                  Also models CAISO's curtailment of large industrial
                  process loads under Stage 3 Emergency conditions.

CF_TOU_SHOCK    Time-of-use price shock on residential AC
                do(β_res = β_res × (1 - ε_price × price_ratio)):
                Critical-peak pricing multiplies electricity price 4×
                during the peak hour.  Residential AC price elasticity
                ε = 0.10 (literature: LBNL 2022 CEC Report).
                Effect: β_res reduced by ~30% during the CPP window.

Industry-specific CFs (year-native uplift)
────────────────────────────────────────────
CF_DC_PUE         Data-center PUE mandate: 30% reduction in thermal draw
                  do(f_dc_thermal × 0.70) — improved cooling efficiency
                  and waste-heat recovery at hyperscale sites.

CF_DC_SHIFT       Off-peak compute scheduling: shift 25% of DC grid draw
                  away from the system peak hour via batch-job deferral.

CF_EV_FLEET_DR    Fleet-wide EV demand response: 85% curtailment of all
                  light-duty and commercial EV charging at peak.

CF_HEAVY_EV_DR    Medium/heavy-duty depot charging DR: 70% curtailment
                  of MHDV fleet charging during Flex Alert windows.

CF_INDUSTRY_STACK Combined industry levers: DC PUE + shift + EV fleet DR
                  + heavy EV DR (no residential interventions).

All CFs are registered in ALL_SCENARIOS for iteration.
"""

from __future__ import annotations
from pipeline.counterfactuals.engine import CounterfactualSpec


# ── Original CF parameters ────────────────────────────────────────────────────
# These are set lazily to avoid import of calibration before it is computed;
# `build_scenarios(cal, peak_tables)` returns the complete dict.

def build_scenarios(
    cal: "SCMCalibration",  # noqa: F821
    peak_tables: dict,
    primary_scenario: str = "LocalRel",
) -> dict[str, CounterfactualSpec]:
    """
    Instantiate all scenario specs given a calibrated SCMCalibration.

    Returns dict of {label -> CounterfactualSpec}.
    """
    peak_tbl = peak_tables[primary_scenario]

    # ── CF1: Accelerated heat pump adoption ──────────────────────────────
    def aafs_2x_ramp(yr: int) -> float:
        """Linear ramp from 1× at 2025 to 2× at 2050."""
        return 1.0 + 1.0 * (yr - 2025) / (2050 - 2025)

    cf1 = CounterfactualSpec(
        label="CF1: 2× heat pump adoption",
        description=(
            "A mandate/subsidy shock doubles AAFS fleet penetration by 2050 "
            "(linear ramp from 1× in 2025). "
            "Causal intervention: do(AAFS(Y) = 2× CEC projection). "
            "This INCREASES heatwave risk as more electrified buildings "
            "draw AC load during extreme events."
        ),
        aafs_scale_fn=aafs_2x_ramp,
    )

    # ── CF2a: Perfect EV demand response ─────────────────────────────────
    cf2a = CounterfactualSpec(
        label="CF2a: EV peak curtailment (β_ev = 0)",
        description=(
            "100% Flex Alert compliance suspends all EV charging during the "
            "peak heatwave hour. "
            "Causal intervention: do(β_ev = 0)."
        ),
        cal_overrides={"beta_ev": 0.0},
    )

    # ── CF2b: Vehicle-to-Grid discharge ───────────────────────────────────
    cf2b = CounterfactualSpec(
        label="CF2b: V2G active discharge (β_ev = −0.2×β_hvac)",
        description=(
            "EVs actively discharge into homes during the peak hour, "
            "providing a net demand reduction. "
            "Causal intervention: do(β_ev = −0.2 × β_aafs). "
            "This is the single most powerful individual lever by 2050."
        ),
        cal_overrides={"beta_ev": -0.20 * cal.beta_aafs},
    )

    # ── CF3: DC emergency curtailment ────────────────────────────────────
    DC_CAP_2030 = float(peak_tbl.loc[2030, "DATA_CENTER"]) if 2030 in peak_tbl.index else 4377.0

    def dc_curtail(yr: int, dc_mw: float) -> float:
        """Cap DC grid draw at lesser of 70% of load or 2030 level."""
        return min(dc_mw * 0.70, DC_CAP_2030)

    cf3 = CounterfactualSpec(
        label="CF3: DC emergency curtailment (70% cap)",
        description=(
            "CAISO orders data centers to shift non-essential compute to "
            "on-site generation or storage. Grid draw capped at min(70%×DC, DC_2030). "
            "Causal intervention: do(DC_grid(Y) = min(0.70×DC(Y), DC_2030))."
        ),
        dc_adj_fn=dc_curtail,
    )

    # ── CF_RES_PRECOOL: Residential smart thermostat pre-cooling ──────────
    # β_res is reduced by 40%: pre-cooling the thermal mass 2h before peak
    # shifts ~40% of residential AC demand off the critical hour.
    # Empirical basis: LBNL field trials show 30-50% peak reduction with
    # smart thermostat pre-cooling (CEC PIR-18-008).
    pre_cool_reduction = 0.40

    cf_res_precool = CounterfactualSpec(
        label="CF_RES_PRECOOL: Residential smart thermostat pre-cooling",
        description=(
            f"Residential smart thermostats pre-cool buildings 2h before "
            f"the heatwave peak, reducing peak-hour AC sensitivity by "
            f"{pre_cool_reduction*100:.0f}%. "
            f"Causal intervention: do(β_res = β_res × {1-pre_cool_reduction:.2f}). "
            f"Empirical: LBNL CEC PIR-18-008 (30–50% peak reduction)."
        ),
        sector_beta_overrides={"RES": cal.beta["RES"] * (1.0 - pre_cool_reduction)},
    )

    # ── CF_IND_INTERRUPT: Industrial interruptible service ────────────────
    # β_ind set to zero: CAISO's Stage 3 Emergency curtailment eliminates
    # industrial contribution to heatwave demand increment.
    cf_ind_interrupt = CounterfactualSpec(
        label="CF_IND_INTERRUPT: Industrial interruptible service",
        description=(
            "Under CAISO Stage 3 Emergency conditions, industrial customers "
            "on interruptible-service contracts curtail all non-essential "
            "process loads during the critical hour. "
            "Causal intervention: do(β_ind = 0). "
            "Effect is modest early but grows as industrial electrification increases."
        ),
        sector_beta_overrides={"IND": 0.0},
    )

    # ── CF_TOU_SHOCK: Critical-peak pricing on residential AC ─────────────
    # Critical-Peak Pricing (CPP) at 4× normal price during the peak event.
    # Residential AC price elasticity ε = −0.10 (LBNL 2022 estimate).
    # price_ratio = 4× → % reduction = ε × (price_ratio - 1) = 0.10 × 3 = 30%
    price_elasticity = 0.10
    price_ratio      = 4.0
    tou_reduction    = price_elasticity * (price_ratio - 1.0)  # 0.30

    cf_tou_shock = CounterfactualSpec(
        label=f"CF_TOU_SHOCK: Critical-peak pricing (4× rate, ε={price_elasticity})",
        description=(
            f"Critical-Peak Pricing at {price_ratio:.0f}× normal rate during the "
            f"heatwave peak hour. Residential AC elasticity ε={price_elasticity} "
            f"(LBNL 2022) → {tou_reduction*100:.0f}% demand reduction. "
            f"Causal intervention: do(β_res = β_res × {1-tou_reduction:.2f}). "
            f"Note: behavioural response is lower during extreme heat events; "
            f"ε=0.10 is a conservative estimate."
        ),
        sector_beta_overrides={"RES": cal.beta["RES"] * (1.0 - tou_reduction)},
    )

    # ── CF_DC_PUE: Data-center cooling efficiency mandate ─────────────────
    cf_dc_pue = CounterfactualSpec(
        label="CF_DC_PUE: DC PUE mandate (−30% thermal)",
        description=(
            "California mandates improved data-center PUE (1.15→1.05) with "
            "waste-heat recovery and liquid cooling. "
            "Causal intervention: do(f_dc_thermal × 0.70)."
        ),
        f_dc_scale=0.70,
    )

    # ── CF_DC_SHIFT: Off-peak compute scheduling ──────────────────────────
    def dc_shift(yr: int, dc_mw: float) -> float:
        return dc_mw * 0.75

    cf_dc_shift = CounterfactualSpec(
        label="CF_DC_SHIFT: DC off-peak scheduling (−25% peak draw)",
        description=(
            "Hyperscalers defer batch compute to off-peak hours under "
            "CAISO emergency orders. "
            "Causal intervention: do(DC_grid_peak = 0.75 × DC(Y))."
        ),
        dc_adj_fn=dc_shift,
    )

    # ── CF_EV_FLEET_DR: Fleet-wide light-duty EV curtailment ──────────────
    def ev_fleet_curtail(yr: int, lev: float, ldv: float) -> tuple[float, float]:
        keep = 0.15  # 85% curtailment
        return lev * keep, ldv * keep

    cf_ev_fleet_dr = CounterfactualSpec(
        label="CF_EV_FLEET_DR: Fleet EV DR (85% curtailment)",
        description=(
            "Mandatory Flex Alert compliance across all light-duty and "
            "commercial EV fleets suspends 85% of peak-hour charging. "
            "Causal intervention: do(EV_peak = 0.15 × EV(Y))."
        ),
        ev_adj_fn=ev_fleet_curtail,
        beta_ev_scale=0.0,
    )

    # ── CF_HEAVY_EV_DR: Medium/heavy-duty depot charging DR ───────────────
    def heavy_ev_curtail(yr: int, heavy_mw: float) -> float:
        return heavy_mw * 0.30  # 70% curtailment

    cf_heavy_ev_dr = CounterfactualSpec(
        label="CF_HEAVY_EV_DR: MHDV depot DR (70% curtailment)",
        description=(
            "Medium/heavy-duty fleet operators pause depot charging during "
            "Stage 2+ emergencies; 70% of MHDV peak draw eliminated. "
            "Causal intervention: do(MHEV_peak = 0.30 × MHEV(Y))."
        ),
        heavy_ev_adj_fn=heavy_ev_curtail,
        beta_heavy_ev_scale=0.0,
    )

    # ── CF_INDUSTRY_STACK: Combined DC + EV industry levers ───────────────
    cf_industry_stack = CounterfactualSpec(
        label="CF_INDUSTRY_STACK: DC + EV industry policy stack",
        description=(
            "Combined industry interventions: PUE mandate, off-peak DC shift, "
            "fleet EV DR, and MHDV depot curtailment. "
            "No residential β or AAFS changes."
        ),
        f_dc_scale=0.70,
        dc_adj_fn=dc_shift,
        ev_adj_fn=ev_fleet_curtail,
        heavy_ev_adj_fn=heavy_ev_curtail,
        beta_ev_scale=0.0,
        beta_heavy_ev_scale=0.0,
    )

    return {
        "CF1":              cf1,
        "CF2a":             cf2a,
        "CF2b":             cf2b,
        "CF3":              cf3,
        "CF_RES_PRECOOL":   cf_res_precool,
        "CF_IND_INTERRUPT": cf_ind_interrupt,
        "CF_TOU_SHOCK":     cf_tou_shock,
        "CF_DC_PUE":        cf_dc_pue,
        "CF_DC_SHIFT":      cf_dc_shift,
        "CF_EV_FLEET_DR":   cf_ev_fleet_dr,
        "CF_HEAVY_EV_DR":   cf_heavy_ev_dr,
        "CF_INDUSTRY_STACK": cf_industry_stack,
    }
