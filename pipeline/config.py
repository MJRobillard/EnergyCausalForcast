"""
Central configuration for the sectoral SCM heatwave pipeline.

All physical constants, SCM posterior means, sector elasticity ratios,
CEC file paths, and derived calibration parameters live here.
"""

from __future__ import annotations
from pathlib import Path

# ── Repository roots ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_CEC  = ROOT / "data" / "CEC"
DATA_WX   = ROOT / "data" / "california"
RESULTS   = ROOT / "results" / "california"
ANALYSIS  = ROOT / "analysis"

# ── CEC hourly scenario files ─────────────────────────────────────────────────
SCENARIO_FILES: dict[str, Path] = {
    "Planning":  DATA_CEC / "TN268127_20260105T135513_CED 2025 Hourly Forecast - CAISO - Planning_Scenario.xlsx",
    "LocalRel":  DATA_CEC / "TN268125_20260105T135506_CED 2025 Hourly Forecast - CAISO - Local_Reliability.xlsx",
    "PlusKnown": DATA_CEC / "TN268126_20260105T135510_CED 2025 Hourly Forecast - CAISO - Local_Reliability_plusKnown (2).xlsx",
}

# Historical sector consumption (for share calibration)
SECTOR_CONSUMPTION_FILE = DATA_CEC / "AGG_CONSUMPTION_ELEC_COUNTY_TBL_MONTHLY.xlsx"

# ── SCM posterior means (from results/california/param_store.pt) ──────────────
K_COOL      = 62.030   # MW/°F above T_cool_base
K_COOL_SQ   = -5.110   # MW/°F² above 80°F  (negative: nonlinear moderation at extremes)
K_MAX       = 53.692   # MW/°F above T_max_base
T_COOL_BASE = 64.360   # °F
T_MAX_BASE  = 85.662   # °F

# ── Hub-composite temperature weights (load-weighted CAISO geography) ─────────
HUB_WEIGHTS: dict[str, float] = {
    "bay_area":    0.65,
    "los_angeles": 0.15,
    "riverside":   0.15,
    "san_diego":   0.05,
}

# ── Sector heat-sensitivity weights (ACEEE/LBNL literature ratios) ────────────
#
# Residential HVAC is the most temperature-responsive end use; commercial is
# moderate (process cooling + HVAC diversity); industrial is dominated by
# process loads with low ambient sensitivity.
#
# These relative weights are normalised inside scm/sector_model.py so that
# sectoral β values sum to the pooled K_COOL / UNADJ_2025.
#
SECTOR_HEAT_WEIGHTS: dict[str, float] = {
    "RES": 2.0,   # residential: ~2× commercial AC intensity per MWh
    "COM": 1.0,   # commercial: reference
    "IND": 0.20,  # industrial: mostly process load, ~0.2× commercial
}

# ── AAFS fleet sector allocation ─────────────────────────────────────────────
# CEC AAFS = building electrification (heat pump adoption replacing gas heating).
# Residential dominates early (single-family heat pumps), commercial follows.
AAFS_SECTOR_SPLIT: dict[str, float] = {
    "RES": 0.70,
    "COM": 0.30,
    "IND": 0.00,
}

# ── EV fleet sector allocation ────────────────────────────────────────────────
# LIGHT_EV: personal vehicles — primarily residential home charging.
# AATE_LDV: light-duty commercial vehicle electrification (delivery fleets, etc.)
LIGHT_EV_SECTOR_SPLIT: dict[str, float] = {
    "RES": 0.85,
    "COM": 0.15,
    "IND": 0.00,
}
AATE_LDV_SECTOR_SPLIT: dict[str, float] = {
    "RES": 0.20,
    "COM": 0.50,
    "IND": 0.30,
}

# ── DC thermal model parameters ───────────────────────────────────────────────
COOLING_SHARE   = 0.35    # fraction of DC load that is cooling
THERMAL_SLOPE   = 0.0075  # additional cooling fraction per °F above threshold
T_DC_THRESHOLD  = 85.0    # °F
DC_COASTAL_SHARE = 0.70   # blended DC location: 70% coastal, 30% inland

# DATA_CENTER is treated as its own category (Commercial/IND hybrid),
# modelled separately using the DC thermal formula rather than β_com.

# ── EV elasticity relative to building HVAC ───────────────────────────────────
EV_BETA_RATIO = 0.40  # EVs have lower heat sensitivity than space-conditioning

# ── Observed 2022 heatwave anchor ─────────────────────────────────────────────
OBSERVED_2022_UPLIFT_MW = 4_625  # actual CAISO demand anomaly, Sep 5 18:00

# ── Sector share forward projection parameters ────────────────────────────────
# Linear annual drift in percentage-point sector share (capped at ±5pp from 2024)
# Derived from 2015-2024 CEC monthly trend.
SECTOR_SHARE_DRIFT_PP_YR: dict[str, float] = {
    "RES": +0.032,   # slowly increasing due to residential electrification
    "COM": -0.012,   # mild decline as energy efficiency offsets growth
    "IND": -0.022,   # industrial share declining as economy tertiarises
}
SECTOR_SHARE_ANCHOR_YEAR = 2024
# Max drift from anchor (prevents runaway extrapolation past 2050)
SECTOR_SHARE_MAX_DRIFT_PP = 5.0

# ── Baseline scenario for primary analysis ───────────────────────────────────
PRIMARY_SCENARIO = "LocalRel"
