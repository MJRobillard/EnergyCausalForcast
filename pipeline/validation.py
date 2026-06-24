"""
Four structural validation checks for the sectoral SCM pipeline.

Each validator returns a ValidationResult.  run_all_validations() executes
them in order, prints a structured report, and raises PipelineHaltError if
any check mandates a halt.

───────────────────────────────────────────────────────────────────────────────
Check 1  — Conservation of MW (Calibration Anchor)
  Sum of disaggregated 2025 components must equal 4,625 MW ± 1 MW.
  Halt on failure; pipeline recalibrates the residual and retries once.

Check 2  — Physical Feasibility Bounds (Capacity Saturation)
  HVAC cooling uplift per sector cannot exceed nameplate capacity ceiling.
  No halt — clipping is applied in sector_model.py and flagged here.
  Reports all years/scenarios where saturation occurred.

Check 3  — Structural Divergence from Static Baseline
  The 2050 causal total must differ from the naive static 4,625 MW by
  more than 1% of static.  If not, the dynamic scaling may be broken.
  Issues a warning but does not halt.

Check 4  — Parameter Sensitivity Stress-Test (β_aafs ± 15%)
  Runs two shadow simulations with β_aafs perturbed by ±15%.
  If the resulting swing in 2050 d_total exceeds 50% of the baseline
  d_total, emit a "High Volatility Alert".
  No halt.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd

from pipeline.config import OBSERVED_2022_UPLIFT_MW, PRIMARY_SCENARIO


# ── Result types ──────────────────────────────────────────────────────────────

class CheckStatus(Enum):
    PASS    = "PASS"
    WARNING = "WARNING"
    FAIL    = "FAIL"


@dataclass
class ValidationResult:
    check_id:    int
    name:        str
    status:      CheckStatus
    message:     str
    detail:      dict = field(default_factory=dict)
    halt:        bool = False

    def __str__(self) -> str:
        icon = {"PASS": "✓", "WARNING": "⚠", "FAIL": "✗"}[self.status.value]
        lines = [f"  [{icon}] Check {self.check_id}: {self.name}  →  {self.status.value}"]
        lines.append(f"      {self.message}")
        for k, v in self.detail.items():
            lines.append(f"      {k}: {v}")
        if self.halt:
            lines.append("      *** PIPELINE HALTED — see failure protocol ***")
        return "\n".join(lines)


class PipelineHaltError(RuntimeError):
    """Raised when a fatal validation fails and the pipeline must stop."""
    pass


# ── Check 1 — Conservation of MW ─────────────────────────────────────────────

def check_calibration_anchor(
    cal: "SCMCalibration",  # noqa: F821
    tolerance_mw: float = 1.0,
) -> ValidationResult:
    """
    Verify that disaggregated 2025 components sum to OBSERVED_2022_UPLIFT_MW.

    Components:
        HVAC structural (all sectors) + AAFS + EV + DC thermal + residual
    """
    total = (
        cal.hvac_structural_2025
        + cal.d_aafs_2025
        + cal.d_ev_2025
        + cal.d_dc_2025
        + cal.d_residual
    )
    discrepancy = abs(total - OBSERVED_2022_UPLIFT_MW)

    detail = {
        "HVAC structural (MW)": f"{cal.hvac_structural_2025:+,.2f}",
        "AAFS (MW)":            f"{cal.d_aafs_2025:+,.2f}",
        "EV fleet (MW)":        f"{cal.d_ev_2025:+,.2f}",
        "DC thermal (MW)":      f"{cal.d_dc_2025:+,.2f}",
        "Residual/lag (MW)":    f"{cal.d_residual:+,.2f}",
        "Sum (MW)":             f"{total:,.3f}",
        "Target (MW)":          f"{OBSERVED_2022_UPLIFT_MW:,}",
        "Discrepancy (MW)":     f"{discrepancy:.4f}",
        "Tolerance (MW)":       f"±{tolerance_mw}",
    }

    sector_detail = {
        f"β_{s} (MW/MW/°F)": f"{cal.beta[s]:.6f}" for s in ["RES", "COM", "IND"]
    }
    detail.update(sector_detail)

    if discrepancy <= tolerance_mw:
        return ValidationResult(
            check_id=1,
            name="Conservation of MW — Calibration Anchor",
            status=CheckStatus.PASS,
            message=(
                f"Disaggregated 2025 components sum to {total:,.3f} MW "
                f"(target {OBSERVED_2022_UPLIFT_MW:,} MW, Δ={discrepancy:.4f} MW)."
            ),
            detail=detail,
            halt=False,
        )
    else:
        return ValidationResult(
            check_id=1,
            name="Conservation of MW — Calibration Anchor",
            status=CheckStatus.FAIL,
            message=(
                f"CALIBRATION FAILURE: components sum to {total:,.3f} MW but "
                f"target is {OBSERVED_2022_UPLIFT_MW:,} MW "
                f"(discrepancy = {discrepancy:.4f} MW > tolerance {tolerance_mw} MW). "
                f"Residual lag distribution must be recalibrated."
            ),
            detail=detail,
            halt=True,
        )


# ── Check 2 — Physical Feasibility / Capacity Saturation ─────────────────────

def check_capacity_saturation(
    trajectories: dict[str, pd.DataFrame],
) -> ValidationResult:
    """
    Scan all trajectory DataFrames for rows where capacity_clipped == True.

    Reports: which scenarios / years / sectors were clipped, and total
    clipped MW.  Issues WARNING if any clipping occurred; PASS if none.
    """
    clipped_events: list[dict] = []
    for scen, df in trajectories.items():
        if "capacity_clipped" not in df.columns:
            continue
        clipped_rows = df[df["capacity_clipped"]]
        for yr, row in clipped_rows.iterrows():
            clipped_events.append({
                "scenario": scen,
                "year":     yr,
                "sectors":  row.get("clipped_sectors", ""),
                "clip_mw":  row.get("clip_amount_mw", 0.0),
            })

    if not clipped_events:
        return ValidationResult(
            check_id=2,
            name="Physical Feasibility — Capacity Saturation",
            status=CheckStatus.PASS,
            message="No capacity ceiling breaches detected across all scenarios and years.",
            halt=False,
        )

    total_clipped_events = len(clipped_events)
    max_clip = max(e["clip_mw"] for e in clipped_events)
    affected_years = sorted({e["year"] for e in clipped_events})

    detail = {
        "Total clipping events": str(total_clipped_events),
        "Max clip per event (MW)": f"{max_clip:.1f}",
        "Affected years": str(affected_years),
        "Events (first 5)": str(clipped_events[:5]),
    }
    return ValidationResult(
        check_id=2,
        name="Physical Feasibility — Capacity Saturation",
        status=CheckStatus.WARNING,
        message=(
            f"CAPACITY SATURATION WARNING: {total_clipped_events} year/scenario "
            f"combinations breached their sector cooling nameplate ceiling. "
            f"Values were clipped to physical maximum. Max single clip: {max_clip:.1f} MW."
        ),
        detail=detail,
        halt=False,
    )


# ── Check 3 — Structural Divergence from Static Baseline ─────────────────────

def check_structural_divergence(
    trajectories: dict[str, pd.DataFrame],
    scenario: str | None = None,
    target_year: int = 2050,
    min_divergence_pct: float = 1.0,
) -> ValidationResult:
    """
    Verify that the 2050 causal total diverges meaningfully from the static baseline.

    If the causal and static values are within min_divergence_pct of each other,
    the dynamic scaling is likely still pooled/locked — issue a WARNING.
    """
    scen = scenario or PRIMARY_SCENARIO
    if scen not in trajectories:
        scen = list(trajectories.keys())[0]

    df = trajectories[scen]
    if target_year not in df.index:
        avail = df.index.tolist()
        return ValidationResult(
            check_id=3,
            name="Structural Divergence — Dynamic vs Static",
            status=CheckStatus.WARNING,
            message=f"Target year {target_year} not in trajectory; available: {avail[-5:]}",
            halt=False,
        )

    d_causal = df.loc[target_year, "d_total"]
    d_static = df.loc[target_year, "d_static"]
    pct_diff  = abs(d_causal - d_static) / d_static * 100.0

    detail = {
        "Scenario":          scen,
        "Causal δMNL 2050":  f"{d_causal:,.1f} MW",
        "Static δMNL 2050":  f"{d_static:,.1f} MW",
        "Divergence":        f"{pct_diff:.2f}%",
        "Minimum required":  f"{min_divergence_pct:.1f}%",
    }

    # Also show trajectory divergence at intermediate years
    display_years = [y for y in [2030, 2035, 2040, 2045, 2050] if y in df.index]
    for y in display_years:
        detail[f"vs_static_{y}"] = f"{df.loc[y, 'vs_static']:+,.0f} MW"

    if pct_diff >= min_divergence_pct:
        return ValidationResult(
            check_id=3,
            name="Structural Divergence — Dynamic vs Static",
            status=CheckStatus.PASS,
            message=(
                f"Dynamic model diverges {pct_diff:.2f}% from static by {target_year} "
                f"(+{d_causal - d_static:+,.0f} MW). Sector scaling is active."
            ),
            detail=detail,
            halt=False,
        )
    else:
        return ValidationResult(
            check_id=3,
            name="Structural Divergence — Dynamic vs Static",
            status=CheckStatus.WARNING,
            message=(
                f"DIVERGENCE WARNING: Causal model is only {pct_diff:.2f}% from static "
                f"at {target_year} (threshold: {min_divergence_pct:.1f}%). "
                f"Dynamic lag/sector scaling parameters may still be pooled or locked. "
                f"Check SECTOR_SHARE_DRIFT_PP_YR and sector beta weights."
            ),
            detail=detail,
            halt=False,
        )


# ── Check 4 — Parameter Sensitivity Stress-Test ───────────────────────────────

def check_beta_aafs_sensitivity(
    peak_tbl: pd.DataFrame,
    cal: "SCMCalibration",  # noqa: F821
    sector_shares_df: pd.DataFrame,
    perturbation_pct: float = 15.0,
    swing_threshold_pct: float = 50.0,
    target_year: int = 2050,
) -> ValidationResult:
    """
    Perturb β_aafs by ±perturbation_pct% and recompute d_total at target_year.

    If the ±swing in d_total exceeds swing_threshold_pct% of the baseline
    d_total, emit a HIGH VOLATILITY ALERT.
    """
    import copy
    from pipeline.scm.sector_model import compute_sector_uplift

    if target_year not in peak_tbl.index:
        return ValidationResult(
            check_id=4,
            name="Sensitivity Stress-Test — β_aafs ±15%",
            status=CheckStatus.WARNING,
            message=f"Target year {target_year} not in peak table.",
            halt=False,
        )

    peak_row = peak_tbl.loc[target_year]
    if target_year in sector_shares_df.index:
        shares = sector_shares_df.loc[target_year].to_dict()
    else:
        nearest = sector_shares_df.index[(sector_shares_df.index - target_year).abs().argmin()]
        shares = sector_shares_df.loc[nearest].to_dict()

    # Baseline
    base_row = compute_sector_uplift(target_year, peak_row, cal, shares)
    d_base = base_row.d_total

    # Perturbed calibrations (shallow copy + override beta_aafs)
    def make_perturbed_cal(factor: float) -> "SCMCalibration":
        pcal = copy.copy(cal)
        pcal = dataclasses.replace(pcal, beta_aafs=cal.beta_aafs * factor)
        return pcal

    cal_hi = make_perturbed_cal(1.0 + perturbation_pct / 100.0)
    cal_lo = make_perturbed_cal(1.0 - perturbation_pct / 100.0)

    row_hi = compute_sector_uplift(target_year, peak_row, cal_hi, shares)
    row_lo = compute_sector_uplift(target_year, peak_row, cal_lo, shares)

    d_hi   = row_hi.d_total
    d_lo   = row_lo.d_total
    swing  = d_hi - d_lo
    swing_pct = (swing / d_base) * 100.0 if d_base != 0 else 0.0

    detail = {
        f"Baseline d_total {target_year} (MW)":           f"{d_base:,.1f}",
        f"β_aafs +{perturbation_pct}% d_total (MW)":      f"{d_hi:,.1f}",
        f"β_aafs -{perturbation_pct}% d_total (MW)":      f"{d_lo:,.1f}",
        f"Total swing (MW)":                               f"{swing:,.1f}",
        f"Swing as % of baseline":                         f"{swing_pct:.1f}%",
        f"High-volatility threshold":                      f">{swing_threshold_pct:.0f}%",
    }

    if swing_pct > swing_threshold_pct:
        return ValidationResult(
            check_id=4,
            name="Sensitivity Stress-Test — β_aafs ±15%",
            status=CheckStatus.WARNING,
            message=(
                f"HIGH VOLATILITY ALERT: A ±{perturbation_pct:.0f}% change in β_aafs "
                f"produces a {swing_pct:.1f}% swing in 2050 tail-risk "
                f"({swing:+,.0f} MW). Grid resilience forecasts are highly sensitive "
                f"to heat pump efficiency assumptions. "
                f"Consider widening uncertainty bounds or sourcing tighter β_aafs priors."
            ),
            detail=detail,
            halt=False,
        )
    else:
        return ValidationResult(
            check_id=4,
            name="Sensitivity Stress-Test — β_aafs ±15%",
            status=CheckStatus.PASS,
            message=(
                f"Sensitivity within acceptable range: ±{perturbation_pct:.0f}% β_aafs "
                f"→ {swing_pct:.1f}% swing in 2050 d_total "
                f"({swing:+,.0f} MW, threshold {swing_threshold_pct:.0f}%)."
            ),
            detail=detail,
            halt=False,
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_validations(
    cal: "SCMCalibration",  # noqa: F821
    trajectories: dict[str, pd.DataFrame],
    peak_tables: dict[str, pd.DataFrame],
    sector_shares_df: pd.DataFrame,
    primary_scenario: str | None = None,
    verbose: bool = True,
) -> list[ValidationResult]:
    """
    Execute all 4 validation checks in order.

    Raises PipelineHaltError if any check sets halt=True.
    Returns the list of ValidationResult objects.
    """
    scen = primary_scenario or PRIMARY_SCENARIO

    results: list[ValidationResult] = []

    # Check 1 — must pass before anything else
    r1 = check_calibration_anchor(cal)
    results.append(r1)

    if r1.halt:
        _print_report(results, verbose)
        raise PipelineHaltError(
            f"Check 1 failed: {r1.message}\n\n"
            "Failure Protocol: The residual (d_residual) is auto-computed as "
            "OBSERVED_UPLIFT - HVAC - AAFS - EV - DC.  If Check 1 fails, verify "
            "that the 2025 peak row from the CEC file matches the anchor year, "
            "and that all component formulas use the same ΔT_hub value."
        )

    # Check 2 — scan all trajectories for capacity saturation
    r2 = check_capacity_saturation(trajectories)
    results.append(r2)

    # Check 3 — structural divergence
    r3 = check_structural_divergence(trajectories, scenario=scen)
    results.append(r3)

    # Check 4 — sensitivity stress-test on primary scenario
    if scen in peak_tables:
        r4 = check_beta_aafs_sensitivity(peak_tables[scen], cal, sector_shares_df)
    else:
        pk = list(peak_tables.keys())[0]
        r4 = check_beta_aafs_sensitivity(peak_tables[pk], cal, sector_shares_df)
    results.append(r4)

    _print_report(results, verbose)
    return results


def _print_report(results: list[ValidationResult], verbose: bool) -> None:
    if not verbose:
        return
    width = 72
    print("\n" + "═" * width)
    print(" PIPELINE VALIDATION REPORT ".center(width, "═"))
    print("═" * width)
    for r in results:
        print(str(r))
        print()
    n_pass = sum(1 for r in results if r.status == CheckStatus.PASS)
    n_warn = sum(1 for r in results if r.status == CheckStatus.WARNING)
    n_fail = sum(1 for r in results if r.status == CheckStatus.FAIL)
    print(f"  Summary: {n_pass} PASS  |  {n_warn} WARNING  |  {n_fail} FAIL")
    print("═" * width + "\n")
