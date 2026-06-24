"""
Sectoral SCM Heatwave Pipeline — Main Orchestrator.

Usage
─────
    python -m pipeline.run                    # full run, all outputs
    python -m pipeline.run --scenario LocalRel
    python -m pipeline.run --no-plots         # skip figures
    python -m pipeline.run --validate-only    # run validations then exit

Pipeline stages
───────────────
1. Load CEC scenario data → hourly + annual-peak tables
2. Load weather → hub composite, Riverside, climatology, 2022 peak
3. Derive sector shares (historical + projected 2025–2050)
4. Calibrate SCM (pooled + sector β coefficients, residual)
5. Compute baseline sectoral trajectories for all CEC scenarios
6. Run all 4 validation checks → halt on fatal failure
7. Build and run policy counterfactuals
8. Emit outputs: summary tables (CSV) + figures (PNG)

Exit codes
──────────
0  — success (all checks PASS or WARNING)
1  — PipelineHaltError (Check 1 failure)
2  — unexpected exception
"""

from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from pipeline.config import PRIMARY_SCENARIO, RESULTS, ANALYSIS
from pipeline.data.cec_loader import load_all_scenarios
from pipeline.data.sector_shares import project_sector_shares, anchor_shares_2025
from pipeline.data.weather import load_weather_for_pipeline
from pipeline.scm.calibration import calibrate
from pipeline.scm.trajectories import compute_all_trajectories
from pipeline.counterfactuals.engine import run_counterfactual
from pipeline.counterfactuals.scenarios import build_scenarios
from pipeline.validation import run_all_validations, PipelineHaltError
from pipeline.outputs.tables import (
    baseline_summary_table,
    sector_attribution_table,
    counterfactual_matrix,
    validation_summary_table,
)
from pipeline.outputs.plots import save_all_figures


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sectoral SCM heatwave pipeline for California CAISO (2025–2050)"
    )
    p.add_argument(
        "--scenario", default=PRIMARY_SCENARIO,
        help="Primary CEC scenario for CF analysis (default: LocalRel)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation (useful for headless CI runs)",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Run validations then exit without producing output files",
    )
    p.add_argument(
        "--out-dir", type=Path, default=ANALYSIS,
        help="Output directory for figures and CSV tables",
    )
    return p


def run(
    scenario: str = PRIMARY_SCENARIO,
    produce_plots: bool = True,
    validate_only: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """
    Execute the full pipeline and return a results dict.

    Returns
    -------
    {
        "calibration":   SCMCalibration,
        "trajectories":  {scenario: DataFrame},
        "cf_results":    {cf_label: DataFrame},
        "validation":    [ValidationResult, ...],
        "tables":        {name: DataFrame},
    }
    """
    out_dir = out_dir or ANALYSIS

    # ── Stage 1: Load CEC data ────────────────────────────────────────────
    print("Stage 1/8 — Loading CEC scenarios…")
    _, peak_tables = load_all_scenarios()
    print(f"  Loaded {len(peak_tables)} scenarios: {list(peak_tables.keys())}")

    # ── Stage 2: Load weather ─────────────────────────────────────────────
    print("Stage 2/8 — Loading weather data…")
    hw_peak = load_weather_for_pipeline()
    print(
        f"  2022 heatwave peak: "
        f"T_hub={hw_peak.T_hub:.1f}°F (ΔT={hw_peak.dT_hub:+.1f}°F), "
        f"T_rv={hw_peak.T_rv:.1f}°F (ΔT={hw_peak.dT_rv:+.1f}°F)"
    )

    # ── Stage 3: Sector shares ────────────────────────────────────────────
    print("Stage 3/8 — Deriving sector shares 2025–2050…")
    anchor   = anchor_shares_2025()
    proj_yrs = list(range(2025, 2051))
    sector_shares_df = project_sector_shares(proj_yrs, anchor_shares=anchor)
    print(
        f"  2025 anchor: RES={anchor['RES']:.3f}, "
        f"COM={anchor['COM']:.3f}, IND={anchor['IND']:.3f}"
    )

    # ── Stage 4: Calibration ──────────────────────────────────────────────
    print("Stage 4/8 — Calibrating sectoral SCM…")
    primary_peak_2025 = peak_tables[scenario].loc[2025]
    sector_shares_2025 = sector_shares_df.loc[2025].to_dict()

    cal = calibrate(primary_peak_2025, hw_peak, sector_shares_2025)
    print(
        f"  HVAC structural 2025: {cal.hvac_structural_2025:+,.0f} MW  |  "
        f"Residual (lag): {cal.d_residual:+,.0f} MW"
    )
    print(
        f"  β_RES={cal.beta['RES']:.6f}  "
        f"β_COM={cal.beta['COM']:.6f}  "
        f"β_IND={cal.beta['IND']:.6f}  (per MW/°F)"
    )

    # ── Stage 5: Baseline trajectories ───────────────────────────────────
    print("Stage 5/8 — Computing baseline sectoral trajectories…")
    trajectories = compute_all_trajectories(peak_tables, cal, sector_shares_df)
    primary_traj = trajectories[scenario]
    print(
        f"  {scenario} 2050 δMNL: {primary_traj.loc[2050, 'd_total']:,.0f} MW  "
        f"(vs static {primary_traj.loc[2050, 'd_static']:,.0f} MW, "
        f"Δ={primary_traj.loc[2050, 'vs_static']:+,.0f} MW)"
    )

    # ── Stage 6: Validations ──────────────────────────────────────────────
    print("Stage 6/8 — Running validation checks…")
    validation_results = run_all_validations(
        cal=cal,
        trajectories=trajectories,
        peak_tables=peak_tables,
        sector_shares_df=sector_shares_df,
    )

    if validate_only:
        print("  --validate-only: stopping after validation.")
        return {
            "calibration":  cal,
            "trajectories": trajectories,
            "validation":   validation_results,
            "tables":       {},
            "cf_results":   {},
        }

    # ── Stage 7: Counterfactuals ─────────────────────────────────────────
    print("Stage 7/8 — Running policy counterfactuals…")
    cf_specs = build_scenarios(cal, peak_tables, primary_scenario=scenario)
    cf_results: dict[str, pd.DataFrame] = {}

    for cf_label, spec in cf_specs.items():
        cf_df = run_counterfactual(
            spec=spec,
            peak_tbl=peak_tables[scenario],
            cal=cal,
            sector_shares_df=sector_shares_df,
            baseline_traj=primary_traj,
        )
        cf_results[cf_label] = cf_df
        d_2050 = cf_df.loc[2050, "d_total"] if 2050 in cf_df.index else float("nan")
        vs_bl  = cf_df.loc[2050, "vs_baseline"] if 2050 in cf_df.index else float("nan")
        print(f"  {cf_label}: 2050 δMNL = {d_2050:,.0f} MW  ({vs_bl:+,.0f} vs baseline)")

    # ── Stage 8: Outputs ─────────────────────────────────────────────────
    print("Stage 8/8 — Writing outputs…")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "baseline_summary":     baseline_summary_table(primary_traj),
        "sector_attribution":   sector_attribution_table(primary_traj),
        "counterfactual_matrix": counterfactual_matrix(primary_traj, cf_results),
        "validation":           validation_summary_table(validation_results),
    }

    for name, df in tables.items():
        csv_path = out_dir / f"scm_sectoral_{name}.csv"
        df.to_csv(csv_path)
        print(f"  CSV: {csv_path.name}")

    if produce_plots:
        save_all_figures(
            trajectories=trajectories,
            primary_traj=primary_traj,
            cf_trajectories=cf_results,
            validation_results=validation_results,
            out_dir=out_dir,
        )

    _print_policy_matrix(primary_traj, cf_results)

    return {
        "calibration":  cal,
        "trajectories": trajectories,
        "cf_results":   cf_results,
        "validation":   validation_results,
        "tables":       tables,
    }


def _print_policy_matrix(
    baseline: pd.DataFrame,
    cf_results: dict[str, pd.DataFrame],
) -> None:
    """Print the policy matrix console summary."""
    display_years = [y for y in [2030, 2035, 2040, 2045, 2050] if y in baseline.index]
    width = 80

    print("\n" + "─" * width)
    print(" POLICY MATRIX — Causal δMNL (MW) ".center(width, "─"))
    print("─" * width)

    header = f"{'Scenario':<40}" + "".join(f"{y:>8}" for y in display_years)
    print(header)
    print("─" * width)

    print(f"{'Baseline (causal)':<40}" + "".join(
        f"{baseline.loc[y, 'd_total']:>8,.0f}" for y in display_years
    ))
    print(f"{'Static 4,625 MW anchor':<40}" + "".join(
        f"{4625:>8,}" for _ in display_years
    ))
    print("─" * width)

    for label, df in cf_results.items():
        d_row  = "".join(f"{df.loc[y, 'd_total']:>8,.0f}" for y in display_years if y in df.index)
        vb_row = "".join(f"{df.loc[y, 'vs_baseline']:>+8,.0f}" for y in display_years if y in df.index)
        short  = label.split(":")[0]
        print(f"{short:<40}" + d_row)
        print(f"  {'→ vs baseline':<38}" + vb_row)

    print("─" * width)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run(
            scenario=args.scenario,
            produce_plots=not args.no_plots,
            validate_only=args.validate_only,
            out_dir=args.out_dir,
        )
        return 0

    except PipelineHaltError as e:
        print(f"\n[PIPELINE HALT] {e}", file=sys.stderr)
        return 1

    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
