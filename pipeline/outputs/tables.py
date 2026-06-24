"""
Summary tables for the sectoral SCM pipeline.

Exposes
-------
baseline_summary_table(traj)        -> yearly δMNL by component
sector_attribution_table(traj)      -> RES/COM/IND/DC/LAG attribution
counterfactual_matrix(baseline, cfs) -> CF comparison table
validation_summary_table(results)   -> validation check status table
"""

from __future__ import annotations
import pandas as pd

from pipeline.validation import ValidationResult


DISPLAY_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]


def baseline_summary_table(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Year × component table showing MW contributions to total δMNL.

    Mirrors §6 of the original notebook but with sector breakdown.
    """
    cols = {
        "HVAC-RES":   "d_hvac_res",
        "HVAC-COM":   "d_hvac_com",
        "HVAC-IND":   "d_hvac_ind",
        "AAFS-RES":   "d_aafs_res",
        "AAFS-COM":   "d_aafs_com",
        "EV-RES":     "d_ev_res",
        "EV-COM":     "d_ev_com",
        "EV-IND":     "d_ev_ind",
        "DC thermal": "d_dc",
        "Lag/Behav":  "d_residual",
        "TOTAL":      "d_total",
        "vs Static":  "vs_static",
    }
    available_years = [y for y in DISPLAY_YEARS if y in traj.index]
    out = traj.loc[available_years, list(cols.values())].copy()
    out.columns = list(cols.keys())
    return out.round(1)


def sector_attribution_table(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Sector-level attribution: total MW by sector per year.

    Columns: RES, COM, IND, DC, LAG, TOTAL
    """
    available_years = [y for y in DISPLAY_YEARS if y in traj.index]
    df = traj.loc[available_years].copy()

    out = pd.DataFrame(index=available_years)
    out["RES"]   = df["d_hvac_res"] + df["d_aafs_res"] + df["d_ev_res"]
    out["COM"]   = df["d_hvac_com"] + df["d_aafs_com"] + df["d_ev_com"]
    out["IND"]   = df["d_hvac_ind"] + df["d_ev_ind"]
    out["DC"]    = df["d_dc"]
    out["LAG"]   = df["d_residual"]
    out["TOTAL"] = df["d_total"]

    # Share columns
    for s in ["RES", "COM", "IND", "DC", "LAG"]:
        out[f"{s}_pct"] = (out[s] / out["TOTAL"] * 100).round(1)

    return out.round(1)


def counterfactual_matrix(
    baseline: pd.DataFrame,
    cf_trajectories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Comparison table: baseline and each CF's d_total across display years.
    Also shows MW delta vs baseline.
    """
    available_years = [y for y in DISPLAY_YEARS if y in baseline.index]
    rows = {"Baseline (LocalRel)": baseline.loc[available_years, "d_total"]}

    for label, cf_df in cf_trajectories.items():
        rows[label] = cf_df.loc[available_years, "d_total"]
        rows[f"  Δ {label}"] = cf_df.loc[available_years, "vs_baseline"]

    df = pd.DataFrame(rows, index=available_years).T
    return df.round(0)


def validation_summary_table(results: list[ValidationResult]) -> pd.DataFrame:
    """
    Compact table of validation outcomes for inclusion in reports.
    """
    records = []
    for r in results:
        records.append({
            "Check": r.check_id,
            "Name": r.name,
            "Status": r.status.value,
            "Halt": "YES" if r.halt else "no",
            "Message (truncated)": r.message[:120] + ("…" if len(r.message) > 120 else ""),
        })
    return pd.DataFrame(records).set_index("Check")
