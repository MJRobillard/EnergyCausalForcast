"""
Plotting layer for the sectoral SCM pipeline.

Figures produced
────────────────
fig_sector_trajectories   Stacked area: sector contribution to δMNL over time
fig_causal_vs_static      Lines: causal total vs static 4,625 MW baseline
fig_counterfactual_matrix Bar chart: 2050 δMNL under each CF vs baseline
fig_sector_shares         Stacked bar: RES/COM/IND/DC/LAG share evolution
fig_sensitivity_tornado   Tornado chart for β_aafs sensitivity
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pipeline.config import ANALYSIS, OBSERVED_2022_UPLIFT_MW

YEARS = list(range(2025, 2051))
PALETTE = {
    "RES": "#2196F3",  # blue
    "COM": "#FF9800",  # orange
    "IND": "#9C27B0",  # purple
    "DC":  "#F44336",  # red
    "LAG": "#9E9E9E",  # grey
}
CF_COLORS = {
    "CF1":              "#E53935",
    "CF2a":             "#43A047",
    "CF2b":             "#00897B",
    "CF3":              "#FB8C00",
    "CF_RES_PRECOOL":   "#1E88E5",
    "CF_IND_INTERRUPT": "#8E24AA",
    "CF_TOU_SHOCK":     "#F06292",
    "CF_DC_PUE":        "#6D4C41",
    "CF_DC_SHIFT":      "#FF7043",
    "CF_EV_FLEET_DR":   "#26A69A",
    "CF_HEAVY_EV_DR":   "#5C6BC0",
    "CF_INDUSTRY_STACK": "#00838F",
    "CF5":              "#2E7D32",
}


def _save(fig: plt.Figure, name: str, out_dir: Path | None) -> plt.Figure:
    """Save figure to disk if out_dir is set; leave figure open for notebook display."""
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path.name}")
    else:
        plt.show()
        plt.close(fig)
    return fig


def show_figures() -> None:
    """Render all open figures inline (Jupyter / Cursor) then close them."""
    import matplotlib._pylab_helpers as pylab_helpers

    managers = list(pylab_helpers.Gcf.get_all_fig_managers())
    if not managers:
        return

    try:
        from IPython.display import display
        for mgr in managers:
            display(mgr.canvas.figure)
    except Exception:
        plt.show()
    plt.close("all")


def fig_sector_trajectories(
    traj: pd.DataFrame,
    out_dir: Path | None = ANALYSIS,
    title_suffix: str = "",
) -> None:
    """
    Stacked area chart: per-sector + DC + lag contribution to δMNL.
    """
    df = traj.copy()
    years = df.index.tolist()

    stacks = {
        "RES": df["d_hvac_res"] + df["d_aafs_res"] + df["d_ev_res"],
        "COM": df["d_hvac_com"] + df["d_aafs_com"] + df["d_ev_com"],
        "IND": df["d_hvac_ind"] + df["d_ev_ind"],
        "DC":  df["d_dc"],
        "LAG": df["d_residual"],
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(years))
    for label, series in stacks.items():
        vals = series.values
        ax.fill_between(years, bottom, bottom + vals,
                        alpha=0.82, color=PALETTE[label], label=label)
        bottom += vals

    ax.axhline(OBSERVED_2022_UPLIFT_MW, ls="--", color="black", lw=1.4,
               label=f"Static anchor ({OBSERVED_2022_UPLIFT_MW:,} MW)")
    ax.plot(years, df["d_total"], color="black", lw=2, label="Total δMNL (causal)")

    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year")
    ax.set_ylabel("Heatwave demand uplift (MW)")
    ax.set_title(f"Sectoral SCM: Causal δMNL Components 2025–2050{title_suffix}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "fig_sector_trajectories", out_dir)


def fig_causal_vs_static(
    trajectories: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
) -> None:
    """
    Multi-scenario line chart: causal total vs static baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    scen_styles = {
        "Planning":  ("C0", "-"),
        "LocalRel":  ("C1", "--"),
        "PlusKnown": ("C2", "-."),
    }

    # Left: absolute δMNL
    ax = axes[0]
    for name, df in trajectories.items():
        style = scen_styles.get(name, ("C3", ":"))
        ax.plot(df.index, df["d_total"], color=style[0], ls=style[1],
                lw=2, label=f"{name} (causal)")
    ax.axhline(OBSERVED_2022_UPLIFT_MW, ls="--", color="black", lw=1.2,
               label="Static 4,625 MW")
    ax.set_title("Total causal δMNL across CEC scenarios")
    ax.set_ylabel("δMNL (MW)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Right: excess vs static
    ax = axes[1]
    for name, df in trajectories.items():
        style = scen_styles.get(name, ("C3", ":"))
        ax.plot(df.index, df["vs_static"], color=style[0], ls=style[1],
                lw=2, label=name)
    ax.axhline(0, ls="-", color="black", lw=0.8)
    ax.set_title("Causal excess over static 4,625 MW baseline")
    ax.set_ylabel("Excess δMNL vs static (MW)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))

    for ax in axes:
        ax.set_xlabel("Year")
    fig.tight_layout()
    _save(fig, "fig_causal_vs_static", out_dir)


def fig_counterfactual_matrix(
    baseline: pd.DataFrame,
    cf_trajectories: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
    target_year: int = 2050,
) -> None:
    """
    Horizontal bar chart: each CF's δMNL delta vs baseline at target_year.
    Sorted by MW impact.
    """
    if target_year not in baseline.index:
        return

    d_base = baseline.loc[target_year, "d_total"]
    deltas = {}
    for label, df in cf_trajectories.items():
        if target_year in df.index and "vs_baseline" in df.columns:
            deltas[label] = df.loc[target_year, "vs_baseline"]

    sorted_labels = sorted(deltas, key=lambda k: deltas[k])
    values = [deltas[k] for k in sorted_labels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_labels) * 0.7)))
    colors = [CF_COLORS.get(k.split(":")[0].strip(), "#78909C") for k in sorted_labels]
    bars = ax.barh(sorted_labels, values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            val + (8 if val >= 0 else -8), bar.get_y() + bar.get_height() / 2,
            f"{val:+,.0f} MW", va="center", ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel(f"MW change vs baseline in {target_year}")
    ax.set_title(f"Policy counterfactual impact on heatwave δMNL ({target_year})\n"
                 f"Baseline: {d_base:,.0f} MW")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
    fig.tight_layout()
    _save(fig, "fig_counterfactual_matrix", out_dir)


def fig_sector_shares(
    traj: pd.DataFrame,
    out_dir: Path | None = ANALYSIS,
) -> None:
    """
    100% stacked bar: how each sector's share of total δMNL evolves.
    """
    display_years = [y for y in [2025, 2030, 2035, 2040, 2045, 2050] if y in traj.index]
    df = traj.loc[display_years].copy()

    totals = df["d_total"]
    stacks = {
        "RES": (df["d_hvac_res"] + df["d_aafs_res"] + df["d_ev_res"]) / totals * 100,
        "COM": (df["d_hvac_com"] + df["d_aafs_com"] + df["d_ev_com"]) / totals * 100,
        "IND": (df["d_hvac_ind"] + df["d_ev_ind"]) / totals * 100,
        "DC":  df["d_dc"] / totals * 100,
        "LAG": df["d_residual"] / totals * 100,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(display_years))
    bottom = np.zeros(len(display_years))

    for label, pct in stacks.items():
        ax.bar(x, pct.values, bottom=bottom, color=PALETTE[label],
               label=label, alpha=0.88, edgecolor="white")
        # Annotate bars > 5%
        for xi, (b, v) in enumerate(zip(bottom, pct.values)):
            if v > 5:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += pct.values

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(y) for y in display_years])
    ax.set_ylabel("Share of total δMNL (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Sector attribution of heatwave demand uplift")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_sector_shares", out_dir)


def fig_sensitivity_tornado(
    sensitivity_detail: dict,
    out_dir: Path | None = ANALYSIS,
    target_year: int = 2050,
) -> None:
    """
    Tornado chart for parameter sensitivity from Check 4.

    sensitivity_detail : the 'detail' dict from the ValidationResult for Check 4.
    """
    try:
        d_base = float(sensitivity_detail.get(f"Baseline d_total {target_year} (MW)", "0").replace(",", ""))
        d_hi_str = [v for k, v in sensitivity_detail.items() if "+15%" in k or "+15.0%" in k]
        d_lo_str = [v for k, v in sensitivity_detail.items() if "-15%" in k or "-15.0%" in k]
        if not d_hi_str or not d_lo_str:
            return
        d_hi = float(d_hi_str[0].replace(",", ""))
        d_lo = float(d_lo_str[0].replace(",", ""))
    except (ValueError, IndexError):
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(["β_aafs"], [d_hi - d_base], left=d_base, color="#EF5350", label="+15%")
    ax.barh(["β_aafs"], [d_lo - d_base], left=d_base, color="#42A5F5", label="−15%")
    ax.axvline(d_base, color="black", lw=1.5, ls="--", label=f"Baseline {d_base:,.0f} MW")
    ax.set_xlabel(f"2050 δMNL (MW)")
    ax.set_title(f"Sensitivity: ±15% β_aafs → 2050 heatwave uplift\n"
                 f"Swing: {d_hi-d_lo:+,.0f} MW  ({(d_hi-d_lo)/d_base*100:.1f}% of baseline)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_sensitivity_tornado", out_dir)


def fig_case_stress_band(
    cases: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
) -> None:
    """Best / average / worst stress_total trajectories with catastrophic reference."""
    from pipeline.scenario_cases import HIST_CATASTROPHIC_MW

    styles = {
        "best":     ("#43A047", "-",  1.8),
        "average":  ("#1E88E5", "-",  2.2),
        "worst":    ("#E53935", "-",  2.5),
        "envelope": ("#E53935", ":",  1.5),
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    years = list(range(2025, 2051))

    for key, df in cases.items():
        if key not in styles:
            continue
        color, ls, lw = styles[key]
        label = df["case_label"].iloc[0] if "case_label" in df.columns else key
        ax.plot(df.index, df["stress_total"], color=color, ls=ls, lw=lw, label=label)

    if "best" in cases and "worst" in cases:
        b, w = cases["best"], cases["worst"]
        common = b.index.intersection(w.index)
        ax.fill_between(
            common, b.loc[common, "stress_total"], w.loc[common, "stress_total"],
            alpha=0.12, color="#E53935", label="Best–worst band",
        )

    ax.axhline(HIST_CATASTROPHIC_MW, color="black", ls="--", lw=1.4,
               label=f"Sep 7 2022 catastrophic ({HIST_CATASTROPHIC_MW:,} MW)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total stress under 2022 heatwave (MNL + δMNL, MW)")
    ax.set_title("Best / Average / Worst Case — Heatwave Stress Trajectories")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "fig_case_stress_band", out_dir)


def fig_headroom_erosion(
    cases: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
) -> None:
    """Headroom vs 2022 catastrophic level by case."""
    from pipeline.scenario_cases import HIST_CATASTROPHIC_MW

    fig, ax = plt.subplots(figsize=(12, 5))
    styles = {"best": "#43A047", "average": "#1E88E5", "worst": "#E53935", "envelope": "#B71C1C"}
    for key, df in cases.items():
        if key not in styles:
            continue
        ax.plot(df.index, df["headroom_mw"], color=styles[key], lw=2,
                label=df["case_label"].iloc[0] if "case_label" in df.columns else key)

    ax.axhline(0, color="black", lw=1.0)
    ax.fill_between(range(2025, 2051), 0, -8000, alpha=0.06, color="red")
    ax.text(2026, -2500, "Exceeds 2022 catastrophic (no weather anomaly needed)",
            fontsize=8, color="darkred", style="italic")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Headroom below {HIST_CATASTROPHIC_MW:,} MW (MW)")
    ax.set_title("Headroom Erosion Under 2022-Class Heatwave")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))
    fig.tight_layout()
    _save(fig, "fig_headroom_erosion", out_dir)


def fig_mitigation_waterfall(
    worst_baseline: pd.DataFrame,
    cf_results: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
    target_year: int = 2035,
) -> None:
    """Waterfall: worst-case stress reduction per policy at target year."""
    if target_year not in worst_baseline.index:
        return

    base_stress = float(worst_baseline.loc[target_year, "stress_total"])
    items = [("Worst baseline", base_stress)]
    for label, df in cf_results.items():
        if target_year not in df.index:
            continue
        stress = float(df.loc[target_year, "mnl_hw"])
        short = label.split(":")[0] if ":" in label else label
        items.append((short, stress))

    # Combined if present
    names = [x[0] for x in items]
    vals = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#E53935"] + ["#43A047"] * (len(vals) - 1)
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(base_stress, ls="--", color="gray", lw=1)
    ax.set_ylabel(f"Total stress MW ({target_year})")
    ax.set_title(f"Mitigation Impact on Worst-Case Heatwave Stress ({target_year})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_mitigation_waterfall", out_dir)


def fig_uplift_method_comparison(
    method_trajs: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
    case_label: str = "Worst case",
) -> None:
    """Compare static 2022, fleet-scaled, and year-native δMNL trajectories."""
    styles = {
        "static_2022":   ("#9E9E9E", "--", 1.5, "Static 4,625 MW"),
        "fleet_scaled":  ("#FB8C00", "-.", 2.0, "Fleet-scaled (ΔT₂₀₂₂)"),
        "year_native":   ("#E53935", "-",  2.5, "Year-native SCM"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for key, df in method_trajs.items():
        if key not in styles:
            continue
        color, ls, lw, label = styles[key]
        ax.plot(df.index, df["d_total"], color=color, ls=ls, lw=lw, label=label)
    ax.axhline(OBSERVED_2022_UPLIFT_MW, ls=":", color="black", lw=1.0)
    ax.set_title(f"δMNL by uplift method — {case_label}")
    ax.set_ylabel("Heatwave uplift δMNL (MW)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax = axes[1]
    if "year_native" in method_trajs and "static_2022" in method_trajs:
        yn = method_trajs["year_native"]
        st = method_trajs["static_2022"]
        common = yn.index.intersection(st.index)
        excess = yn.loc[common, "d_total"] - st.loc[common, "d_total"]
        ax.fill_between(common, 0, excess, alpha=0.35, color="#E53935",
                        label="Year-native excess vs static")
        ax.plot(common, excess, color="#B71C1C", lw=2)
        for yr in [2035, 2040, 2050]:
            if yr in excess.index:
                ax.annotate(f"{excess.loc[yr]:+,.0f}",
                            (yr, excess.loc[yr]), fontsize=8, ha="center", va="bottom")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Year-native excess above static 2022 anchor")
    ax.set_ylabel("Additional δMNL (MW)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+,.0f}"))

    for ax in axes:
        ax.set_xlabel("Year")
    fig.tight_layout()
    _save(fig, "fig_uplift_method_comparison", out_dir)


def fig_industry_policy_matrix(
    baseline: pd.DataFrame,
    industry_cfs: dict[str, pd.DataFrame],
    out_dir: Path | None = ANALYSIS,
    target_year: int = 2035,
) -> None:
    """Bar chart of industry-specific policy impacts on total stress."""
    if target_year not in baseline.index:
        return

    base_stress = float(baseline.loc[target_year, "stress_total"])
    items = [("Worst baseline", base_stress, "#E53935")]
    for label, df in industry_cfs.items():
        if target_year not in df.index:
            continue
        short = label.split(":")[0].strip()
        stress = float(df.loc[target_year, "stress_total"])
        color = CF_COLORS.get(short, "#78909C")
        items.append((short, stress, color))

    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(base_stress, ls="--", color="gray", lw=1)
    for bar, val in zip(bars, vals):
        delta = val - base_stress
        ax.text(bar.get_x() + bar.get_width() / 2, val + 80,
                f"{val:,.0f}\n({delta:+,.0f})", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(f"Total stress MW ({target_year})")
    ax.set_title(f"Industry Policy Interventions — Worst Case ({target_year})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_industry_policy_matrix", out_dir)


def save_all_figures(
    trajectories: dict[str, pd.DataFrame],
    primary_traj: pd.DataFrame,
    cf_trajectories: dict[str, pd.DataFrame],
    validation_results: list,
    out_dir: Path | None = ANALYSIS,
) -> None:
    """Convenience: produce all figures in one call."""
    print("Saving figures…")
    fig_sector_trajectories(primary_traj, out_dir)
    fig_causal_vs_static(trajectories, out_dir)
    fig_counterfactual_matrix(primary_traj, cf_trajectories, out_dir)
    fig_sector_shares(primary_traj, out_dir)

    # Sensitivity tornado from Check 4 detail
    for r in validation_results:
        if r.check_id == 4 and r.detail:
            fig_sensitivity_tornado(r.detail, out_dir)
            break
