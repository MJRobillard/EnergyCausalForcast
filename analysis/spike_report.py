"""
Reporting layer — produces CEC/internship-friendly output files and figures.

Reads:
  - results/california/predictions.csv  (from train_california.py)
  - results/california/decomposition.csv (from train_california.py)
  - results/california/counterfactual_peaks.csv
  - results/california/spike_decomposition.csv

Writes to results/california/:
  - peak_metrics.csv
  - counterfactual_peaks.csv
  - spike_decomposition.csv
  - figures/actual_vs_predicted_peaks.png
  - figures/spike_decomposition_bar.png
  - figures/weather_uplift.png
  - figures/scenario_peak_impact.png

Run standalone:
    python analysis/spike_report.py

Or call generate_report(results_dir) from train_california.py after training.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results" / "california"
FIGURES = RESULTS / "figures"

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.counterfactual import COMPONENTS, COMPONENT_LABELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_predictions() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["period"])
    df["residual"] = df["pred_mwh"] - df["demand_mwh"]
    df["ape"]      = df["residual"].abs() / df["demand_mwh"] * 100
    df["month"]    = df["period"].dt.month
    df["hour"]     = df["period"].dt.hour
    return df


# ── Figure 1: Actual vs predicted peaks over the test year ────────────────────

def plot_actual_vs_predicted(df: pd.DataFrame, split: str = "test") -> Path:
    sub  = df[df["split"] == split].sort_values("period")
    # monthly peaks
    sub["ym"] = sub["period"].dt.to_period("M")
    mp = sub.groupby("ym")[["demand_mwh", "pred_mwh"]].max().reset_index()
    mp["ym_str"] = mp["ym"].astype(str)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"hspace": 0.4})

    # top: full time series
    ax = axes[0]
    ax.plot(sub["period"], sub["demand_mwh"], lw=0.5, alpha=0.7, label="Actual")
    ax.plot(sub["period"], sub["pred_mwh"],  lw=0.5, alpha=0.7, label="Predicted")
    top1 = sub["demand_mwh"].quantile(0.99)
    ax.axhline(top1, ls="--", lw=0.8, color="red", alpha=0.6, label="99th pct")
    ax.set_title(f"Actual vs Predicted — {split.capitalize()} Period", fontsize=11)
    ax.set_ylabel("Demand (MW)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # bottom: monthly peak bar chart
    ax2 = axes[1]
    x  = np.arange(len(mp))
    w  = 0.35
    ax2.bar(x - w/2, mp["demand_mwh"] / 1000, w, label="Actual peak", color="steelblue", alpha=0.85)
    ax2.bar(x + w/2, mp["pred_mwh"]   / 1000, w, label="Predicted peak", color="coral", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(mp["ym_str"], rotation=45, ha="right", fontsize=8)
    ax2.set_title("Monthly Peak: Actual vs Predicted", fontsize=11)
    ax2.set_ylabel("Peak Demand (GW)")
    ax2.legend(fontsize=8)

    out = FIGURES / "actual_vs_predicted_peaks.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


# ── Figure 2: Spike decomposition bar chart ───────────────────────────────────

def plot_spike_decomposition(decomp_path: Path) -> Path | None:
    if not decomp_path.exists():
        return None
    uplift = pd.read_csv(decomp_path)
    if "uplift_mw" not in uplift.columns:
        return None

    uplift = uplift.sort_values("uplift_mw", ascending=True)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in uplift["uplift_mw"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(uplift["label"], uplift["uplift_mw"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Demand Uplift: Spike Hours vs Matched Normal Hours\n"
                 "(positive = drives demand higher during spikes)", fontsize=11)
    ax.set_xlabel("Mean MW difference (spike − normal)")
    for bar, val in zip(bars, uplift["uplift_mw"]):
        ax.text(val + (50 if val >= 0 else -50), bar.get_y() + bar.get_height()/2,
                f"{val:+.0f} MW", va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    total = uplift["total_demand_uplift_mw"].iloc[0]
    ax.set_title(ax.get_title() + f"\nTotal observed uplift: {total:+,.0f} MW", fontsize=10)

    out = FIGURES / "spike_decomposition_bar.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


# ── Figure 3: Scenario peak impact ───────────────────────────────────────────

def plot_scenario_impact(cf_path: Path) -> Path | None:
    if not cf_path.exists():
        return None
    cf = pd.read_csv(cf_path)
    if "delta_peak_mw" not in cf.columns:
        return None

    cf = cf[cf["scenario"] != "observed"].copy()
    cf = cf.sort_values("delta_peak_mw")
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in cf["delta_peak_mw"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cf["scenario"], cf["delta_peak_mw"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Scenario Analysis: Change in Predicted Annual Peak Demand", fontsize=11)
    ax.set_xlabel("ΔPeak (MW vs observed weather)")

    out = FIGURES / "scenario_peak_impact.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


# ── Figure 4: Weather uplift scatter ─────────────────────────────────────────

def plot_weather_uplift(df: pd.DataFrame, split: str = "test") -> Path:
    sub = df[df["split"] == split].copy()
    top1_thresh = sub["demand_mwh"].quantile(0.99)
    sub["is_spike"] = sub["demand_mwh"] >= top1_thresh

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, xlabel in [
        (axes[0], "temperature_f", "Composite Temperature (°F)"),
        (axes[1], "temp_max",      "Regional Max Temperature (°F)"),
    ]:
        if col not in sub.columns:
            continue
        normal = sub[~sub["is_spike"]]
        spike  = sub[sub["is_spike"]]
        ax.scatter(normal[col], normal["demand_mwh"] / 1000,
                   s=1, alpha=0.1, color="steelblue", label="Normal hours")
        ax.scatter(spike[col],  spike["demand_mwh"]  / 1000,
                   s=12, alpha=0.6, color="red", label="Top-1% hours", zorder=5)
        ax.set_xlabel(xlabel); ax.set_ylabel("Demand (GW)")
        ax.set_title(f"Demand vs {xlabel.split('(')[0].strip()}")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig.suptitle("Weather vs Demand: Spike Hours vs Normal Hours", fontsize=12)
    out = FIGURES / "weather_uplift.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


# ── Console summary ───────────────────────────────────────────────────────────

def print_uplift_summary(uplift_df: pd.DataFrame) -> None:
    total = uplift_df["total_demand_uplift_mw"].iloc[0]
    print(f"\n{'─'*60}")
    print(f"  Spike vs Normal Demand Uplift  (total: {total:+,.0f} MW)")
    print(f"{'─'*60}")
    print(f"  {'Component':<28}  {'Uplift MW':>10}  {'% of total':>10}")
    for _, row in uplift_df.iterrows():
        bar = "▓" * int(abs(row["uplift_pct_of_total"]) / 3)
        print(f"  {row['label']:<28}  {row['uplift_mw']:>+10,.0f}  "
              f"{row['uplift_pct_of_total']:>9.1f}%  {bar}")


def print_scenario_summary(cf_df: pd.DataFrame) -> None:
    print(f"\n{'─'*68}")
    print(f"  Scenario Analysis — Peak Demand Impact")
    print(f"{'─'*68}")
    print(f"  {'Scenario':<25}  {'Peak MW':>9}  {'ΔPeak MW':>10}  {'Top-1% ΔMW':>12}")
    for _, row in cf_df.iterrows():
        print(f"  {row['scenario']:<25}  {row['peak_mw']:>9,.0f}  "
              f"{row.get('delta_peak_mw', 0):>+10,.0f}  "
              f"{row.get('delta_top1_mw', 0):>+12,.0f}")


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(results_dir: Path | None = None) -> None:
    """Generate all figures and print summaries from saved result files."""
    global RESULTS, FIGURES
    if results_dir is not None:
        RESULTS = Path(results_dir)
        FIGURES = RESULTS / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("\n=== Spike Report ===")

    df = _load_predictions()

    out1 = plot_actual_vs_predicted(df)
    print(f"  Saved: {out1.name}")

    out2 = plot_weather_uplift(df)
    print(f"  Saved: {out2.name}")

    decomp_path = RESULTS / "spike_decomposition.csv"
    out3 = plot_spike_decomposition(decomp_path)
    if out3:
        print(f"  Saved: {out3.name}")
        print_uplift_summary(pd.read_csv(decomp_path))

    cf_path = RESULTS / "counterfactual_peaks.csv"
    out4 = plot_scenario_impact(cf_path)
    if out4:
        print(f"  Saved: {out4.name}")
        print_scenario_summary(pd.read_csv(cf_path))


if __name__ == "__main__":
    generate_report()
