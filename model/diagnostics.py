"""
Error diagnostics for the SCM predictions.

Reads results/california/predictions.csv and produces:
  - MAPE breakdowns by hour, month, day-of-week, weekend/weekday, holiday
  - Residual distribution by temperature bin
  - Peak-day error (top 5% and top 10 load hours)
  - Residual plots: time series, vs hour, vs temp, vs solar, autocorrelation
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results" / "california"

US_HOLIDAYS = {
    "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-10",
    "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-11",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-11",
    "2025-11-27", "2025-12-25",
}

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


def load_preds() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["period"])
    df["residual"]  = df["pred_mwh"] - df["demand_mwh"]
    df["abs_err"]   = df["residual"].abs()
    df["pct_err"]   = df["abs_err"] / df["demand_mwh"] * 100
    df["hour"]      = df["period"].dt.hour
    df["month"]     = df["period"].dt.month
    df["dow"]       = df["period"].dt.dayofweek          # 0=Mon
    df["weekend"]   = (df["dow"] >= 5).astype(int)
    df["date_str"]  = df["period"].dt.strftime("%Y-%m-%d")
    df["holiday"]   = df["date_str"].isin(US_HOLIDAYS).astype(int)
    # temperature bins
    bins   = [0, 50, 60, 70, 80, 90, 200]
    labels = ["<50", "50-60", "60-70", "70-80", "80-90", "90+"]
    df["temp_bin"] = pd.cut(df["temperature_f"], bins=bins, labels=labels)
    return df


def print_breakdown(df: pd.DataFrame, col: str, label: str,
                    name_map=None, split="test") -> pd.DataFrame:
    sub = df[df["split"] == split]
    grp = sub.groupby(col).agg(
        mape=("pct_err", "mean"),
        mae=("abs_err", "mean"),
        n=("pct_err", "count"),
    ).reset_index()
    if name_map:
        grp[col] = grp[col].map(name_map) if isinstance(name_map, dict) \
                   else [name_map[int(v)] for v in grp[col]]
    print(f"\n{'─'*52}")
    print(f"  {label}  [{split}]")
    print(f"{'─'*52}")
    print(f"  {col:<12}  {'MAPE%':>7}  {'MAE MW':>8}  {'N':>6}")
    for _, row in grp.iterrows():
        print(f"  {str(row[col]):<12}  {row['mape']:>7.2f}  {row['mae']:>8.0f}  {row['n']:>6,}")
    return grp


def print_peak_errors(df: pd.DataFrame, split="test"):
    sub = df[df["split"] == split].copy()
    n5  = max(1, int(len(sub) * 0.05))
    top5_thresh = sub["demand_mwh"].quantile(0.95)

    print(f"\n{'─'*52}")
    print(f"  Peak-load error diagnostics  [{split}]")
    print(f"{'─'*52}")

    all_mape = sub["pct_err"].mean()
    top5     = sub[sub["demand_mwh"] >= top5_thresh]
    top10    = sub.nlargest(10, "demand_mwh")

    print(f"  All hours          MAPE={all_mape:.2f}%  MAE={sub['abs_err'].mean():.0f} MW")
    print(f"  Top 5% load hours  MAPE={top5['pct_err'].mean():.2f}%"
          f"  MAE={top5['abs_err'].mean():.0f} MW  n={len(top5):,}")
    print(f"  Top 10 load hours  MAPE={top10['pct_err'].mean():.2f}%"
          f"  MAE={top10['abs_err'].mean():.0f} MW")
    print(f"\n  Top 10 highest-load hours:")
    print(f"  {'Period':<22}  {'Actual MW':>10}  {'Pred MW':>10}  {'Error MW':>9}  {'APE%':>6}")
    for _, r in top10.iterrows():
        print(f"  {str(r['period']):<22}  {r['demand_mwh']:>10,.0f}  "
              f"{r['pred_mwh']:>10,.0f}  {r['residual']:>+9,.0f}  {r['pct_err']:>6.1f}")


def plot_diagnostics(df: pd.DataFrame):
    test = df[df["split"] == "test"].copy().sort_values("period")
    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Actual vs predicted over time
    ax = fig.add_subplot(gs[0, :])
    ax.plot(test["period"], test["demand_mwh"], lw=0.6, alpha=0.7, label="Actual")
    ax.plot(test["period"], test["pred_mwh"],  lw=0.6, alpha=0.7, label="Predicted")
    ax.set_title("Actual vs Predicted — Test Period (Sep 2024 – Aug 2025)", fontsize=11)
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

    # 2. Residual vs hour
    ax2 = fig.add_subplot(gs[1, 0])
    hourly = test.groupby("hour")["residual"].agg(["mean", "std"]).reset_index()
    ax2.bar(hourly["hour"], hourly["mean"], yerr=hourly["std"],
            capsize=3, color="steelblue", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Mean Residual by Hour of Day", fontsize=10)
    ax2.set_xlabel("Hour"); ax2.set_ylabel("Pred − Actual (MW)")
    ax2.set_xticks(range(0, 24, 3))

    # 3. Residual vs temperature
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(test["temperature_f"], test["residual"],
                s=2, alpha=0.15, color="coral")
    # running mean
    temp_sorted = test.sort_values("temperature_f")
    roll = temp_sorted["residual"].rolling(300, center=True, min_periods=50).mean()
    ax3.plot(temp_sorted["temperature_f"], roll, color="darkred", lw=1.5,
             label="rolling mean")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Residual vs Temperature", fontsize=10)
    ax3.set_xlabel("Temperature (°F)"); ax3.set_ylabel("Pred − Actual (MW)")
    ax3.legend(fontsize=8)

    # 4. MAPE by temperature bin
    ax4 = fig.add_subplot(gs[2, 0])
    tbin = test.groupby("temp_bin", observed=True)["pct_err"].mean()
    ax4.bar(tbin.index.astype(str), tbin.values, color="darkorange", alpha=0.85)
    ax4.set_title("MAPE by Temperature Bin", fontsize=10)
    ax4.set_xlabel("Temperature bin (°F)"); ax4.set_ylabel("MAPE (%)")
    ax4.tick_params(axis="x", labelsize=8)

    # 5. Residual vs solar radiation
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(test["solar_radiation_wm2"], test["residual"],
                s=2, alpha=0.15, color="goldenrod")
    sol_sorted = test.sort_values("solar_radiation_wm2")
    roll_sol = sol_sorted["residual"].rolling(300, center=True, min_periods=50).mean()
    ax5.plot(sol_sorted["solar_radiation_wm2"], roll_sol,
             color="saddlebrown", lw=1.5, label="rolling mean")
    ax5.axhline(0, color="black", lw=0.8)
    ax5.set_title("Residual vs Solar Radiation", fontsize=10)
    ax5.set_xlabel("Solar radiation (W/m²)"); ax5.set_ylabel("Pred − Actual (MW)")
    ax5.legend(fontsize=8)

    # 6. Residual autocorrelation (lags 1–48 h)
    ax6 = fig.add_subplot(gs[3, 0])
    resid = test["residual"].values
    max_lag = 48
    acf = [1.0] + [
        np.corrcoef(resid[:-k], resid[k:])[0, 1] for k in range(1, max_lag + 1)
    ]
    ci = 1.96 / np.sqrt(len(resid))
    ax6.bar(range(max_lag + 1), acf, color="mediumpurple", alpha=0.85)
    ax6.axhline(ci,  color="red", lw=0.9, ls="--", label=f"95% CI (±{ci:.3f})")
    ax6.axhline(-ci, color="red", lw=0.9, ls="--")
    ax6.axhline(0,   color="black", lw=0.8)
    ax6.set_title("Residual Autocorrelation (lags 1–48 h)", fontsize=10)
    ax6.set_xlabel("Lag (hours)"); ax6.set_ylabel("ACF")
    ax6.legend(fontsize=8)

    # 7. MAPE by month
    ax7 = fig.add_subplot(gs[3, 1])
    monthly = test.groupby("month")["pct_err"].mean()
    ax7.bar([MONTH_NAMES[m-1] for m in monthly.index], monthly.values,
            color="seagreen", alpha=0.85)
    ax7.set_title("MAPE by Month — Test Year", fontsize=10)
    ax7.set_xlabel("Month"); ax7.set_ylabel("MAPE (%)")
    ax7.tick_params(axis="x", labelsize=8)

    out = RESULTS / "diagnostics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Plots saved → {out}")


def main():
    print("Loading predictions ...")
    df = load_preds()
    print(f"  {len(df):,} rows  ({df['split'].value_counts().to_dict()})")

    print_breakdown(df, "hour",    "MAPE by Hour of Day")
    print_breakdown(df, "month",   "MAPE by Month",
                    name_map={i+1: MONTH_NAMES[i] for i in range(12)})
    print_breakdown(df, "dow",     "MAPE by Day of Week",
                    name_map={i: DOW_NAMES[i] for i in range(7)})
    print_breakdown(df, "weekend", "MAPE: Weekday vs Weekend",
                    name_map={0: "Weekday", 1: "Weekend"})
    print_breakdown(df, "holiday", "MAPE: Non-holiday vs Holiday",
                    name_map={0: "Normal", 1: "Holiday"})
    print_breakdown(df, "temp_bin","MAPE by Temperature Bin")
    print_peak_errors(df)
    plot_diagnostics(df)


if __name__ == "__main__":
    main()
