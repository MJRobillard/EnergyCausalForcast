"""
Visualisation layer for peak event causal analysis.

Functions:
  plot_event_catalog        — table + timeline of all identified events
  plot_heat_trajectory      — per-event temperature build-up in the 72h before peak
  plot_event_waterfall      — demand excess decomposed hour-by-hour for one event
  plot_cross_event_heatmap  — events × components attribution heatmap
  plot_excess_vs_duration   — does excess grow with heatwave day count?
  print_event_summary       — text table of event attribution

  # Peak-hour analysis
  decompose_peak_hours      — causal decomposition for top-N demand hours only
  plot_peak_hour_waterfall  — waterfall for the single worst hour

  # Cumulative excess energy
  cumulative_excess_mwh     — MWh integral under the excess curve per event
  plot_cumulative_excess     — stacked area: how excess energy built up hour by hour

  # Ramp rate analysis
  compute_ramp_contributions — MW/hr per component during the duck-curve ramp window
  plot_ramp_contributions    — which component drove the steepest afternoon ramp
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# ── Component display order and colours ──────────────────────────────────────

COMPONENT_LABELS = {
    "comp_lag":      "Lag demand",
    "comp_cooling":  "Cooling (linear)",
    "comp_heating":  "Heating",
    "comp_inland":   "Inland heat",
    "comp_persist":  "Heat persistence",
    "comp_cool_sq":  "Cooling (nonlinear)",
    "comp_humid":    "Humidity",
    "comp_wind":     "Wind",
    "comp_daily":    "Daily shape",
    "comp_seasonal": "Seasonal",
    "comp_dow":      "DOW",
    "comp_holiday":  "Holiday",
    "comp_lighting": "Lighting",
    "comp_baseline": "Baseline (E0)",
}

COMPONENT_COLORS = {
    "comp_lag":      "#c44e52",
    "comp_cooling":  "#4c72b0",
    "comp_heating":  "#dd8452",
    "comp_inland":   "#e05c5c",
    "comp_persist":  "#937860",
    "comp_cool_sq":  "#8172b2",
    "comp_humid":    "#55a868",
    "comp_wind":     "#64b5cd",
    "comp_daily":    "#aaaaaa",
    "comp_seasonal": "#bbbbbb",
    "comp_dow":      "#cccccc",
    "comp_holiday":  "#dddddd",
    "comp_lighting": "#eeeeee",
    "comp_baseline": "#999999",
}

WEATHER_COMPONENTS = {"comp_cooling", "comp_heating", "comp_inland",
                      "comp_persist", "comp_cool_sq", "comp_humid", "comp_wind"}


# ── Event catalog ─────────────────────────────────────────────────────────────

def plot_event_catalog(events: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """
    Horizontal bar chart showing each event's duration and peak MW.
    Bar length = duration_hours, colour intensity = peak_mw.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, len(events) * 0.5 + 1))) if ax is None else (ax.figure, ax)

    ev = events.sort_values("peak_mw", ascending=True).reset_index(drop=True)
    norm  = mcolors.Normalize(vmin=ev["peak_mw"].min(), vmax=ev["peak_mw"].max())
    cmap  = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in ev["peak_mw"]]

    bars = ax.barh(ev["event_name"], ev["duration_hours"], color=colors, alpha=0.9)
    for bar, row in zip(bars, ev.itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row.peak_mw/1000:.1f} GW  ({row.duration_hours}h)",
                va="center", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Peak MW", fraction=0.02, pad=0.01)
    ax.set_xlabel("Duration (hours)")
    ax.set_title("Peak Event Catalog", fontsize=12)
    plt.tight_layout()
    return fig


# ── Heat trajectory ───────────────────────────────────────────────────────────

def plot_heat_trajectory(
    df: pd.DataFrame,
    events: pd.DataFrame,
    hours_before: int = 72,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    For each event plot temp_max in the hours_before window leading to the peak.
    Reveals whether events were preceded by a slow build or sudden spike.
    """
    n = min(len(events), 8)
    fig, ax = plt.subplots(figsize=(13, 5)) if ax is None else (ax.figure, ax)

    cmap = plt.cm.tab10
    for i, (_, ev) in enumerate(events.head(n).iterrows()):
        peak = ev["peak_hour"]
        window = df[
            (df["period"] >= peak - pd.Timedelta(hours=hours_before)) &
            (df["period"] <= peak)
        ].copy()
        if window.empty:
            continue
        hours_rel = (window["period"] - peak).dt.total_seconds() / 3600
        col = "temp_max" if "temp_max" in window.columns else "temperature_f"
        ax.plot(hours_rel, window[col], label=ev["event_name"],
                color=cmap(i / n), lw=1.5, alpha=0.85)
        ax.axvline(0, color="black", lw=0.7, ls="--")

    ax.set_xlabel(f"Hours before peak")
    ax.set_ylabel("Temp max (°F)")
    ax.set_title(f"Regional Temp Max — {hours_before}h Before Each Peak Hour", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.invert_xaxis()
    plt.tight_layout()
    return fig


# ── Per-event waterfall ───────────────────────────────────────────────────────

def plot_event_waterfall(
    event_name: str,
    attribution: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Horizontal waterfall showing component MW attribution for one event.
    attribution: output of uplift_decomposition() from counterfactual.py
    """
    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)

    u = attribution.sort_values("uplift_mw", ascending=True)
    colors = [
        COMPONENT_COLORS.get(row["component"], "#888888")
        if row["uplift_mw"] >= 0 else "#1f77b4"
        for _, row in u.iterrows()
    ]
    bars = ax.barh(u["label"], u["uplift_mw"], color=colors, alpha=0.88)
    ax.axvline(0, color="black", lw=0.8)

    for bar, val in zip(bars, u["uplift_mw"]):
        ax.text(val + (20 if val >= 0 else -20),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.0f} MW", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    total = attribution["total_demand_uplift_mw"].iloc[0]
    ax.set_title(f"{event_name} — Demand Excess Attribution  (total: {total:+,.0f} MW)",
                 fontsize=11)
    ax.set_xlabel("MW difference (event − synthetic control)")
    plt.tight_layout()
    return fig


# ── Cross-event heatmap ───────────────────────────────────────────────────────

def plot_cross_event_heatmap(
    attributions: dict[str, pd.DataFrame],
    normalise: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Heatmap: events (rows) × components (columns), values = MW attribution.
    normalise=True: each row divided by total uplift (shows relative share).

    attributions: dict event_name -> uplift_decomposition() DataFrame
    """
    # Build matrix
    all_components = list(COMPONENT_LABELS.keys())
    rows = {}
    for event_name, df_attr in attributions.items():
        row = {r["component"]: r["uplift_mw"] for _, r in df_attr.iterrows()}
        rows[event_name] = row

    matrix = pd.DataFrame(rows, index=all_components).T
    matrix = matrix.reindex(columns=all_components).fillna(0.0)

    if normalise:
        totals = matrix.abs().sum(axis=1).replace(0, 1)
        matrix = matrix.div(totals, axis=0) * 100
        label = "% of total uplift"
        fmt   = ".0f"
        vmax  = 60
    else:
        label = "MW attribution"
        fmt   = ".0f"
        vmax  = matrix.abs().max().max()

    col_labels = [COMPONENT_LABELS.get(c, c) for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(all_components) * 1.1),
                                    max(4, len(matrix) * 0.6 + 1.5))) if ax is None else (ax.figure, ax)

    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)

    for i in range(len(matrix)):
        for j in range(len(all_components)):
            val = matrix.values[i, j]
            if abs(val) > (vmax * 0.05):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > vmax * 0.4 else "black")

    fig.colorbar(im, ax=ax, label=label, fraction=0.02, pad=0.02)
    ax.set_title("Cross-Event Causal Attribution Heatmap", fontsize=12)
    plt.tight_layout()
    return fig


# ── Excess vs heatwave duration ───────────────────────────────────────────────

def plot_excess_vs_duration(
    events: pd.DataFrame,
    attributions: dict[str, pd.DataFrame],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Scatter: x = days_above_95f_at_peak, y = total demand excess MW.
    Reveals whether lag compounding grows with heatwave duration.
    """
    fig, ax = plt.subplots(figsize=(8, 5)) if ax is None else (ax.figure, ax)

    for _, ev in events.iterrows():
        name = ev["event_name"]
        if name not in attributions:
            continue
        total = attributions[name]["total_demand_uplift_mw"].iloc[0]
        days  = ev.get("days_above_95f_at_peak", np.nan)
        lag_attr = attributions[name][attributions[name]["component"] == "comp_lag"]
        lag_mw   = lag_attr["uplift_mw"].iloc[0] if len(lag_attr) else 0

        ax.scatter(days, total, s=100, zorder=3, color="#4c72b0")
        ax.scatter(days, lag_mw, s=60, marker="^", zorder=3, color="#c44e52")
        ax.annotate(name, (days, total), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Consecutive days above 95°F at peak hour")
    ax.set_ylabel("MW excess (event − synthetic control)")
    ax.set_title("Does Demand Excess Grow with Heatwave Duration?", fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0",
               markersize=8, label="Total excess"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#c44e52",
               markersize=8, label="Lag attribution"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    return fig


# ── Text summary ──────────────────────────────────────────────────────────────

def print_event_summary(event_name: str, attribution: pd.DataFrame) -> None:
    total = attribution["total_demand_uplift_mw"].iloc[0]
    top = attribution.sort_values("uplift_mw", ascending=False).head(5)
    weather_mw = attribution[attribution["component"].isin(WEATHER_COMPONENTS)]["uplift_mw"].sum()
    lag_mw     = attribution[attribution["component"] == "comp_lag"]["uplift_mw"].sum()

    width = 62
    print("─" * width)
    print(f"  {event_name}   (total excess: {total:+,.0f} MW)")
    print("─" * width)
    print(f"  {'Component':<26}  {'MW':>8}  {'%':>6}")
    for _, r in top.iterrows():
        pct = 100 * r["uplift_mw"] / total if total else 0
        print(f"  {r['label']:<26}  {r['uplift_mw']:>+8,.0f}  {pct:>5.1f}%")
    print("─" * width)
    print(f"  Direct weather (excl. lag)   {weather_mw:>+8,.0f}  "
          f"{100*weather_mw/total:>5.1f}%" if total else "")
    print(f"  Lag (incl. mediated weather) {lag_mw:>+8,.0f}  "
          f"{100*lag_mw/total:>5.1f}%" if total else "")
    print("─" * width)


# ── 1. Peak-hour decomposition ────────────────────────────────────────────────

def decompose_peak_hours(
    event_name: str,
    event_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    guide,
    model,
    top_n: int = 3,
    num_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run causal decomposition for only the top_n highest-demand hours of an event.

    Returns (peak_hour_attribution, peak_hours_df) where:
      peak_hour_attribution — uplift_decomposition() output for those hours
      peak_hours_df         — the subset of event_df used
    """
    from model.counterfactual import decompose_hours, uplift_decomposition

    top_idx    = event_df["demand_mwh"].nlargest(top_n).index
    peak_hours = event_df.loc[top_idx].reset_index(drop=True)

    # Corresponding baseline rows (same positional index)
    baseline_peak = baseline_df.loc[top_idx].reset_index(drop=True)

    actual_decomp   = decompose_hours(guide, model, peak_hours,   num_samples=num_samples)
    baseline_decomp = decompose_hours(guide, model, baseline_peak, num_samples=num_samples)
    attr = uplift_decomposition(actual_decomp, baseline_decomp)

    print(f"\n{'─'*62}")
    print(f"  {event_name} — Top-{top_n} peak hours")
    for _, r in peak_hours.iterrows():
        print(f"    {r['period']}  {r['demand_mwh']:,.0f} MW")
    print(f"{'─'*62}")
    top5 = attr.sort_values("uplift_mw", ascending=False).head(5)
    total = attr["total_demand_uplift_mw"].iloc[0]
    for _, r in top5.iterrows():
        bar = "▓" * max(0, int(20 * abs(r["uplift_mw"]) / (abs(total) + 1)))
        sign = "+" if r["uplift_mw"] >= 0 else ""
        print(f"  {r['label']:<26}  {sign}{r['uplift_mw']:>7,.0f} MW  {bar}")
    print(f"{'─'*62}")

    return attr, peak_hours


def plot_peak_hour_waterfall(
    event_name: str,
    peak_attr: pd.DataFrame,
    full_attr: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Side-by-side horizontal bar: full event average vs peak-hour decomposition.
    Highlights how much more extreme the causal drivers are at the worst hour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True) \
        if ax is None else (ax.figure, [ax, ax])

    for axis, (attr, title) in zip(axes, [
        (full_attr, "Full event average"),
        (peak_attr, "Peak hour(s) only"),
    ]):
        u = attr.sort_values("uplift_mw", ascending=True)
        colors = [COMPONENT_COLORS.get(r["component"], "#888")
                  if r["uplift_mw"] >= 0 else "#1f77b4"
                  for _, r in u.iterrows()]
        bars = axis.barh(u["label"], u["uplift_mw"], color=colors, alpha=0.88)
        axis.axvline(0, color="black", lw=0.8)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("MW excess (event − synthetic control)")
        for bar, val in zip(bars, u["uplift_mw"]):
            axis.text(val + (20 if val >= 0 else -20),
                      bar.get_y() + bar.get_height() / 2,
                      f"{val:+.0f}", va="center",
                      ha="left" if val >= 0 else "right", fontsize=7)

    fig.suptitle(f"{event_name} — Average vs Peak-Hour Causal Attribution", fontsize=12)
    plt.tight_layout()
    return fig


# ── 2. Cumulative excess energy ───────────────────────────────────────────────

def cumulative_excess_mwh(
    event_name: str,
    event_df: pd.DataFrame,
    actual_decomp: pd.DataFrame,
    baseline_decomp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute hour-by-hour cumulative excess MWh and per-component contribution.

    Each component's excess = actual_component_mean − baseline_component_mean,
    accumulated over the event window. One row per hour.

    Returns DataFrame with columns: period, hour_idx, excess_mw,
    cumulative_excess_mwh, comp_<name>_excess, comp_<name>_cumulative, ...
    """
    from model.counterfactual import COMPONENTS

    out = event_df[["period"]].copy().reset_index(drop=True)
    out["hour_idx"] = range(len(out))

    # Total excess MW per hour
    out["excess_mw"] = (actual_decomp["pred_mwh"].values
                        - baseline_decomp["pred_mwh"].values)
    out["cumulative_excess_mwh"] = out["excess_mw"].cumsum()

    # Per-component excess and cumulative
    for comp in COMPONENTS:
        if comp in actual_decomp.columns and comp in baseline_decomp.columns:
            excess = actual_decomp[comp].values - baseline_decomp[comp].values
            out[f"{comp}_excess"]     = excess
            out[f"{comp}_cumulative"] = excess.cumsum()

    return out


def plot_cumulative_excess(
    event_name: str,
    cumulative_df: pd.DataFrame,
    top_n_components: int = 5,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Two-panel figure:
      Top: MW excess per hour (bar) + cumulative MWh line
      Bottom: stacked area of top contributing components (cumulative MWh)

    Tells storage operators how much total energy capacity the event consumed
    above the normal baseline.
    """
    from model.counterfactual import COMPONENTS

    fig = plt.figure(figsize=(13, 8)) if ax is None else ax.figure
    gs  = fig.add_gridspec(2, 1, hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x = cumulative_df["hour_idx"].values

    # Top panel: hourly bar + cumulative line
    colors_bar = ["#d62728" if v > 0 else "#1f77b4"
                  for v in cumulative_df["excess_mw"]]
    ax1.bar(x, cumulative_df["excess_mw"], color=colors_bar, alpha=0.75, label="Hourly excess MW")
    ax1_r = ax1.twinx()
    ax1_r.plot(x, cumulative_df["cumulative_excess_mwh"], color="black",
               lw=2, label="Cumulative MWh")
    ax1_r.set_ylabel("Cumulative excess (MWh)", fontsize=9)
    ax1.set_ylabel("Hourly excess (MW)", fontsize=9)
    ax1.set_title(f"{event_name} — Hourly & Cumulative Excess Demand", fontsize=11)
    ax1.axhline(0, color="black", lw=0.5)

    # Annotate total MWh
    total_mwh = cumulative_df["cumulative_excess_mwh"].iloc[-1]
    ax1_r.annotate(f"Total: {total_mwh:,.0f} MWh",
                   xy=(x[-1], total_mwh), fontsize=9, color="black",
                   xytext=(-60, 10), textcoords="offset points")

    # Bottom panel: stacked component cumulative
    comp_cols = [c for c in [f"{comp}_cumulative" for comp in COMPONENTS]
                 if c in cumulative_df.columns]
    # Rank by final absolute value, take top_n
    final_vals = {c: abs(cumulative_df[c].iloc[-1]) for c in comp_cols}
    top_cols = sorted(final_vals, key=final_vals.get, reverse=True)[:top_n_components]

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    for col in top_cols:
        comp_key = col.replace("_cumulative", "")
        label    = COMPONENT_LABELS.get(comp_key, comp_key)
        color    = COMPONENT_COLORS.get(comp_key, "#aaaaaa")
        vals     = cumulative_df[col].values
        bottom   = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax2.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.82)
        pos_bottom = np.where(vals >= 0, pos_bottom + vals, pos_bottom)
        neg_bottom = np.where(vals < 0,  neg_bottom + vals, neg_bottom)

    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Hours into event")
    ax2.set_ylabel("Cumulative component MWh", fontsize=9)
    ax2.set_title("Component Breakdown of Cumulative Excess", fontsize=11)
    ax2.legend(fontsize=7, ncol=3, loc="upper left")

    plt.tight_layout()
    return fig


# ── 3. Ramp rate analysis ─────────────────────────────────────────────────────

RAMP_WINDOW = (14, 21)   # 2 PM – 9 PM: solar drop + peak demand overlap


def compute_ramp_contributions(
    event_df: pd.DataFrame,
    actual_decomp: pd.DataFrame,
    baseline_decomp: pd.DataFrame,
    ramp_start_h: int = RAMP_WINDOW[0],
    ramp_end_h:   int = RAMP_WINDOW[1],
) -> pd.DataFrame:
    """
    Compute MW/hr ramp rate for each component during the duck-curve window.

    For each day of the event, fits a linear slope (MW/hr) to each component's
    hourly values within [ramp_start_h, ramp_end_h]. Returns the mean slope
    across event days, split into actual vs baseline, and the excess ramp.

    Returns one row per component with columns:
        component, label,
        actual_ramp_mw_per_hr, baseline_ramp_mw_per_hr, excess_ramp_mw_per_hr
    """
    from model.counterfactual import COMPONENTS

    def _daily_slope(series: np.ndarray, hours: np.ndarray) -> float:
        if len(hours) < 2:
            return np.nan
        slope = np.polyfit(hours, series, 1)[0]
        return float(slope)

    merged = event_df[["period"]].copy().reset_index(drop=True)
    merged["hour"] = pd.to_datetime(merged["period"]).dt.hour
    merged["date"] = pd.to_datetime(merged["period"]).dt.date

    ramp_mask = merged["hour"].between(ramp_start_h, ramp_end_h - 1)
    ramp_hours = merged.loc[ramp_mask, "hour"].values
    ramp_dates = merged.loc[ramp_mask, "date"].values

    rows = []
    for comp in COMPONENTS:
        if comp not in actual_decomp.columns:
            continue
        actual_vals   = actual_decomp[comp].values[ramp_mask.values]
        baseline_vals = baseline_decomp[comp].values[ramp_mask.values]

        # Per-day slopes
        actual_slopes, baseline_slopes = [], []
        for day in np.unique(ramp_dates):
            day_mask = ramp_dates == day
            if day_mask.sum() < 2:
                continue
            actual_slopes.append(_daily_slope(actual_vals[day_mask], ramp_hours[day_mask]))
            baseline_slopes.append(_daily_slope(baseline_vals[day_mask], ramp_hours[day_mask]))

        actual_ramp   = float(np.nanmean(actual_slopes))   if actual_slopes   else np.nan
        baseline_ramp = float(np.nanmean(baseline_slopes)) if baseline_slopes else np.nan
        rows.append({
            "component":             comp,
            "label":                 COMPONENT_LABELS.get(comp, comp),
            "actual_ramp_mw_per_hr": actual_ramp,
            "baseline_ramp_mw_per_hr": baseline_ramp,
            "excess_ramp_mw_per_hr": actual_ramp - baseline_ramp
                                     if not np.isnan(actual_ramp) and not np.isnan(baseline_ramp)
                                     else np.nan,
        })

    return pd.DataFrame(rows).sort_values("excess_ramp_mw_per_hr", ascending=False)


def plot_ramp_contributions(
    event_name: str,
    ramp_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of excess ramp rate (MW/hr) per component during the
    duck-curve window. Positive = component accelerated demand ramp above normal.

    Tells operators which factor is driving the hardest-to-serve afternoon ramp.
    """
    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)

    r = ramp_df.dropna(subset=["excess_ramp_mw_per_hr"]).sort_values(
        "excess_ramp_mw_per_hr", ascending=True
    )
    colors = [COMPONENT_COLORS.get(row["component"], "#888")
              if row["excess_ramp_mw_per_hr"] >= 0 else "#1f77b4"
              for _, row in r.iterrows()]

    bars = ax.barh(r["label"], r["excess_ramp_mw_per_hr"], color=colors, alpha=0.88)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Excess ramp rate (MW/hr)  [event − synthetic control]")
    ax.set_title(
        f"{event_name} — Duck-Curve Ramp Contribution by Component\n"
        f"(2 PM – 9 PM window, mean over event days)",
        fontsize=11,
    )

    for bar, val in zip(bars, r["excess_ramp_mw_per_hr"]):
        if not np.isnan(val):
            ax.text(val + (0.5 if val >= 0 else -0.5),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f} MW/hr", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)

    plt.tight_layout()
    return fig
