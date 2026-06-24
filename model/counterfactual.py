"""
Counterfactual analysis layer.

Responsibilities:
  - Predict demand under weather scenarios using a trained guide
  - Decompose spike hours into component contributions
  - Compute weather-driven uplift between spike and matched normal hours
  - Expose results as plain DataFrames; plotting lives in analysis/spike_report.py

This module does not modify the SCM or load data. It takes a trained
guide + DataFrames prepared by data/ca_pipeline.py and returns results.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import pyro
from pyro.infer import Predictive

# Component names in the order they appear in the decomposition output.
COMPONENTS = [
    "comp_lag",
    "comp_cooling",
    "comp_heating",
    "comp_inland",
    "comp_persist",
    "comp_cool_sq",
    "comp_humid",
    "comp_wind",
    "comp_daily",
    "comp_seasonal",
    "comp_dow",
    "comp_holiday",
    "comp_lighting",
    "comp_baseline",
]

COMPONENT_LABELS = {
    "comp_lag":      "Lag demand",
    "comp_cooling":  "Cooling (linear)",
    "comp_heating":  "Heating",
    "comp_inland":   "Inland extreme heat",
    "comp_persist":  "Heat persistence",
    "comp_cool_sq":  "Cooling (nonlinear)",
    "comp_humid":    "Humidity",
    "comp_wind":     "Wind",
    "comp_daily":    "Daily shape",
    "comp_seasonal": "Seasonal",
    "comp_dow":      "Day-of-week",
    "comp_holiday":  "Holiday",
    "comp_lighting": "Lighting",
    "comp_baseline": "Baseline (E0)",
}


# ── Core prediction with component extraction ─────────────────────────────────

def predict_with_components(
    guide,
    model,
    tensors: dict,
    num_samples: int = 200,
) -> dict[str, np.ndarray]:
    """
    Run posterior predictive and return mean prediction plus all components.

    Returns dict with keys:
        "pred_mwh"   — posterior mean of E_mu  (shape: N)
        "pred_p05"   — 5th percentile of E_mu  (shape: N)
        "pred_p95"   — 95th percentile of E_mu (shape: N)
        "comp_*"     — posterior mean of each named component (shape: N)
    """
    inp = {k: v for k, v in tensors.items() if k != "E_obs"}
    predictive = Predictive(model, guide=guide, num_samples=num_samples,
                            return_sites=["E_mu"] + COMPONENTS)
    samples = predictive(**inp)

    result = {
        "pred_mwh": samples["E_mu"].mean(0).detach().numpy().flatten(),
        "pred_p05": samples["E_mu"].quantile(0.05, 0).detach().numpy().flatten(),
        "pred_p95": samples["E_mu"].quantile(0.95, 0).detach().numpy().flatten(),
    }
    for comp in COMPONENTS:
        if comp in samples:
            result[comp] = samples[comp].mean(0).detach().numpy().flatten()
    return result


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_scenarios(
    guide,
    model,
    df_base: pd.DataFrame,
    scenarios: dict[str, pd.DataFrame],
    num_samples: int = 200,
) -> pd.DataFrame:
    """
    Predict demand under each scenario and return a tidy results DataFrame.

    Args:
        guide: trained Pyro AutoNormal guide
        model: Pyro model function
        df_base: baseline DataFrame (observed)
        scenarios: dict mapping scenario name -> perturbed DataFrame
                   (from data.ca_pipeline.make_scenario)
        num_samples: posterior samples per scenario

    Returns DataFrame with columns:
        period, scenario, pred_mwh, delta_mw (vs observed)
    """
    from model.scm import make_tensors

    rows = []
    # observed first
    base_tensors = make_tensors(df_base)
    base_pred = predict_with_components(guide, model, base_tensors, num_samples)

    for name, df_scenario in scenarios.items():
        tensors = make_tensors(df_scenario)
        preds   = predict_with_components(guide, model, tensors, num_samples)
        for i, row in df_base.iterrows():
            rows.append({
                "period":    row["period"],
                "scenario":  name,
                "pred_mwh":  float(preds["pred_mwh"][i]),
                "obs_mwh":   float(row["demand_mwh"]),
                "delta_mw":  float(preds["pred_mwh"][i] - base_pred["pred_mwh"][i]),
            })

    return pd.DataFrame(rows)


# ── Spike decomposition ───────────────────────────────────────────────────────

def decompose_hours(
    guide,
    model,
    df: pd.DataFrame,
    num_samples: int = 200,
) -> pd.DataFrame:
    """
    Return a DataFrame with posterior mean component contributions for each hour.

    Columns: period, demand_mwh, pred_mwh, pred_p05, pred_p95,
             comp_lag, comp_cooling, comp_heating, comp_inland, comp_persist,
             comp_cool_sq, comp_humid, comp_wind, comp_daily, comp_seasonal,
             comp_dow, comp_holiday, comp_lighting, comp_baseline, residual
    """
    from model.scm import make_tensors

    tensors = make_tensors(df)
    result  = predict_with_components(guide, model, tensors, num_samples)

    out = df[["period", "demand_mwh"]].copy().reset_index(drop=True)
    out["pred_mwh"] = result["pred_mwh"]
    out["pred_p05"] = result["pred_p05"]
    out["pred_p95"] = result["pred_p95"]

    for comp in COMPONENTS:
        out[comp] = result.get(comp, 0.0)

    # residual: what the SCM can't explain
    explained = sum(out[c] for c in COMPONENTS if c in out.columns)
    out["residual"] = out["pred_mwh"] - explained

    return out


# ── Spike vs. normal uplift decomposition ─────────────────────────────────────

def uplift_decomposition(
    spike_decomp: pd.DataFrame,
    normal_decomp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify how much of the spike-vs-normal demand gap is explained by
    each component.

    Args:
        spike_decomp : output of decompose_hours() for spike hours
        normal_decomp: output of decompose_hours() for matched normal hours

    Returns a summary DataFrame with one row per component:
        component, label,
        spike_mean_mw, normal_mean_mw, uplift_mw, uplift_pct_of_total
    """
    spike_means  = spike_decomp[[c for c in COMPONENTS if c in spike_decomp]].mean()
    normal_means = normal_decomp[[c for c in COMPONENTS if c in normal_decomp]].mean()

    total_uplift = spike_decomp["demand_mwh"].mean() - normal_decomp["demand_mwh"].mean()

    rows = []
    for comp in COMPONENTS:
        if comp not in spike_means.index:
            continue
        uplift_mw = float(spike_means[comp] - normal_means.get(comp, 0.0))
        rows.append({
            "component":          comp,
            "label":              COMPONENT_LABELS.get(comp, comp),
            "spike_mean_mw":      round(float(spike_means[comp]),  1),
            "normal_mean_mw":     round(float(normal_means.get(comp, 0.0)), 1),
            "uplift_mw":          round(uplift_mw, 1),
            "uplift_pct_of_total": round(100 * uplift_mw / total_uplift, 1)
                                   if abs(total_uplift) > 0 else 0.0,
        })

    summary = pd.DataFrame(rows).sort_values("uplift_mw", ascending=False)
    summary["total_demand_uplift_mw"] = round(total_uplift, 1)
    return summary


# ── Peak scenario summary ─────────────────────────────────────────────────────

def peak_scenario_summary(
    guide,
    model,
    df_test: pd.DataFrame,
    scenario_specs: dict[str, dict],
    num_samples: int = 100,
) -> pd.DataFrame:
    """
    For each weather scenario, report: mean demand, peak demand,
    top-1% mean demand, and delta vs. observed.

    scenario_specs: dict name -> scenario dict as used by data.ca_pipeline.make_scenario
    Returns one row per scenario.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.ca_pipeline import make_scenario
    from model.scm import make_tensors

    base_tensors = make_tensors(df_test)
    base_pred    = predict_with_components(guide, model, base_tensors, num_samples)
    base_mean    = base_pred["pred_mwh"].mean()
    base_peak    = base_pred["pred_mwh"].max()
    thresh1      = np.quantile(base_pred["pred_mwh"], 0.99)
    base_top1    = base_pred["pred_mwh"][base_pred["pred_mwh"] >= thresh1].mean()

    rows = [{"scenario": "observed", "mean_mw": round(base_mean, 0),
             "peak_mw": round(base_peak, 0), "top1_mean_mw": round(base_top1, 0),
             "delta_peak_mw": 0.0, "delta_top1_mw": 0.0}]

    for name, spec in scenario_specs.items():
        df_s  = make_scenario(df_test, spec)
        pred  = predict_with_components(guide, model, make_tensors(df_s), num_samples)
        peak  = pred["pred_mwh"].max()
        top1  = pred["pred_mwh"][pred["pred_mwh"] >= thresh1].mean()
        rows.append({
            "scenario":       name,
            "mean_mw":        round(pred["pred_mwh"].mean(), 0),
            "peak_mw":        round(peak, 0),
            "top1_mean_mw":   round(top1, 0),
            "delta_peak_mw":  round(peak - base_peak, 0),
            "delta_top1_mw":  round(top1 - base_top1, 0),
        })

    return pd.DataFrame(rows)
