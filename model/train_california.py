"""
Train and evaluate the SCM on California (CISO) data.

Priors are estimated from training data only (no test leakage).
CV uses TimeSeriesSplit so every validation fold is strictly in the future
relative to its training fold, and priors are re-estimated per fold.

Peak-focused metrics (CEC/CPUC-relevant) are computed alongside all-hour MAPE
and persisted to results/california/peak_metrics.csv.

Usage:
    python model/train_california.py
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
import pyro
from pathlib import Path
from numpy.linalg import lstsq
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.scm import make_tensors, make_model, train, predict, mape

DATA_DIR    = Path(__file__).parent.parent / "data" / "california"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "california"

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

# Expected hourly coverage: 2023-09-01 00:00 → 2025-08-31 23:00 inclusive
EXPECTED_HOURS = (
    pd.Timestamp("2025-09-01") - pd.Timestamp("2023-09-01")
).total_seconds() / 3600  # 17,544


# ── Peak / load-shape metrics ─────────────────────────────────────────────────

def peak_metrics(df: pd.DataFrame, label: str = "") -> dict:
    """
    Compute CEC/CPUC-relevant peak and load-shape metrics.

    Args:
        df: DataFrame with columns period (datetime), demand_mwh, pred_mwh.
        label: identifier stored in the returned dict (e.g. "train", "test").

    Returns dict with keys:
        split, n_hours,
        all_hour_mape, top1_mape, top5_mape,
        monthly_peak_mape, annual_peak_pct_error,
        peak_timing_accuracy, peak_underforecast_rate, peak_bias_mw
    """
    actual = df["demand_mwh"].values
    pred   = df["pred_mwh"].values
    period = pd.to_datetime(df["period"])

    # ── all-hour MAPE ──────────────────────────────────────────────────────
    nonzero = actual != 0
    all_mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100)

    # ── top 1% and top 5% load-hour MAPE ──────────────────────────────────
    def top_n_mape(pct: float) -> float:
        thresh = np.quantile(actual, 1 - pct / 100)
        mask   = actual >= thresh
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

    top1_mape = top_n_mape(1.0)
    top5_mape = top_n_mape(5.0)

    # ── monthly peak percent error ─────────────────────────────────────────
    # For each calendar month, compare predicted peak to actual peak.
    tmp = df[["period", "demand_mwh", "pred_mwh"]].copy()
    tmp["ym"] = period.dt.to_period("M")
    monthly = tmp.groupby("ym").agg(
        actual_peak=("demand_mwh", "max"),
        pred_peak=("pred_mwh", "max"),
    )
    monthly_pe = ((monthly["pred_peak"] - monthly["actual_peak"])
                  / monthly["actual_peak"] * 100).abs()
    monthly_peak_mape = float(monthly_pe.mean())

    # ── annual peak percent error ──────────────────────────────────────────
    annual_actual = float(actual.max())
    annual_pred   = float(pred.max())
    annual_peak_pct_error = float(
        abs(annual_pred - annual_actual) / annual_actual * 100
    )

    # ── daily peak-hour timing accuracy ───────────────────────────────────
    # Share of days where predicted peak hour is within ±1 h of actual peak.
    tmp["date"] = period.dt.date
    tmp["hour"] = period.dt.hour
    actual_peak_hour = tmp.loc[tmp.groupby("date")["demand_mwh"].idxmax(), ["date","hour"]] \
                          .set_index("date")["hour"]
    pred_peak_hour   = tmp.loc[tmp.groupby("date")["pred_mwh"].idxmax(),   ["date","hour"]] \
                          .set_index("date")["hour"]
    common_dates = actual_peak_hour.index.intersection(pred_peak_hour.index)
    hour_diff = (pred_peak_hour[common_dates] - actual_peak_hour[common_dates]).abs()
    peak_timing_acc = float((hour_diff <= 1).mean() * 100)

    # ── peak underforecast rate ────────────────────────────────────────────
    # Share of top-1% actual hours where the model underpredicts.
    thresh1 = np.quantile(actual, 0.99)
    top1_mask = actual >= thresh1
    if top1_mask.sum() > 0:
        underforecast_rate = float((pred[top1_mask] < actual[top1_mask]).mean() * 100)
        peak_bias_mw       = float((pred[top1_mask] - actual[top1_mask]).mean())
    else:
        underforecast_rate = float("nan")
        peak_bias_mw       = float("nan")

    return {
        "split":                  label,
        "n_hours":                int(len(df)),
        "all_hour_mape":          round(all_mape,             3),
        "top1_mape":              round(top1_mape,            3),
        "top5_mape":              round(top5_mape,            3),
        "monthly_peak_mape":      round(monthly_peak_mape,    3),
        "annual_peak_pct_error":  round(annual_peak_pct_error,3),
        "peak_timing_accuracy":   round(peak_timing_acc,      2),
        "peak_underforecast_rate":round(underforecast_rate,   2),
        "peak_bias_mw":           round(peak_bias_mw,         1),
    }


def print_peak_summary(metrics: dict, header: str) -> None:
    print(f"\n  {header}")
    print(f"    All-hour MAPE         : {metrics['all_hour_mape']:.2f}%")
    print(f"    Top-1% load MAPE      : {metrics['top1_mape']:.2f}%")
    print(f"    Top-5% load MAPE      : {metrics['top5_mape']:.2f}%")
    print(f"    Monthly peak MAPE     : {metrics['monthly_peak_mape']:.2f}%")
    print(f"    Annual peak error     : {metrics['annual_peak_pct_error']:.2f}%")
    print(f"    Peak underforecast    : {metrics['peak_underforecast_rate']:.1f}% of top-1% hours")


# ── Data loading / splitting ──────────────────────────────────────────────────

def check_coverage(df: pd.DataFrame) -> None:
    full_idx = pd.date_range("2023-09-01", "2025-08-31 23:00", freq="h")
    missing = full_idx.difference(df["period"])
    print(f"  Expected hours : {int(EXPECTED_HOURS):,}")
    print(f"  Present hours  : {len(df):,}")
    print(f"  Missing hours  : {len(missing)}", end="")
    if len(missing):
        print(f"  ({missing.min()} … {missing.max()})", end="")
    print()


def priors_from_data(df: pd.DataFrame) -> dict:
    """
    Estimate SCM prior means from df (train data only).

    When lag features are present, Fourier priors are estimated on
    lag-adjusted residuals so they are calibrated to corrections, not
    full demand. HVAC slopes use physically grounded defaults for CAISO
    scale (polyfit is too noisy across the full hour distribution).
    """
    h = df["hour"].values.astype(float)
    m = df["month"].values.astype(float)
    T = df["temperature_f"].values
    E = df["demand_mwh"].values

    has_lags = "E_lag24" in df.columns and df["E_lag24"].notna().any()

    if has_lags:
        lag24  = df["E_lag24"].fillna(df["demand_mwh"].mean()).values
        lag168 = df["E_lag168"].fillna(df["demand_mwh"].mean()).values
        E_base = 0.5 * lag24 + 0.3 * lag168
        E_fit  = E - E_base
        E0_prior = (0.0, 2000.0)
    else:
        E_fit    = E
        near_mid = (T > 50) & (T < 62)
        E0_est   = float(np.median(E[near_mid]) if near_mid.sum() > 10 else np.median(E))
        E0_prior = (E0_est, 500.0)

    k_cool_est = 60.0   # MW/°F — typical CAISO cooling sensitivity
    k_heat_est = 15.0   # MW/°F — California heating is mild

    dT   = np.maximum(T - 65.0, 0.0)
    E_dt = E_fit - k_cool_est * dT
    Fh = np.column_stack(
        [np.sin(2*np.pi*j*h/24) for j in range(1, 5)] +
        [np.cos(2*np.pi*j*h/24) for j in range(1, 5)]
    )
    a_raw = lstsq(Fh, E_dt, rcond=None)[0]
    a     = [v for pair in zip(a_raw[:4], a_raw[4:]) for v in pair]

    Fm = np.column_stack(
        [np.sin(2*np.pi*j*m/12) for j in range(1, 4)] +
        [np.cos(2*np.pi*j*m/12) for j in range(1, 4)]
    )
    alpha_raw = lstsq(Fm, E_dt - Fh @ np.concatenate([a_raw[:4], a_raw[4:]]),
                      rcond=None)[0]
    alpha     = [v for pair in zip(alpha_raw[:3], alpha_raw[3:]) for v in pair]

    return {
        "E0":           E0_prior,
        "k_cool":       (k_cool_est,                              50.0),
        "k_heat":       (k_heat_est,                              20.0),
        "T_cool_base":  (65.0,                                     4.0),
        "T_heat_base":  (55.0,                                     4.0),
        "T_base":       (float(df["temperature_f"].mean()),        4.0),
        "a":            (a,                                       40.0),
        "alpha":        (alpha,                                   40.0),
        "mu_rh_base":   (float(df["humidity_pct"].mean()),       20.0),
        "mu_w_base":    (float(df["wind_mph"].mean()),            10.0),
        "rad_base":     (float(df["solar_radiation_wm2"].mean()), 50.0),
    }


def load_data() -> pd.DataFrame:
    """Delegate to the data pipeline layer."""
    from data.ca_pipeline import load
    return load(DATA_DIR / "ca_merged.csv")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate to the data pipeline layer."""
    from data.ca_pipeline import train_test_split
    return train_test_split(df)


# ── CV ────────────────────────────────────────────────────────────────────────

def run_cv(df: pd.DataFrame, n_splits: int = 5,
           num_steps: int = 3000) -> tuple[list[dict], list[float]]:
    """
    Rolling-origin CV.  Returns (fold_metrics_list, fold_mapes).
    Steps scale with fold size so larger folds converge comparably to fold 1.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_tr_sizes = [len(tr_idx) for tr_idx, _ in tscv.split(df)]
    ref_size     = all_tr_sizes[0]

    fold_metrics_list: list[dict] = []
    fold_mapes: list[float]       = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(df), 1):
        tr   = df.iloc[tr_idx].reset_index(drop=True)
        val  = df.iloc[val_idx].reset_index(drop=True)
        fold_steps  = int(num_steps * len(tr_idx) / ref_size)
        fold_priors = priors_from_data(tr)
        fold_model  = make_model(fold_priors)
        _, guide    = train(make_tensors(tr), model=fold_model, num_steps=fold_steps)
        y_pred      = predict(guide, make_tensors(val), model=fold_model)

        val_with_pred = val.assign(pred_mwh=y_pred)
        fm = peak_metrics(val_with_pred, label=f"cv_fold_{fold}")
        fold_metrics_list.append(fm)
        fold_mapes.append(fm["all_hour_mape"])

        print(f"  Fold {fold}  train={len(tr):,}  val={len(val):,}  "
              f"steps={fold_steps:,}  MAPE={fm['all_hour_mape']:.2f}%  "
              f"top1={fm['top1_mape']:.2f}%  monthly_peak={fm['monthly_peak_mape']:.2f}%")

    return fold_metrics_list, fold_mapes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CA data ...")
    df = load_data()
    check_coverage(df)

    df_train, df_test = split(df)
    print(f"  Train: {len(df_train):,} rows  "
          f"({df_train['period'].min().date()} – {df_train['period'].max().date()})")
    print(f"  Test:  {len(df_test):,} rows  "
          f"({df_test['period'].min().date()} – {df_test['period'].max().date()})")

    print("\nEstimating priors from train data ...")
    train_priors = priors_from_data(df_train)
    for name, (mu, sig) in train_priors.items():
        mu_str = f"{mu:.1f}" if not isinstance(mu, list) else "[...]"
        print(f"  {name:<14}: mean={mu_str}  std={sig}")

    ca_model = make_model(train_priors)

    print("\n--- Train / Test Evaluation ---")
    tensors_train = make_tensors(df_train)
    tensors_test  = make_tensors(df_test)

    svi, guide = train(tensors_train, model=ca_model, num_steps=10000)

    y_train_pred = predict(guide, tensors_train, model=ca_model)
    y_test_pred  = predict(guide, tensors_test,  model=ca_model)

    df_train_pred = df_train.assign(pred_mwh=y_train_pred)
    df_test_pred  = df_test.assign(pred_mwh=y_test_pred)

    train_pm = peak_metrics(df_train_pred, label="train")
    test_pm  = peak_metrics(df_test_pred,  label="test")

    print("(Test weather is observed — this measures demand-given-weather fit,")
    print(" not an operational forecast with weather uncertainty.)")
    print_peak_summary(train_pm, "Train")
    print_peak_summary(test_pm,  "Test")

    df_out = pd.concat([
        df_train.assign(pred_mwh=y_train_pred, split="train"),
        df_test.assign(pred_mwh=y_test_pred,   split="test"),
    ])
    df_out.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    print("\n--- 5-Fold Rolling-Origin CV (train-before-validate) ---")
    fold_metrics_list, fold_mapes = run_cv(df_train, n_splits=5, num_steps=10000)
    cv_mean_mape  = float(np.mean(fold_mapes))

    def cv_mean_metric(key: str) -> float:
        vals = [fm[key] for fm in fold_metrics_list if not np.isnan(fm[key])]
        return float(np.mean(vals)) if vals else float("nan")

    cv_summary = {
        "split":                   "cv_mean",
        "n_hours":                 int(np.mean([fm["n_hours"] for fm in fold_metrics_list])),
        "all_hour_mape":           round(cv_mean_metric("all_hour_mape"),           3),
        "top1_mape":               round(cv_mean_metric("top1_mape"),               3),
        "top5_mape":               round(cv_mean_metric("top5_mape"),               3),
        "monthly_peak_mape":       round(cv_mean_metric("monthly_peak_mape"),       3),
        "annual_peak_pct_error":   round(cv_mean_metric("annual_peak_pct_error"),   3),
        "peak_timing_accuracy":    round(cv_mean_metric("peak_timing_accuracy"),    2),
        "peak_underforecast_rate": round(cv_mean_metric("peak_underforecast_rate"), 2),
        "peak_bias_mw":            round(cv_mean_metric("peak_bias_mw"),            1),
    }

    print(f"\n  CV mean — all-hour MAPE       : {cv_summary['all_hour_mape']:.2f}%")
    print(f"  CV mean — top-1% MAPE         : {cv_summary['top1_mape']:.2f}%")
    print(f"  CV mean — top-5% MAPE         : {cv_summary['top5_mape']:.2f}%")
    print(f"  CV mean — monthly peak MAPE   : {cv_summary['monthly_peak_mape']:.2f}%")
    print(f"  CV mean — peak underforecast  : {cv_summary['peak_underforecast_rate']:.1f}%")

    # ── persist ───────────────────────────────────────────────────────────
    all_metrics = [train_pm, test_pm] + fold_metrics_list + [cv_summary]
    pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "peak_metrics.csv", index=False)

    summary = {
        "region": "california",
        "note": (
            "priors estimated from train split only; "
            "CV uses TimeSeriesSplit with per-fold prior re-estimation; "
            "test MAPE conditioned on observed weather; "
            "weather: 6-station load-weighted composite "
            "(LA, Bay Area, Sacramento, Riverside, San Diego, Fresno); "
            "SCM: asymmetric HVAC, day-of-week, holiday, lag-24h and lag-168h demand"
        ),
        "train":    train_pm,
        "test":     test_pm,
        "cv_mean":  cv_summary,
        "cv_folds": fold_metrics_list,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save param store so counterfactual analysis can be run standalone
    pyro.get_param_store().save(str(RESULTS_DIR / "param_store.pt"))

    # ── Counterfactual + decomposition analysis ───────────────────────────
    print("\n--- Spike Decomposition & Counterfactual Analysis ---")
    try:
        from data.ca_pipeline import label_spikes, match_normals, make_scenario, SCENARIOS
        from model.counterfactual import (
            decompose_hours, uplift_decomposition, peak_scenario_summary
        )
        from analysis.spike_report import generate_report, print_uplift_summary, print_scenario_summary

        # Decompose test predictions into components
        print("  Decomposing test hours into SCM components ...")
        test_decomp = decompose_hours(guide, ca_model, df_test, num_samples=200)
        test_decomp.to_csv(RESULTS_DIR / "decomposition.csv", index=False)

        # Spike vs matched normal uplift
        df_test_labeled = label_spikes(df_test_pred)
        spike_mask = df_test_labeled["is_top1pct"]
        spike_decomp  = decompose_hours(guide, ca_model,
                                        df_test_labeled[spike_mask].reset_index(drop=True))
        normal_matched = match_normals(df_test_labeled, spike_mask)
        if len(normal_matched) > 0:
            normal_decomp = decompose_hours(guide, ca_model,
                                            normal_matched.reset_index(drop=True))
            uplift = uplift_decomposition(spike_decomp, normal_decomp)
            uplift.to_csv(RESULTS_DIR / "spike_decomposition.csv", index=False)
            print_uplift_summary(uplift)

        # Weather scenario analysis on test set
        print("\n  Running weather scenarios ...")
        scenario_dfs = {name: make_scenario(df_test, spec)
                        for name, spec in SCENARIOS.items() if name != "observed"}
        cf_summary = peak_scenario_summary(guide, ca_model, df_test,
                                           SCENARIOS, num_samples=100)
        cf_summary.to_csv(RESULTS_DIR / "counterfactual_peaks.csv", index=False)
        print_scenario_summary(cf_summary)

        # Figures
        generate_report(RESULTS_DIR)

    except Exception as exc:
        print(f"  [warn] counterfactual analysis skipped: {exc}")

    print(f"\nResults saved to {RESULTS_DIR}/")


# ── sanity check (runs without training, uses synthetic data) ─────────────────

def _test_peak_metrics() -> None:
    """Quick smoke-test for peak_metrics() — no model required."""
    rng = np.random.default_rng(0)
    n   = 8760
    periods = pd.date_range("2024-01-01", periods=n, freq="h")
    actual  = 25000 + 5000 * np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 500, n)
    pred    = actual + rng.normal(0, 1000, n)   # ~4% error

    df = pd.DataFrame({"period": periods, "demand_mwh": actual, "pred_mwh": pred})
    m  = peak_metrics(df, label="synthetic")

    assert 0 < m["all_hour_mape"] < 20,          f"all_hour_mape out of range: {m}"
    assert 0 < m["top1_mape"],                    f"top1_mape non-positive: {m}"
    assert 0 < m["monthly_peak_mape"] < 50,       f"monthly_peak_mape out of range: {m}"
    assert 0 <= m["annual_peak_pct_error"] < 50,  f"annual_peak_pct_error out of range: {m}"
    assert 0 <= m["peak_timing_accuracy"] <= 100, f"peak_timing_accuracy out of range: {m}"
    assert 0 <= m["peak_underforecast_rate"] <= 100
    assert m["n_hours"] == n
    print("  peak_metrics() sanity check passed")
    print(f"    {m}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test_peak_metrics()
    else:
        main()
