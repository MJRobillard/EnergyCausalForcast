"""
End-to-end training and evaluation of the SCM energy demand model.
Reproduces results from Ma et al. 2024 (arXiv:2512.11653).

Usage:
    python model/train.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.scm import make_tensors, model, train, predict, mape
import pyro

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_and_merge() -> pd.DataFrame:
    load = pd.read_csv(DATA_DIR / "waue_load.csv", parse_dates=["period"])
    weather = pd.read_csv(DATA_DIR / "waue_weather.csv", parse_dates=["period"])

    # align on hour boundary
    load["period"] = load["period"].dt.floor("h")
    weather["period"] = weather["period"].dt.floor("h")

    df = load.merge(weather, on="period", how="inner")
    df = df.dropna()
    df["hour"] = df["period"].dt.hour.astype(float)
    df["month"] = df["period"].dt.month.astype(float)
    df = df.sort_values("period").reset_index(drop=True)
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: Sep 2023 – Aug 2024.  Test: Sep 2024 – Aug 2025."""
    train_mask = df["period"] < "2024-09-01"
    test_mask = df["period"] >= "2024-09-01"
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def run_cv(df: pd.DataFrame, n_splits: int = 5,
           num_steps: int = 3000) -> list[float]:
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_mapes = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df), 1):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        val = df.iloc[val_idx].reset_index(drop=True)
        tensors_tr = make_tensors(tr)
        tensors_val = make_tensors(val)
        _, guide = train(tensors_tr, num_steps=num_steps)
        y_pred = predict(guide, tensors_val)
        y_true = val["demand_mwh"].values
        m = mape(y_true, y_pred)
        fold_mapes.append(m)
        print(f"  Fold {fold} MAPE: {m:.2f}%")
    return fold_mapes


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data ...")
    df = load_and_merge()
    print(f"  Total rows: {len(df)}, columns: {list(df.columns)}")

    df_train, df_test = split(df)
    print(f"  Train: {len(df_train)} rows  ({df_train['period'].min().date()} – {df_train['period'].max().date()})")
    print(f"  Test:  {len(df_test)} rows  ({df_test['period'].min().date()} – {df_test['period'].max().date()})")

    # ---- main train/test run ----
    print("\n--- Train / Test Evaluation ---")
    tensors_train = make_tensors(df_train)
    tensors_test = make_tensors(df_test)

    svi, guide = train(tensors_train, num_steps=5000)

    y_train_pred = predict(guide, tensors_train)
    y_test_pred = predict(guide, tensors_test)

    train_mape = mape(df_train["demand_mwh"].values, y_train_pred)
    test_mape = mape(df_test["demand_mwh"].values, y_test_pred)
    print(f"\nTrain MAPE: {train_mape:.2f}%  (paper: 3.23%)")
    print(f"Test  MAPE: {test_mape:.2f}%  (paper: 3.84%)")

    # save predictions
    df_train = df_train.copy()
    df_train["pred_mwh"] = y_train_pred
    df_test = df_test.copy()
    df_test["pred_mwh"] = y_test_pred
    pd.concat([df_train, df_test]).to_csv(RESULTS_DIR / "predictions.csv", index=False)

    # ---- 5-fold CV ----
    print("\n--- 5-Fold Cross-Validation ---")
    fold_mapes = run_cv(df, n_splits=5, num_steps=3000)
    cv_mean = np.mean(fold_mapes)
    print(f"\nCV MAPE (mean): {cv_mean:.2f}%  (paper: 3.88%)")

    # save summary
    summary = {
        "train_mape": train_mape,
        "test_mape": test_mape,
        "cv_mape_mean": cv_mean,
        "cv_fold_mapes": fold_mapes,
    }
    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
