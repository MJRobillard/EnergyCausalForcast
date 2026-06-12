"""
Baseline comparison for California hourly demand forecasting.

Baselines evaluated on same train/test split as the SCM
(train: Sep 2023 – Aug 2024, test: Sep 2024 – Aug 2025).

All baselines are trained on train data only; no test leakage.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.scm import mape

DATA_DIR    = Path(__file__).parent.parent / "data" / "california"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "california"

US_HOLIDAYS = {
    # 2023
    "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-10",
    "2023-11-23", "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-11",
    "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-11",
    "2025-11-27", "2025-12-25",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"])
    df["dow"]     = df["period"].dt.dayofweek          # 0=Mon … 6=Sun
    df["date_str"]= df["period"].dt.strftime("%Y-%m-%d")
    df["holiday"] = df["date_str"].isin(US_HOLIDAYS).astype(int)
    df["E_lag24"]  = df["demand_mwh"].shift(24)
    df["E_lag168"] = df["demand_mwh"].shift(168)
    return df


def load_and_split():
    df = pd.read_csv(DATA_DIR / "ca_merged.csv", parse_dates=["period"])
    df["period"] = df["period"].dt.floor("h")
    df = df.sort_values("period").reset_index(drop=True)
    df = add_features(df)
    train = df[df["period"] < "2024-09-01"].reset_index(drop=True)
    test  = df[df["period"] >= "2024-09-01"].reset_index(drop=True)
    return train, test


# ── individual baseline functions ────────────────────────────────────────────

def baseline_lag24(train, test):
    """Same hour yesterday."""
    return test["E_lag24"].values, "Lag-24h"


def baseline_lag168(train, test):
    """Same hour last week."""
    return test["E_lag168"].values, "Lag-168h (same-week)"


def baseline_month_hour(train, test):
    """Mean demand by (month, hour) from train."""
    lut = train.groupby(["month", "hour"])["demand_mwh"].mean()
    pred = test.apply(lambda r: lut.get((r["month"], r["hour"]), np.nan), axis=1)
    return pred.values, "Month×Hour mean"


def baseline_dow_hour(train, test):
    """Mean demand by (day-of-week, hour) from train."""
    lut = train.groupby(["dow", "hour"])["demand_mwh"].mean()
    pred = test.apply(lambda r: lut.get((r["dow"], r["hour"]), np.nan), axis=1)
    return pred.values, "DOW×Hour mean"


def baseline_linear(train, test):
    """Ridge regression: weather + Fourier(hour,month) + dow dummies + holiday."""
    def featurize(df):
        h = df["hour"].values
        m = df["month"].values
        fh = np.column_stack(
            [np.sin(2*np.pi*j*h/24) for j in range(1, 5)] +
            [np.cos(2*np.pi*j*h/24) for j in range(1, 5)]
        )
        fm = np.column_stack(
            [np.sin(2*np.pi*j*m/12) for j in range(1, 4)] +
            [np.cos(2*np.pi*j*m/12) for j in range(1, 4)]
        )
        dow_dummies = np.eye(7)[df["dow"].values.astype(int)]
        weather = df[["temperature_f", "humidity_pct", "wind_mph",
                       "solar_radiation_wm2"]].values
        holiday = df[["holiday"]].values
        return np.hstack([fh, fm, dow_dummies, weather, holiday])

    X_tr = featurize(train);  y_tr = train["demand_mwh"].values
    X_te = featurize(test)
    scaler = StandardScaler().fit(X_tr)
    mdl = Ridge(alpha=10.0).fit(scaler.transform(X_tr), y_tr)
    return mdl.predict(scaler.transform(X_te)), "Ridge (weather+calendar)"


def baseline_gbm(train, test):
    """HistGradientBoosting: weather + calendar + lag features."""
    FEATS = ["temperature_f", "humidity_pct", "wind_mph", "solar_radiation_wm2",
             "hour", "month", "dow", "holiday", "E_lag24", "E_lag168"]
    mdl = HistGradientBoostingRegressor(max_iter=500, max_depth=6,
                                        learning_rate=0.05, random_state=42)
    mdl.fit(train[FEATS], train["demand_mwh"])
    return mdl.predict(test[FEATS]), "HistGBM (weather+calendar+lags)"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data ...")
    train, test = load_and_split()
    print(f"  Train: {len(train):,}  Test: {len(test):,}")

    # Drop test rows where lags are NaN (first ~168 h at boundary)
    valid = test["E_lag168"].notna() & test["E_lag24"].notna()
    test_full = test
    test_lag  = test[valid].reset_index(drop=True)

    results = []
    for fn, uses_lag in [
        (baseline_lag24,    True),
        (baseline_lag168,   True),
        (baseline_month_hour, False),
        (baseline_dow_hour,   False),
        (baseline_linear,   False),
        (baseline_gbm,      True),
    ]:
        t = test_lag if uses_lag else test_full
        pred, name = fn(train, t)
        mask = ~np.isnan(pred)
        m = mape(t["demand_mwh"].values[mask], pred[mask])
        results.append((name, m, mask.sum()))
        print(f"  {name:<35}  MAPE = {m:.2f}%  (n={mask.sum():,})")

    # Save
    pd.DataFrame(results, columns=["baseline", "mape_pct", "n_hours"]).to_csv(
        RESULTS_DIR / "baseline_mapes.csv", index=False
    )
    print(f"\nSaved to {RESULTS_DIR}/baseline_mapes.csv")


if __name__ == "__main__":
    main()
