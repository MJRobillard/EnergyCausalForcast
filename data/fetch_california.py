"""
Fetch California hourly demand (CISO via EIA API v2) and weather (Open-Meteo).
Then run baseline EDA + hyperparameter estimation vs. current WAUE model.
"""

import os, json, time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

OUT = Path(__file__).parent / "california"
OUT.mkdir(exist_ok=True)

EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


# ── 1. CISO load via EIA API v2 ───────────────────────────────────────────

def fetch_eia_ciso(start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Pull hourly demand for CISO (California ISO) balancing authority.
    start/end: 'YYYY-MM-DDTHH'
    Pages in 60-day chunks; EIA returns max 5000 rows per request.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "EIA API key required. Set EIA_API_KEY env var.\n"
            "Register free at https://www.eia.gov/opendata/register.php"
        )

    params_base = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": "D",        # D = demand
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        "offset": 0,
    }

    all_rows = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while current < end_ts:
        chunk_end = min(current + pd.Timedelta(days=59), end_ts)
        params = {
            **params_base,
            "start": current.strftime("%Y-%m-%dT%H"),
            "end": chunk_end.strftime("%Y-%m-%dT%H"),
        }
        resp = requests.get(EIA_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("response", {}).get("data", [])
        if not data:
            print(f"  [load] no data for {current.date()} – {chunk_end.date()}")
        else:
            all_rows.extend(data)
            print(f"  [load] {current.date()} – {chunk_end.date()}: {len(data)} rows")
        current = chunk_end + pd.Timedelta(hours=1)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    df["period"] = pd.to_datetime(df["period"])
    df = df.rename(columns={"value": "demand_mwh"})
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    df = df[["period", "demand_mwh"]].sort_values("period").reset_index(drop=True)
    return df


def load_ca_demand(start="2023-09-01T00", end="2025-08-31T23") -> pd.DataFrame:
    cache = OUT / "ca_load.csv"
    if cache.exists():
        print("  [load] using cached ca_load.csv")
        df = pd.read_csv(cache, parse_dates=["period"])
        df["period"] = df["period"].dt.floor("h")
        return df

    df = fetch_eia_ciso(start, end)
    df["period"] = df["period"].dt.floor("h")
    df = df.groupby("period")["demand_mwh"].mean().reset_index()
    df = df.sort_values("period").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"  [load] saved {len(df)} rows to ca_load.csv")
    return df


# ── 2. Open-Meteo weather — multi-station load-weighted composite ──────────
#
# CAISO territory is served by PG&E, SCE, and SDG&E (LADWP is its own BA).
# Weights approximate each utility's share of CAISO-wide hourly demand.
#
# Station                  lat       lon      weight  rationale
# Los Angeles (Burbank)    34.18  -118.35     0.30    SCE South — largest load center
# Bay Area (San Jose)      37.34  -121.89     0.25    PG&E North — dense pop + tech
# Sacramento               38.58  -121.49     0.15    PG&E Central + state capital
# Riverside (Inland Empire)33.95  -117.40     0.15    SCE inland — extreme heat events
# San Diego                32.72  -117.15     0.08    SDG&E territory
# Fresno                   36.74  -119.79     0.07    PG&E Central Valley heat
#
# Composite columns (same names as the old single-station CSV for backward
# compatibility): load-weighted mean of each variable.
# Additional column `temp_max`: unweighted max across all stations —
# captures inland/Southern CA heat wave severity beyond what the weighted
# mean shows.

STATIONS = [
    {"name": "los_angeles",    "lat":  34.18, "lon": -118.35, "weight": 0.30},
    {"name": "bay_area",       "lat":  37.34, "lon": -121.89, "weight": 0.25},
    {"name": "sacramento",     "lat":  38.58, "lon": -121.49, "weight": 0.15},
    {"name": "riverside",      "lat":  33.95, "lon": -117.40, "weight": 0.15},
    {"name": "san_diego",      "lat":  32.72, "lon": -117.15, "weight": 0.08},
    {"name": "fresno",         "lat":  36.74, "lon": -119.79, "weight": 0.07},
]

OPENMETEO_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"
OPENMETEO_URL  = "https://archive-api.open-meteo.com/v1/archive"


def fetch_station(station: dict, start: str, end: str) -> pd.DataFrame:
    """Fetch and cache one Open-Meteo station. Returns hourly DataFrame."""
    cache = OUT / f"wx_{station['name']}.parquet"
    if cache.exists():
        print(f"  [weather] cached  {station['name']}")
        return pd.read_parquet(cache)

    print(f"  [weather] fetching {station['name']} ({station['lat']}, {station['lon']}) ...")
    params = {
        "latitude":         station["lat"],
        "longitude":        station["lon"],
        "start_date":       start,
        "end_date":         end,
        "hourly":           OPENMETEO_VARS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone":         "America/Los_Angeles",
    }
    r = requests.get(OPENMETEO_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "time":                  "period",
        "temperature_2m":        "temperature_f",
        "relative_humidity_2m":  "humidity_pct",
        "wind_speed_10m":        "wind_mph",
        "shortwave_radiation":   "solar_radiation_wm2",
    })
    df.to_parquet(cache)
    time.sleep(0.5)   # be polite to the free API
    return df


def fetch_openmeteo_weather(start="2023-09-01", end="2025-08-31") -> pd.DataFrame:
    """
    Return a load-weighted composite weather DataFrame for CAISO territory.

    Columns match the old single-station schema (temperature_f, humidity_pct,
    wind_mph, solar_radiation_wm2) plus `temp_max` (unweighted peak across
    all stations, useful for heat-event features).
    """
    composite_cache = OUT / "ca_weather_composite.parquet"
    if composite_cache.exists():
        print("  [weather] using cached ca_weather_composite.parquet")
        return pd.read_parquet(composite_cache)

    station_dfs = [fetch_station(s, start, end) for s in STATIONS]
    weights     = [s["weight"] for s in STATIONS]

    # Align on period index
    base = station_dfs[0].set_index("period")[["temperature_f", "humidity_pct",
                                                "wind_mph", "solar_radiation_wm2"]]
    wx_vars = ["temperature_f", "humidity_pct", "wind_mph", "solar_radiation_wm2"]
    weighted = {v: sum(w * df.set_index("period")[v] for w, df in zip(weights, station_dfs))
                for v in wx_vars}
    temp_max = pd.concat(
        [df.set_index("period")["temperature_f"] for df in station_dfs], axis=1
    ).max(axis=1)

    out = pd.DataFrame(weighted)
    out["temp_max"] = temp_max
    out = out.reset_index().rename(columns={"index": "period"})
    out.to_parquet(composite_cache)
    print(f"  [weather] composite saved ({len(out):,} rows, {len(STATIONS)} stations)")
    return out


# ── 3. EDA + hyperparameter estimation ────────────────────────────────────

def run_eda(df: pd.DataFrame, label: str = "CA"):
    print(f"\n{'='*60}")
    print(f"  {label} DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Shape:        {df.shape}")
    print(f"  Period:       {df['period'].min()}  →  {df['period'].max()}")
    print(f"  Missing rows: {df.isnull().sum().sum()}")
    print()

    demand = df["demand_mwh"]
    temp   = df["temperature_f"]
    print(f"  demand_mwh   mean={demand.mean():.0f}  std={demand.std():.0f}"
          f"  min={demand.min():.0f}  max={demand.max():.0f}")
    print(f"  temperature  mean={temp.mean():.1f}°F  std={temp.std():.1f}"
          f"  min={temp.min():.1f}  max={temp.max():.1f}")
    print(f"  humidity     mean={df['humidity_pct'].mean():.1f}%"
          f"  std={df['humidity_pct'].std():.1f}")
    print(f"  wind_mph     mean={df['wind_mph'].mean():.1f}"
          f"  std={df['wind_mph'].std():.1f}")
    print(f"  solar_rad    mean={df['solar_radiation_wm2'].mean():.0f}"
          f"  std={df['solar_radiation_wm2'].std():.0f}")


def estimate_hyperparams(df: pd.DataFrame) -> dict:
    """
    Data-driven prior estimates for the SCM priors.
    Strategy:
      E0    ← median hourly demand when T near T_mid (50–62°F)
      k     ← OLS slope of demand vs |T - T_mid|
      a     ← daily Fourier coefficients (sin/cos of hour)
      alpha ← yearly Fourier coefficients (sin/cos of month)
      mu_*  ← marginal weather means
    """
    from numpy.linalg import lstsq

    h = df["hour"].values.astype(float)
    m = df["month"].values.astype(float)
    T = df["temperature_f"].values
    E = df["demand_mwh"].values

    print(f"\n{'='*60}")
    print("  DATA-DRIVEN HYPERPARAMETER ESTIMATES")
    print(f"{'='*60}")

    near_mid = (T > 50) & (T < 62)
    E0_est = np.median(E[near_mid]) if near_mid.sum() > 10 else np.median(E)
    print(f"\n  E0 (baseline demand near T_mid):  {E0_est:.0f} MWh")
    print(f"    → prior: Normal({E0_est:.0f}, 500)")

    dT = np.abs(T - 56.0)
    X_k = np.column_stack([dT, np.ones(len(dT))])
    k_est, E0_ols = lstsq(X_k, E, rcond=None)[0]
    print(f"\n  k  (V-shape slope):               {k_est:.2f} MWh/°F")
    print(f"    → prior: Normal({k_est:.1f}, 50)")
    print(f"  E0 (OLS intercept):               {E0_ols:.0f} MWh")

    E_detrend = E - (k_est * dT + E0_ols)
    Fh = np.column_stack(
        [np.sin(2*np.pi*j*h/24) for j in range(1, 5)] +
        [np.cos(2*np.pi*j*h/24) for j in range(1, 5)]
    )
    a_est = lstsq(Fh, E_detrend, rcond=None)[0]
    a_interleaved = np.array([v for pair in zip(a_est[:4], a_est[4:]) for v in pair])
    print(f"\n  a (daily Fourier coeffs) [sin1,cos1,sin2,cos2,...]:")
    print(f"    {np.round(a_interleaved, 1).tolist()}")

    Fm = np.column_stack(
        [np.sin(2*np.pi*j*m/12) for j in range(1, 4)] +
        [np.cos(2*np.pi*j*m/12) for j in range(1, 4)]
    )
    alpha_est = lstsq(Fm, E_detrend, rcond=None)[0]
    alpha_interleaved = np.array([v for pair in zip(alpha_est[:3], alpha_est[3:]) for v in pair])
    print(f"\n  alpha (yearly Fourier coeffs) [sin1,cos1,sin2,cos2,...]:")
    print(f"    {np.round(alpha_interleaved, 1).tolist()}")

    print(f"\n  Weather marginals:")
    print(f"    mu_rh   = {df['humidity_pct'].mean():.1f}  (std={df['humidity_pct'].std():.1f})")
    print(f"    mu_w    = {df['wind_mph'].mean():.1f}  (std={df['wind_mph'].std():.1f})")
    print(f"    mu_rad  = {df['solar_radiation_wm2'].mean():.0f}  (std={df['solar_radiation_wm2'].std():.0f})")
    print(f"\n  Temperature model:")
    print(f"    T_base  = {df['temperature_f'].mean():.1f}°F")

    return {
        "E0": float(E0_est),
        "k": float(k_est),
        "a": a_interleaved[:8].tolist(),
        "alpha": alpha_interleaved[:6].tolist(),
        "mu_rh": float(df["humidity_pct"].mean()),
        "mu_w": float(df["wind_mph"].mean()),
        "mu_rad": float(df["solar_radiation_wm2"].mean()),
        "T_base": float(df["temperature_f"].mean()),
    }


def compare_waue_vs_ca(df_ca: pd.DataFrame):
    waue_load = Path(__file__).parent / "raw" / "waue_load.csv"
    waue_wx   = Path(__file__).parent / "raw" / "waue_weather.csv"
    if not waue_load.exists():
        print("\n  [compare] WAUE raw files not found, skipping.")
        return

    load = pd.read_csv(waue_load, parse_dates=["period"])
    wx   = pd.read_csv(waue_wx,   parse_dates=["period"])
    load["period"] = load["period"].dt.floor("h")
    wx["period"]   = wx["period"].dt.floor("h")
    df_waue = load.merge(wx, on="period").dropna()
    df_waue["hour"]  = df_waue["period"].dt.hour.astype(float)
    df_waue["month"] = df_waue["period"].dt.month.astype(float)

    print(f"\n{'='*60}")
    print("  WAUE vs CALIFORNIA COMPARISON")
    print(f"{'='*60}")
    rows = [
        ("Rows",          f"{len(df_waue):,}",                  f"{len(df_ca):,}"),
        ("Period start",  str(df_waue["period"].min().date()),   str(df_ca["period"].min().date())),
        ("Period end",    str(df_waue["period"].max().date()),   str(df_ca["period"].max().date())),
        ("Demand mean",   f"{df_waue['demand_mwh'].mean():.0f} MWh",  f"{df_ca['demand_mwh'].mean():.0f} MWh"),
        ("Demand std",    f"{df_waue['demand_mwh'].std():.0f}",        f"{df_ca['demand_mwh'].std():.0f}"),
        ("Demand max",    f"{df_waue['demand_mwh'].max():.0f}",        f"{df_ca['demand_mwh'].max():.0f}"),
        ("Temp mean",     f"{df_waue['temperature_f'].mean():.1f}°F",  f"{df_ca['temperature_f'].mean():.1f}°F"),
        ("Humidity mean", f"{df_waue['humidity_pct'].mean():.1f}%",    f"{df_ca['humidity_pct'].mean():.1f}%"),
        ("Wind mean",     f"{df_waue['wind_mph'].mean():.1f} mph",     f"{df_ca['wind_mph'].mean():.1f} mph"),
        ("Solar mean",    f"{df_waue['solar_radiation_wm2'].mean():.0f} W/m²",
                          f"{df_ca['solar_radiation_wm2'].mean():.0f} W/m²"),
    ]
    print(f"  {'Metric':<18} {'WAUE':>22} {'California':>22}")
    print(f"  {'-'*64}")
    for label, w, c in rows:
        print(f"  {label:<18} {w:>22} {c:>22}")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching CISO load data via EIA API v2 ...")
    df_load = load_ca_demand("2023-09-01T00", "2025-08-31T23")

    print("\nFetching Open-Meteo weather — multi-station CAISO composite ...")
    df_wx = fetch_openmeteo_weather("2023-09-01", "2025-08-31")
    df_wx["period"] = df_wx["period"].dt.floor("h")
    df_wx.to_csv(OUT / "ca_weather.csv", index=False)

    print("\nMerging ...")
    df = df_load.merge(df_wx, on="period", how="inner").dropna()
    df["hour"]  = df["period"].dt.hour.astype(float)
    df["month"] = df["period"].dt.month.astype(float)
    df = df.sort_values("period").reset_index(drop=True)
    df.to_csv(OUT / "ca_merged.csv", index=False)
    print(f"  Merged: {len(df):,} rows")

    run_eda(df, "California (CISO via EIA + Open-Meteo 6-station composite)")
    params = estimate_hyperparams(df)
    compare_waue_vs_ca(df)

    # ca_hyperparams.json is for EDA reference only — it is estimated from
    # the full dataset including the test period and must NOT be used as
    # model priors in train/test evaluation. train_california.py re-estimates
    # priors from training data only.
    with open(OUT / "ca_hyperparams.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {OUT}/")
