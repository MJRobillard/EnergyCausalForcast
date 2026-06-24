"""
Fetch historical CAISO demand (EIA-930) and regional weather (Open-Meteo)
for California, 2018-01-01 through the current data start date.

Usage:
    python data/fetch_historical.py

Writes:
    data/california/ca_load_historical.csv
    data/california/wx_<region>_historical.parquet

These are then merged with the existing 2023-2025 files by peak_pipeline.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry

DATA_DIR   = Path(__file__).parent / "california"
EIA_KEY    = os.environ.get("EIA_API_KEY", "")
FETCH_START = "2018-01-01"
FETCH_END   = "2023-08-31"   # existing data starts 2023-09-01

# Open-Meteo station coordinates (lat, lon) for each CAISO region
REGION_COORDS = {
    "bay_area":    (37.77,  -122.42),   # San Francisco
    "fresno":      (36.74,  -119.77),   # Fresno
    "los_angeles": (34.05,  -118.24),   # Los Angeles
    "riverside":   (33.98,  -117.37),   # Riverside
    "sacramento":  (38.58,  -121.49),   # Sacramento
    "san_diego":   (32.72,  -117.15),   # San Diego
}


# ── EIA-930 CAISO demand ──────────────────────────────────────────────────────

def fetch_eia_demand(start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Fetch hourly CAISO demand from EIA-930 API.
    Paginates automatically (EIA returns max 5000 rows per call).
    """
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_rows = []
    offset   = 0
    length   = 5000

    print(f"Fetching EIA-930 CAISO demand {start} – {end} ...")
    while True:
        params = {
            "api_key":              api_key,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "CISO",
            "facets[type][]":       "D",
            "start":                start,
            "end":                  end,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
            "offset":               offset,
            "length":               length,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()["response"]["data"]
        if not data:
            break
        all_rows.extend(data)
        print(f"  fetched {len(all_rows):,} rows ...", end="\r")
        if len(data) < length:
            break
        offset += length
        time.sleep(0.3)   # be polite

    print(f"\n  total: {len(all_rows):,} rows")
    df = pd.DataFrame(all_rows)
    df["period"]     = pd.to_datetime(df["period"]).dt.floor("h")
    df["demand_mwh"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["period", "demand_mwh"]].dropna().sort_values("period").reset_index(drop=True)


# ── Open-Meteo regional weather ───────────────────────────────────────────────

def fetch_openmeteo_region(region: str, lat: float, lon: float,
                           start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly weather for one region via Open-Meteo historical API.
    Returns DataFrame with columns matching the existing wx_<region>.parquet schema.
    """
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.4)
    om = openmeteo_requests.Client(session=retry_session)

    print(f"  Fetching Open-Meteo: {region} ({lat}, {lon}) ...")
    responses = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":              lat,
            "longitude":             lon,
            "start_date":            start,
            "end_date":              end,
            "hourly":                [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "apparent_temperature",
            ],
            "temperature_unit":      "fahrenheit",
            "wind_speed_unit":       "mph",
            "timezone":              "America/Los_Angeles",
        },
    )
    r  = responses[0]
    h  = r.Hourly()

    times = pd.date_range(
        start=pd.Timestamp(h.Time(),    unit="s", tz="America/Los_Angeles"),
        end=  pd.Timestamp(h.TimeEnd(), unit="s", tz="America/Los_Angeles"),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    ).tz_localize(None)

    df = pd.DataFrame({
        "period":               times,
        "temperature_f":        h.Variables(0).ValuesAsNumpy(),
        "humidity_pct":         h.Variables(1).ValuesAsNumpy(),
        "wind_mph":             h.Variables(2).ValuesAsNumpy(),
        "solar_radiation_wm2":  h.Variables(3).ValuesAsNumpy(),
    })
    df["period"] = df["period"].dt.floor("h")
    return df.dropna(subset=["temperature_f"]).reset_index(drop=True)


# ── Daily temp_max per region → composite ────────────────────────────────────

def build_temp_max(region_dfs: dict[str, pd.DataFrame]) -> pd.Series:
    """Return hourly series of the max temperature across all regions."""
    temp_cols = pd.concat(
        [df.set_index("period")["temperature_f"].rename(r)
         for r, df in region_dfs.items()],
        axis=1,
    )
    return temp_cols.max(axis=1).rename("temp_max")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not EIA_KEY:
        raise RuntimeError("EIA_API_KEY not set. Add it to .env or export it.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. CAISO demand ───────────────────────────────────────────────────────
    load_out = DATA_DIR / "ca_load_historical.csv"
    if load_out.exists():
        print(f"Skipping EIA fetch — {load_out.name} already exists.")
        df_load = pd.read_csv(load_out, parse_dates=["period"])
    else:
        df_load = fetch_eia_demand(FETCH_START, FETCH_END, EIA_KEY)
        df_load.to_csv(load_out, index=False)
        print(f"Saved {load_out.name}  ({len(df_load):,} rows)")

    # ── 2. Regional weather ───────────────────────────────────────────────────
    print("\nFetching regional weather from Open-Meteo ...")
    region_dfs: dict[str, pd.DataFrame] = {}
    for region, (lat, lon) in REGION_COORDS.items():
        out_path = DATA_DIR / f"wx_{region}_historical.parquet"
        if out_path.exists():
            print(f"  Skipping {region} — parquet already exists.")
            region_dfs[region] = pd.read_parquet(out_path)
        else:
            rdf = fetch_openmeteo_region(region, lat, lon, FETCH_START, FETCH_END)
            rdf.to_parquet(out_path, index=False)
            print(f"  Saved wx_{region}_historical.parquet  ({len(rdf):,} rows)")
            region_dfs[region] = rdf
            time.sleep(1)

    # ── 3. Quick validation ───────────────────────────────────────────────────
    print("\nValidation:")
    print(f"  Load rows  : {len(df_load):,}  "
          f"{df_load['period'].min().date()} – {df_load['period'].max().date()}")
    for region, rdf in region_dfs.items():
        print(f"  {region:<14}: {len(rdf):,} rows  "
              f"{rdf['period'].min().date()} – {rdf['period'].max().date()}")

    # Expected hourly rows for 2018-01-01 – 2023-08-31 ≈ 48,528
    expected = int((pd.Timestamp(FETCH_END) - pd.Timestamp(FETCH_START)).total_seconds() / 3600)
    coverage = len(df_load) / expected * 100
    print(f"\n  Load coverage: {coverage:.1f}% of expected {expected:,} hours")
    print("\nDone. Run load_extended() to merge with 2023-2025 data.")


if __name__ == "__main__":
    main()
