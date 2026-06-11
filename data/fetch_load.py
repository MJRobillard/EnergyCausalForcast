"""
Fetch hourly electricity demand for the WAUE balancing authority from EIA API v2.
No API key required for basic access; set EIA_API_KEY env var for higher rate limits.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
SUBBA = "WAUE"  # Western Area Power Upper Great Plains East (sub-BA under SWPP)
OUT_PATH = Path(__file__).parent / "raw" / "waue_load.csv"


def fetch_eia_hourly(start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Pull hourly demand for WAUE sub-BA between start and end (YYYY-MM-DDTHH).
    EIA returns at most 5000 rows per request, so we page in 60-day chunks.
    """
    api_key = api_key or os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "EIA API key required.\n"
            "  1. Register free at https://www.eia.gov/opendata/register.php\n"
            "  2. Run with: EIA_API_KEY=your_key python reproduce.py"
        )
    params_base = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[subba][]": SUBBA,
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
            print(f"  No data for {current} – {chunk_end}")
        else:
            all_rows.extend(data)
            print(f"  Fetched {len(data)} rows ({current.date()} to {chunk_end.date()})")
        current = chunk_end + pd.Timedelta(hours=1)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    df["period"] = pd.to_datetime(df["period"])
    df = df.rename(columns={"value": "demand_mwh"})
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    df = df[["period", "demand_mwh"]].sort_values("period").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Fetching WAUE hourly load Sep 2023 – Aug 2025 ...")
    df = fetch_eia_hourly("2023-09-01T00", "2025-08-31T23")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df.head())
    print(f"Missing: {df['demand_mwh'].isna().sum()}")
