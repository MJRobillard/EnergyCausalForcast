"""
Fetch hourly weather data for WAUE representative coordinates from Open-Meteo
Historical Weather API (ERA5 reanalysis). No API key required.
Coordinates from paper: 44.6321 N, -100.2753 W (central South Dakota).
"""

import requests
import pandas as pd
from pathlib import Path

LAT = 44.6321
LON = -100.2753
OUT_PATH = Path(__file__).parent / "raw" / "waue_weather.csv"

VARIABLES = [
    "temperature_2m",
    "relativehumidity_2m",
    "windspeed_10m",
    "shortwave_radiation",
]


def fetch_open_meteo(start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis weather for the WAUE centroid.
    start/end format: YYYY-MM-DD
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "America/Chicago",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "period": pd.to_datetime(hourly["time"]),
        "temperature_f": hourly["temperature_2m"],
        "humidity_pct": hourly["relativehumidity_2m"],
        "wind_mph": hourly["windspeed_10m"],
        "solar_radiation_wm2": hourly["shortwave_radiation"],
    })
    return df


if __name__ == "__main__":
    print("Fetching Open-Meteo ERA5 weather Sep 2023 – Aug 2025 ...")
    df = fetch_open_meteo("2023-09-01", "2025-08-31")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df.head())
    print(df.describe())
