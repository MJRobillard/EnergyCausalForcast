"""
Load historical weather parquets and derive:
  - Hub-composite temperature (load-weighted CAISO geography)
  - Riverside standalone temperature (inland extreme heat proxy)
  - Climatological normals for Aug/Sep (excluding event years)
  - 2022 heatwave peak-hour temperatures

Exposes
-------
load_hub_and_riverside()       -> (hub_df, rv_df) hourly DataFrames
compute_climatology(hub_df, rv_df, exclude_years)
                               -> (hub_clim, rv_clim) Series indexed by (month, hour)
get_heatwave_peak(hub_df, rv_df, hub_clim, rv_clim, year, month, day, hour)
                               -> HeatwavePeak namedtuple
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from pipeline.config import DATA_WX, HUB_WEIGHTS


@dataclass
class HeatwavePeak:
    """Peak-hour temperatures and anomalies for a specific event."""
    year: int
    month: int
    day: int
    hour: int
    T_hub: float           # hub-composite temperature (°F)
    T_rv: float            # Riverside temperature (°F)
    T_hub_clim: float      # climatological normal at same (month, hour)
    T_rv_clim: float
    dT_hub: float          # T_hub - T_hub_clim
    dT_rv: float           # T_rv - T_rv_clim


def load_hub_and_riverside(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hub-composite and Riverside hourly temperature DataFrames.

    Returns
    -------
    hub_df : columns [year, month, day, hour, T_hub]
    rv_df  : columns [year, month, day, hour, temperature_f]
    """
    data_dir = Path(data_dir or DATA_WX)

    wx_frames = []
    for region, weight in HUB_WEIGHTS.items():
        parquet_path = data_dir / f"wx_{region}_historical.parquet"
        wx = pd.read_parquet(parquet_path)
        wx = wx[wx["period"].dt.month.isin([8, 9])].copy()
        wx["year"]  = wx["period"].dt.year
        wx["month"] = wx["period"].dt.month
        wx["day"]   = wx["period"].dt.day
        wx["hour"]  = wx["period"].dt.hour
        wx["t_w"]   = wx["temperature_f"] * weight
        wx_frames.append(wx[["year", "month", "day", "hour", "t_w"]])

    hub_raw = wx_frames[0][["year", "month", "day", "hour"]].copy()
    hub_raw["T_hub"] = sum(f["t_w"] for f in wx_frames)

    rv = pd.read_parquet(data_dir / "wx_riverside_historical.parquet")
    rv = rv[rv["period"].dt.month.isin([8, 9])].copy()
    rv["year"]  = rv["period"].dt.year
    rv["month"] = rv["period"].dt.month
    rv["day"]   = rv["period"].dt.day
    rv["hour"]  = rv["period"].dt.hour

    return hub_raw, rv[["year", "month", "day", "hour", "temperature_f"]]


def compute_climatology(
    hub_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    exclude_years: list[int] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Aug/Sep climatological normals by (month, hour).

    Parameters
    ----------
    hub_df, rv_df    : from load_hub_and_riverside()
    exclude_years    : event years to exclude from the normal (default: [2022])

    Returns
    -------
    hub_clim, rv_clim : Series indexed by (month, hour)
    """
    exclude = set(exclude_years or [2022])

    hub_clim = (
        hub_df[~hub_df["year"].isin(exclude)]
        .groupby(["month", "hour"])["T_hub"]
        .mean()
    )
    rv_clim = (
        rv_df[~rv_df["year"].isin(exclude)]
        .groupby(["month", "hour"])["temperature_f"]
        .mean()
    )
    return hub_clim, rv_clim


def get_heatwave_peak(
    hub_df: pd.DataFrame,
    rv_df: pd.DataFrame,
    hub_clim: pd.Series,
    rv_clim: pd.Series,
    year: int = 2022,
    month: int = 9,
    day: int = 5,
    hour: int = 18,
) -> HeatwavePeak:
    """
    Extract peak-hour temperatures and anomalies for a named event.
    """
    hub_mask = (
        (hub_df["year"] == year)
        & (hub_df["month"] == month)
        & (hub_df["day"] == day)
        & (hub_df["hour"] == hour)
    )
    rv_mask = (
        (rv_df["year"] == year)
        & (rv_df["month"] == month)
        & (rv_df["day"] == day)
        & (rv_df["hour"] == hour)
    )

    T_hub  = float(hub_df.loc[hub_mask, "T_hub"].values[0])
    T_rv   = float(rv_df.loc[rv_mask, "temperature_f"].values[0])
    T_hub_c = float(hub_clim.loc[(month, hour)])
    T_rv_c  = float(rv_clim.loc[(month, hour)])

    return HeatwavePeak(
        year=year, month=month, day=day, hour=hour,
        T_hub=T_hub, T_rv=T_rv,
        T_hub_clim=T_hub_c, T_rv_clim=T_rv_c,
        dT_hub=T_hub - T_hub_c,
        dT_rv=T_rv - T_rv_c,
    )


def load_weather_for_pipeline(
    data_dir: str | Path | None = None,
    event_year: int = 2022,
    event_month: int = 9,
    event_day: int = 5,
    event_hour: int = 18,
) -> HeatwavePeak:
    """
    One-shot convenience: load weather, compute climatology, return peak.
    """
    hub_df, rv_df = load_hub_and_riverside(data_dir)
    hub_clim, rv_clim = compute_climatology(hub_df, rv_df)
    return get_heatwave_peak(
        hub_df, rv_df, hub_clim, rv_clim,
        year=event_year, month=event_month,
        day=event_day, hour=event_hour,
    )
