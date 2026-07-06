"""Shared helpers for the SF 311 analyses: locate raw files and build tidy
monthly / weekly time series (as calls-per-day rates) for a given category.

Modeling calls *per day* removes the calendar artifact that months (28-31 days)
and partial ISO weeks would otherwise inject into the counts. The Apr-Jun 2020
COVID lockdown flag is attached for use as a control.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

LOCKDOWN_MONTHS = [(2020, 4), (2020, 5), (2020, 6)]  # SF strict shelter-in-place trough


def latest_raw(category):
    """Newest date-stamped raw CSV for a category (e.g. 'graffiti', 'noise')."""
    files = sorted(RAW_DIR.glob(f"sf311_{category}_2017_2022_*.csv"))
    if not files:
        raise SystemExit(f"No raw file for '{category}'; run "
                         f"`python scripts/download_data.py {category}` first.")
    return files[-1]


def load_requests(category):
    """Raw requests for a category, filtered to full years 2017-2022."""
    df = pd.read_csv(latest_raw(category), parse_dates=["requested_datetime"])
    yr = df["requested_datetime"].dt.year
    return df[(yr >= 2017) & (yr <= 2022)].copy()


def monthly_rate(category):
    """Month-start-indexed frame: calls, days, rate (calls/day), and covariates."""
    df = load_requests(category)
    m = (df.set_index("requested_datetime").resample("MS").size()
           .rename("calls").to_frame())
    expected = pd.date_range("2017-01-01", "2022-12-01", freq="MS")
    assert list(m.index) == list(expected), f"{category}: monthly index has gaps"
    assert m["calls"].gt(0).all(), f"{category}: a month has zero calls"
    m["days"] = m.index.days_in_month
    m["rate"] = m["calls"] / m["days"]
    m["year"] = m.index.year
    m["month"] = m.index.month
    m["t"] = np.arange(len(m))
    m["lockdown"] = [(i.year, i.month) in LOCKDOWN_MONTHS for i in m.index]
    m["lockdown"] = m["lockdown"].astype(int)
    return m


def weather_daily():
    """Daily SF weather frame (date, tmax_f, tmin_f, tmean_f, precip_in)."""
    files = sorted(RAW_DIR.glob("sf_weather_2017_2022_*.csv"))
    if not files:
        raise SystemExit("No weather file; run `python scripts/download_weather.py` first.")
    return pd.read_csv(files[-1], parse_dates=["date"])


def weather_monthly():
    """Month-start-indexed weather: mean temp, total precip, count of rainy days."""
    w = weather_daily().set_index("date")
    g = w.resample("MS").agg(
        tmean_f=("tmean_f", "mean"),
        tmax_f=("tmax_f", "mean"),
        precip_in=("precip_in", "sum"),
    )
    g["rain_days"] = w["precip_in"].gt(0.01).resample("MS").sum()
    return g


def weekly_rate(category, drop_partial=True):
    """Week-start-indexed frame: calls, days, rate (calls/day), and covariates.

    Weeks are Monday-anchored ('W-SUN' bins are Sun-anchored; we use 'W-MON'
    start). Partial weeks at the ends are dropped when drop_partial=True so the
    per-day rate isn't distorted, then validated for a gap-free 7-day cadence.
    """
    df = load_requests(category)
    w = (df.set_index("requested_datetime").resample("W-SUN").size()
           .rename("calls").to_frame())
    # Each 'W-SUN' label is the week-ending Sunday; count the days present.
    w["days"] = 7
    if drop_partial:
        # First and last bins may be partial (data starts 2017-01-01, ends 2022-12-31).
        w = w.iloc[1:-1]
    w["rate"] = w["calls"] / w["days"]
    w["year"] = w.index.year
    w["week"] = w.index.isocalendar().week.astype(int)
    w["t"] = np.arange(len(w))
    # gap check: consecutive weekly cadence
    gaps = w.index.to_series().diff().dropna().dt.days
    assert (gaps == 7).all(), f"{category}: weekly index not contiguous"
    return w
