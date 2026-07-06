"""Download daily San Francisco weather (2017-2022) from the Open-Meteo archive.

Free, no API key. Writes a date-stamped CSV to data/raw/ (raw data is never
edited in place). Columns: date, tmax_f, tmin_f, tmean_f, precip_in.

Usage:
    python scripts/download_weather.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Downtown San Francisco.
LAT, LON = 37.7749, -122.4194
START, END = "2017-01-01", "2022-12-31"
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"


def main():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/Los_Angeles",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }
    print(f"Fetching SF daily weather {START}..{END} from Open-Meteo archive")
    resp = requests.get(ENDPOINT, params=params, timeout=120)
    resp.raise_for_status()
    d = resp.json()["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "tmax_f": d["temperature_2m_max"],
        "tmin_f": d["temperature_2m_min"],
        "precip_in": d["precipitation_sum"],
    })
    df["tmean_f"] = (df["tmax_f"] + df["tmin_f"]) / 2

    # Validation: contiguous daily coverage, sane ranges.
    expected = pd.date_range(START, END, freq="D")
    assert list(df["date"]) == list(expected), "weather dates have gaps"
    assert df["tmax_f"].between(30, 115).all(), "tmax out of plausible SF range"
    assert (df["precip_in"] >= 0).all(), "negative precipitation"
    print(f"Validation: {len(df)} contiguous days, "
          f"tmean {df['tmean_f'].min():.0f}-{df['tmean_f'].max():.0f}F, "
          f"precip total {df['precip_in'].sum():.1f}in, "
          f"missing values: {int(df.isna().sum().sum())}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    out = RAW_DIR / f"sf_weather_2017_2022_{stamp}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out.name}")


if __name__ == "__main__":
    main()
