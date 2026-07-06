"""Download San Francisco 311 graffiti service requests (2017-2022).

Pulls from the DataSF "311 Cases" Socrata dataset (vw6y-z8j6), filtered to
graffiti requests, and writes a single date-stamped CSV to data/raw/.

Raw data is written once and never edited in place (see README). Re-running
produces a new date-stamped file rather than overwriting an old one.

Usage:
    python scripts/download_data.py
    SODA_APP_TOKEN=<token> python scripts/download_data.py   # avoids throttling
"""

import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

DATASET = "vw6y-z8j6"
ENDPOINT = f"https://data.sfgov.org/resource/{DATASET}.json"

# Inclusive of 2017, exclusive of 2023 -> full years 2017..2022.
START = "2017-01-01"
END = "2023-01-01"

WHERE = (
    "service_name like '%Graffiti%' "
    f"and requested_datetime >= '{START}' "
    f"and requested_datetime < '{END}'"
)

# Only the columns the analysis needs, keeps the file small.
SELECT = (
    "service_request_id, requested_datetime, service_name, service_subtype, "
    "supervisor_district, analysis_neighborhood, status_description, source"
)

PAGE = 50000
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_all():
    """Page through the API with $order for stable pagination."""
    session = requests.Session()
    token = os.environ.get("SODA_APP_TOKEN")
    if token:
        session.headers["X-App-Token"] = token

    rows = []
    offset = 0
    while True:
        params = {
            "$select": SELECT,
            "$where": WHERE,
            "$order": "service_request_id",  # stable order for paging
            "$limit": PAGE,
            "$offset": offset,
        }
        for attempt in range(4):
            resp = session.get(ENDPOINT, params=params, timeout=120)
            if resp.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        print(f"  fetched {len(rows):>7} rows (offset {offset})", flush=True)
        offset += PAGE
        if len(batch) < PAGE:
            break
    return rows


def main():
    print(f"Downloading graffiti 311 cases {START}..{END} from DataSF ({DATASET})")
    rows = fetch_all()
    if not rows:
        sys.exit("ERROR: no rows returned; aborting.")

    import pandas as pd

    df = pd.DataFrame(rows)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    out = RAW_DIR / f"sf311_graffiti_2017_2022_{stamp}.csv"
    df.to_csv(out, index=False)

    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {len(df):,} rows -> {out} ({size_mb:.1f} MB)")
    print("service_name values:")
    print(df["service_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
