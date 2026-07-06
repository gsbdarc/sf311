"""Analyze SF 311 graffiti calls to evaluate the COVID hypothesis.

Loads the raw graffiti download (read-only), runs validation checks, writes
aggregated monthly/yearly counts to data/processed/, saves two figures to
figures/, and prints the key statistics used in the writeup.

Usage:
    python scripts/analyze.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

YEARS = range(2017, 2023)  # 2017..2022
# SF shelter-in-place began 2020-03-17; the strict-lockdown trough is Apr-Jun.
LOCKDOWN_MONTHS = [(2020, m) for m in (4, 5, 6)]


def latest_raw():
    files = sorted(RAW_DIR.glob("sf311_graffiti_2017_2022_*.csv"))
    if not files:
        sys.exit("ERROR: no raw file found; run scripts/download_data.py first.")
    return files[-1]


def validate(df):
    """Explicit data-integrity checks (github-for-research: check every transform)."""
    problems = []
    print("Validation")
    print(f"  rows: {len(df):,}")
    if len(df) == 0:
        problems.append("zero rows")

    dupes = df["service_request_id"].duplicated().sum()
    print(f"  duplicate service_request_id: {dupes}")
    if dupes:
        problems.append(f"{dupes} duplicate ids")

    names = set(df["service_name"].dropna().unique())
    print(f"  service_name values: {sorted(names)}")
    if not all("Graffiti" in n for n in names):
        problems.append(f"non-graffiti service_name present: {names}")

    n_bad_dt = df["requested_datetime"].isna().sum()
    print(f"  unparseable requested_datetime: {n_bad_dt}")
    yr = df["requested_datetime"].dt.year
    out_of_range = ((yr < 2017) | (yr > 2022)).sum()
    print(f"  rows outside 2017-2022: {out_of_range}")
    if out_of_range:
        problems.append(f"{out_of_range} rows outside 2017-2022")

    print(f"  date span: {df['requested_datetime'].min()} .. {df['requested_datetime'].max()}")

    if problems:
        print("  WARNING: validation issues -> " + "; ".join(problems))
        print("  (log these as a `data` issue on GitHub)")
    else:
        print("  all checks passed")
    return not problems


def main():
    raw = latest_raw()
    print(f"Loading {raw.name}\n")
    df = pd.read_csv(raw, parse_dates=["requested_datetime"])

    validate(df)

    # Keep only in-range rows for aggregation (defensive; download already filters).
    df = df[(df["requested_datetime"].dt.year >= 2017) & (df["requested_datetime"].dt.year <= 2022)]

    df["year"] = df["requested_datetime"].dt.year
    df["month"] = df["requested_datetime"].dt.month

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Aggregate ---
    yearly = df.groupby("year").size().rename("calls").reset_index()
    yearly.to_csv(PROC_DIR / "graffiti_yearly.csv", index=False)

    monthly = df.groupby(["year", "month"]).size().rename("calls").reset_index()
    monthly["date"] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    monthly = monthly.sort_values("date")
    monthly.to_csv(PROC_DIR / "graffiti_monthly.csv", index=False)

    # --- Key statistics ---
    yr_map = yearly.set_index("year")["calls"].to_dict()
    m = monthly.set_index(["year", "month"])["calls"]
    lockdown_2020 = sum(m.get(k, 0) for k in LOCKDOWN_MONTHS)
    lockdown_2019 = sum(m.get((2019, mo), 0) for _, mo in LOCKDOWN_MONTHS)
    h2_2020 = sum(m.get((2020, mo), 0) for mo in range(7, 13))
    h2_2019 = sum(m.get((2019, mo), 0) for mo in range(7, 13))

    def pct(a, b):
        return 100 * (a - b) / b if b else float("nan")

    print("\nKey statistics")
    for y in YEARS:
        print(f"  {y}: {yr_map.get(y, 0):,}")
    print(f"  2020 vs 2019 (annual):        {pct(yr_map.get(2020,0), yr_map.get(2019,0)):+.1f}%")
    print(f"  Apr-Jun 2020 vs Apr-Jun 2019: {pct(lockdown_2020, lockdown_2019):+.1f}% "
          f"({lockdown_2020:,} vs {lockdown_2019:,})")
    print(f"  Jul-Dec 2020 vs Jul-Dec 2019: {pct(h2_2020, h2_2019):+.1f}% "
          f"({h2_2020:,} vs {h2_2019:,})")

    # --- Figure 1: monthly trend with lockdown window shaded ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["date"], monthly["calls"], color="#1f4e79", lw=1.6)
    ax.axvspan(pd.Timestamp("2020-03-17"), pd.Timestamp("2020-06-30"),
               color="#d93f0b", alpha=0.15, label="SF strict lockdown\n(Mar 17 – Jun 2020)")
    ax.set_title("SF 311 graffiti calls by month, 2017–2022")
    ax.set_ylabel("calls per month")
    ax.set_xlabel("month")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "graffiti_monthly_2017_2022.png", dpi=130)
    plt.close(fig)

    # --- Figure 2: yearly bars, 2020 highlighted ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d93f0b" if y == 2020 else "#9bb8d3" for y in yearly["year"]]
    ax.bar(yearly["year"].astype(str), yearly["calls"], color=colors)
    for x, v in zip(yearly["year"].astype(str), yearly["calls"]):
        ax.text(x, v + 700, f"{v:,}", ha="center", fontsize=9)
    ax.set_title("SF 311 graffiti calls by year (2020 highlighted)")
    ax.set_ylabel("calls per year")
    ax.set_ylim(top=yearly["calls"].max() * 1.12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "graffiti_yearly_2017_2022.png", dpi=130)
    plt.close(fig)

    print(f"\nWrote processed CSVs -> {PROC_DIR}")
    print(f"Wrote figures        -> {FIG_DIR}")


if __name__ == "__main__":
    main()
