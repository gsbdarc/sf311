"""Experiment 3: weekly-resolution seasonality for graffiti and noise.

Redoes the seasonality question at weekly (not monthly) resolution, for both
categories:
  1. STL with period=52 -> seasonal strength F_S (compare to the monthly result).
  2. Week-of-year profile (finer within-year timing than 12 monthly buckets).
  3. Day-of-week profile from the raw daily data (intra-week structure).

All series use per-day rates so unequal bin lengths don't distort them.

Usage:
    python scripts/seasonality_weekly.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from sf311lib import FIG_DIR, PROC_DIR, load_requests, weekly_rate

CATS = {"graffiti": "#1f4e79", "noise": "#d93f0b"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def stl_strength(rate):
    res = STL(rate, period=52, robust=True).fit()
    resid, seas = res.resid, res.seasonal
    return max(0.0, 1 - resid.var() / (seas + resid).var())


def week_profile(w):
    """Mean per-day rate by ISO week-of-year (weeks 1-52)."""
    return w[w["week"] <= 52].groupby("week")["rate"].mean().reindex(range(1, 53))


def dow_profile(cat):
    """Mean calls per weekday, from raw daily data (0=Mon..6=Sun)."""
    df = load_requests(cat)
    counts = df.groupby(df["requested_datetime"].dt.dayofweek).size()
    alldays = pd.date_range("2017-01-01", "2022-12-31", freq="D")
    n_each = pd.Series(alldays.dayofweek).value_counts().sort_index()
    return (counts / n_each).reindex(range(7))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    wk_profiles, dw_profiles = {}, {}
    for cat in CATS:
        w = weekly_rate(cat)
        strength = stl_strength(w["rate"])
        wk = week_profile(w)
        dw = dow_profile(cat)
        wk_profiles[cat], dw_profiles[cat] = wk, dw

        peak_wk, trough_wk = wk.idxmax(), wk.idxmin()
        wk_ratio = wk.max() / wk.min()
        wke = dw.iloc[5:].mean() / dw.iloc[:5].mean()  # weekend vs weekday
        print(f"=== {cat} ===")
        print(f"  weekly STL seasonal strength F_S = {strength:.2f}  ({len(w)} weeks)")
        print(f"  week-of-year: peak wk {peak_wk} ({wk.max():.1f}/day), "
              f"trough wk {trough_wk} ({wk.min():.1f}/day), ratio {wk_ratio:.2f}x")
        print(f"  day-of-week: {dict(zip(DOW, dw.round(1)))}")
        print(f"    weekend/weekday ratio = {wke:.2f}x\n")

        rows.append({
            "category": cat,
            "weekly_stl_strength": round(strength, 3),
            "peak_week": int(peak_wk), "trough_week": int(trough_wk),
            "week_peak_trough_ratio": round(wk_ratio, 3),
            "weekend_weekday_ratio": round(wke, 3),
        })

    pd.DataFrame(rows).to_csv(PROC_DIR / "weekly_seasonality_summary.csv", index=False)

    # Figure 1: week-of-year profiles, standardized so the two shapes compare.
    fig, ax = plt.subplots(figsize=(12, 5))
    for cat, color in CATS.items():
        wk = wk_profiles[cat]
        z = (wk - wk.mean()) / wk.std()
        ax.plot(wk.index, z, color=color, lw=1.8, label=cat)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Week-of-year seasonal profile (standardized), 2017–2022")
    ax.set_xlabel("ISO week of year")
    ax.set_ylabel("standardized calls/day (z)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_seasonality_weekofyear.png", dpi=130)
    plt.close(fig)

    # Figure 2: day-of-week profile, indexed to each category's own mean (=100).
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(7)
    width = 0.4
    for i, (cat, color) in enumerate(CATS.items()):
        dw = dw_profiles[cat]
        idx = 100 * dw / dw.mean()
        ax.bar(x + (i - 0.5) * width, idx.values, width, color=color, label=cat)
    ax.axhline(100, color="grey", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DOW)
    ax.set_title("Day-of-week profile (100 = each category's own daily mean)")
    ax.set_ylabel("index (mean = 100)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_seasonality_dayofweek.png", dpi=130)
    plt.close(fig)

    print(f"Wrote figures -> {FIG_DIR}/weekly_seasonality_weekofyear.png, weekly_seasonality_dayofweek.png")
    print(f"Wrote summary -> {PROC_DIR}/weekly_seasonality_summary.csv")


if __name__ == "__main__":
    main()
