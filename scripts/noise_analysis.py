"""Experiment 1: noise complaints & COVID, and the graffiti-noise relationship.

Part A -- Noise vs COVID: yearly/monthly noise rates, Apr-Jun 2020 vs 2019,
          contrasted with graffiti (which dipped in the lockdown).
Part B -- Do graffiti and noise co-move? Correlate the monthly per-day rates
          both raw and as STL remainders (trend + seasonality removed), since a
          shared trend would otherwise inflate the raw correlation.

Usage:
    python scripts/noise_analysis.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

from sf311lib import FIG_DIR, PROC_DIR, monthly_rate

LOCKDOWN = [4, 5, 6]  # Apr-Jun 2020 strict-lockdown trough


def pct(a, b):
    return 100 * (a - b) / b


def covid_stats(m, label):
    yearly = m.groupby("year")["calls"].sum()
    mm = m.set_index(["year", "month"])["calls"]
    ld20 = sum(mm.get((2020, mo), 0) for mo in LOCKDOWN)
    ld19 = sum(mm.get((2019, mo), 0) for mo in LOCKDOWN)
    print(f"\n[{label}] annual calls:")
    for y, v in yearly.items():
        print(f"    {y}: {v:,}")
    print(f"    2020 vs 2019 (annual):        {pct(yearly[2020], yearly[2019]):+.1f}%")
    print(f"    Apr-Jun 2020 vs 2019 (lockdown): {pct(ld20, ld19):+.1f}%  ({ld20:,} vs {ld19:,})")
    return yearly


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    graf = monthly_rate("graffiti")
    noise = monthly_rate("noise")

    # ---------- Part A: COVID ----------
    print("=" * 60, "\nPart A -- COVID effect")
    covid_stats(graf, "graffiti")
    covid_stats(noise, "noise")

    # ---------- Part B: relationship ----------
    print("\n" + "=" * 60, "\nPart B -- graffiti vs noise relationship")
    g, n = graf["rate"], noise["rate"]

    r_raw, p_raw = stats.pearsonr(g, n)
    rho_raw, _ = stats.spearmanr(g, n)
    print(f"\nRaw monthly rate correlation: Pearson r = {r_raw:+.2f} (p={p_raw:.3f}), "
          f"Spearman rho = {rho_raw:+.2f}")

    # STL remainders: strip trend + 12-month seasonality from each, correlate what's left.
    gr = STL(g, period=12, robust=True).fit().resid
    nr = STL(n, period=12, robust=True).fit().resid
    r_res, p_res = stats.pearsonr(gr, nr)
    print(f"STL-remainder correlation (trend+season removed): Pearson r = {r_res:+.2f} (p={p_res:.3f})")

    # Correlation with 2020 excluded, to see how much the COVID divergence drives it.
    mask = g.index.year != 2020
    r_ex, p_ex = stats.pearsonr(g[mask], n[mask])
    print(f"Raw correlation excluding 2020: Pearson r = {r_ex:+.2f} (p={p_ex:.3f})")

    verdict = ("weak/none" if abs(r_res) < 0.2 else
               "modest" if abs(r_res) < 0.5 else "strong")
    print(f"\nVerdict: once trend + season are removed, co-movement is {verdict} "
          f"(r={r_res:+.2f}). During the lockdown the series DIVERGE: "
          f"graffiti fell while noise rose.")

    # ---------- Figures ----------
    # (1) dual-axis time series with lockdown shaded
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(graf.index, g, color="#1f4e79", lw=1.5, label="graffiti (left)")
    ax1.set_ylabel("graffiti calls/day", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(noise.index, n, color="#d93f0b", lw=1.5, label="noise (right)")
    ax2.set_ylabel("noise calls/day", color="#d93f0b")
    ax2.tick_params(axis="y", labelcolor="#d93f0b")
    ax1.axvspan(pd.Timestamp("2020-03-17"), pd.Timestamp("2020-06-30"),
                color="grey", alpha=0.15)
    ax1.set_title("Graffiti vs noise 311 calls/day, 2017-2022 "
                  "(grey = Apr-Jun 2020 lockdown: graffiti ↓, noise ↑)")
    ax1.set_xlabel("month")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_vs_graffiti_timeseries.png", dpi=130)
    plt.close(fig)

    # (2) scatter of STL remainders
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#d93f0b" if d.year == 2020 else "#1f4e79" for d in gr.index]
    ax.scatter(gr, nr, c=colors, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("graffiti remainder (calls/day)")
    ax.set_ylabel("noise remainder (calls/day)")
    ax.set_title(f"Idiosyncratic co-movement (STL remainders)\nPearson r = {r_res:+.2f}"
                 f"   (orange = 2020)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_vs_graffiti_scatter.png", dpi=130)
    plt.close(fig)

    # ---------- Persist ----------
    pd.DataFrame([{
        "graffiti_2020_vs_2019_annual_pct": round(pct(graf.groupby("year")["calls"].sum()[2020],
                                                       graf.groupby("year")["calls"].sum()[2019]), 1),
        "noise_2020_vs_2019_annual_pct": round(pct(noise.groupby("year")["calls"].sum()[2020],
                                                    noise.groupby("year")["calls"].sum()[2019]), 1),
        "corr_raw": round(r_raw, 3),
        "corr_stl_remainder": round(r_res, 3),
        "corr_excl_2020": round(r_ex, 3),
    }]).to_csv(PROC_DIR / "noise_vs_graffiti_summary.csv", index=False)

    print(f"\nWrote figures -> {FIG_DIR}")
    print(f"Wrote summary -> {PROC_DIR}/noise_vs_graffiti_summary.csv")


if __name__ == "__main__":
    main()
