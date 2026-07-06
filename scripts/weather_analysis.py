"""Experiment 2: does weather relate to graffiti / noise 311 complaints?

Merges monthly SF weather (mean temperature, total precipitation) with each
complaint's per-day rate and regresses:

    rate ~ tmean_f + precip_in + C(year) + lockdown

Year fixed effects absorb the U-shaped multi-year trend; the Apr-Jun 2020
lockdown is controlled. The weather coefficients then capture the within-year
association of complaints with temperature and rain.

Caveat: weather and season are confounded -- a warm month is also a summer
month -- so these coefficients measure "weather / time-of-year", not a weather
effect cleanly separated from the calendar.

Usage:
    python scripts/weather_analysis.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from sf311lib import FIG_DIR, PROC_DIR, monthly_rate, weather_monthly

CATS = {"graffiti": "#1f4e79", "noise": "#d93f0b"}


def build(cat, wx):
    m = monthly_rate(cat)[["rate", "year", "lockdown"]].join(wx)
    assert m.notna().all().all(), f"{cat}: NaNs after weather join"
    return m


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    wx = weather_monthly()
    print(f"Weather monthly: tmean {wx.tmean_f.min():.0f}-{wx.tmean_f.max():.0f}F, "
          f"precip/mo 0-{wx.precip_in.max():.1f}in\n")

    rows = []
    for cat in CATS:
        m = build(cat, wx)
        model = smf.ols("rate ~ tmean_f + precip_in + C(year) + lockdown", data=m).fit()
        # bivariate correlations for intuition
        r_temp, p_temp = stats.pearsonr(m["rate"], m["tmean_f"])
        r_rain, p_rain = stats.pearsonr(m["rate"], m["precip_in"])

        b_t, p_t = model.params["tmean_f"], model.pvalues["tmean_f"]
        b_p, p_p = model.params["precip_in"], model.pvalues["precip_in"]
        print(f"=== {cat} (mean {m['rate'].mean():.1f} calls/day) ===")
        print(f"  bivariate: rate~temp r={r_temp:+.2f} (p={p_temp:.3f}); "
              f"rate~precip r={r_rain:+.2f} (p={p_rain:.3f})")
        print(f"  regression (year FE + lockdown), adj R2 = {model.rsquared_adj:.2f}")
        print(f"    temperature: {b_t:+.2f} calls/day per +1F  (={b_t*10:+.1f} per +10F, p={p_t:.3f})")
        print(f"    precip:      {b_p:+.2f} calls/day per +1in rain  (p={p_p:.3f})")
        sig = lambda p: "sig" if p < 0.05 else "n.s."
        print(f"    -> temp {sig(p_t)}, precip {sig(p_p)}\n")

        rows.append({
            "category": cat,
            "corr_temp": round(r_temp, 3), "corr_precip": round(r_rain, 3),
            "beta_temp_per_10F": round(b_t * 10, 2), "p_temp": round(float(p_t), 4),
            "beta_precip_per_in": round(b_p, 2), "p_precip": round(float(p_p), 4),
            "adj_r2": round(model.rsquared_adj, 3),
        })

    pd.DataFrame(rows).to_csv(PROC_DIR / "weather_summary.csv", index=False)

    # Figure: rate vs temperature and rate vs precip, both categories (standardized rates
    # so the two very different scales overlay).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cat, color in CATS.items():
        m = build(cat, wx)
        z = (m["rate"] - m["rate"].mean()) / m["rate"].std()
        axes[0].scatter(m["tmean_f"], z, color=color, alpha=0.6, label=cat)
        axes[1].scatter(m["precip_in"], z, color=color, alpha=0.6, label=cat)
    axes[0].set_xlabel("monthly mean temperature (F)")
    axes[1].set_xlabel("monthly total precipitation (in)")
    for ax in axes:
        ax.set_ylabel("standardized calls/day (z)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.legend()
        ax.grid(alpha=0.25)
    axes[0].set_title("Complaints vs temperature")
    axes[1].set_title("Complaints vs precipitation")
    fig.suptitle("SF 311 complaint rates vs weather, monthly 2017-2022")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weather_scatter.png", dpi=130)
    plt.close(fig)

    print(f"Wrote figure  -> {FIG_DIR}/weather_scatter.png")
    print(f"Wrote summary -> {PROC_DIR}/weather_summary.csv")


if __name__ == "__main__":
    main()
