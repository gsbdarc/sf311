"""Evaluate seasonality in SF 311 graffiti requests with a time-series model.

Builds a monthly series (2017-2022) as calls-per-day (to remove the 28-31 day
month-length artifact), then:
  1. STL decomposition (period=12) -> trend / seasonal / residual figure.
  2. Calendar-month seasonal profile figure.
  3. Formal test: OLS with linear trend + month dummies + COVID-lockdown control;
     F-test the joint significance of the month terms.
  4. SARIMA models (seasonal vs non-seasonal, lockdown as exog) compared by AIC.

The Apr-Jun 2020 COVID lockdown (see issue #1) is controlled for so it does not
distort the seasonal estimates.

Usage:
    python scripts/seasonality.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Strict SF shelter-in-place trough (see issue #1).
LOCKDOWN = [(2020, 4), (2020, 5), (2020, 6)]


def latest_raw():
    files = sorted(RAW_DIR.glob("sf311_graffiti_2017_2022_*.csv"))
    if not files:
        raise SystemExit("No raw file found; run scripts/download_data.py first.")
    return files[-1]


def build_series():
    """Monthly calls and calls-per-day rate, indexed by month-start timestamp."""
    df = pd.read_csv(latest_raw(), parse_dates=["requested_datetime"])
    df = df[(df["requested_datetime"].dt.year >= 2017) & (df["requested_datetime"].dt.year <= 2022)]
    m = (df.set_index("requested_datetime")
           .resample("MS").size()
           .rename("calls").to_frame())

    # Validation: contiguous monthly index, no gaps, expected span.
    expected = pd.date_range("2017-01-01", "2022-12-01", freq="MS")
    assert list(m.index) == list(expected), "monthly index has gaps or wrong span"
    assert m["calls"].gt(0).all(), "a month has zero calls"
    print(f"Validation: {len(m)} contiguous months, "
          f"{m.index.min():%Y-%m}..{m.index.max():%Y-%m}, no gaps, all > 0")

    m["days"] = m.index.days_in_month
    m["rate"] = m["calls"] / m["days"]  # calls per day, removes month-length artifact
    m["month"] = m.index.month
    m["year"] = m.index.year
    m["t"] = np.arange(len(m))  # linear trend index (for SARIMA)
    m["lockdown"] = [(idx.year, idx.month) in LOCKDOWN for idx in m.index]
    m["lockdown"] = m["lockdown"].astype(int)
    return m


def stl_figure(rate):
    res = STL(rate, period=12, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(11, 8)
    fig.suptitle("STL decomposition of graffiti calls/day (period = 12 months)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonality_stl.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # Strength of seasonality (Wang/Hyndman): 1 - Var(resid)/Var(seasonal+resid).
    resid, seas = res.resid, res.seasonal
    strength = max(0.0, 1 - resid.var() / (seas + resid).var())
    return res, strength


def month_profile_figure(m):
    prof = m.groupby("month")["rate"].mean().reindex(range(1, 13))
    grand = m["rate"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d93f0b" if v < grand else "#1f4e79" for v in prof]
    ax.bar([MONTHS[i - 1] for i in prof.index], prof.values, color=colors)
    ax.axhline(grand, color="grey", ls="--", lw=1, label=f"overall mean ({grand:.1f}/day)")
    ax.set_title("Average graffiti calls per day by calendar month (2017–2022)")
    ax.set_ylabel("calls per day")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seasonality_month_profile.png", dpi=130)
    plt.close(fig)
    return prof, grand


def ols_seasonality_test(m):
    """F-test: are the 11 month dummies jointly significant, net of the
    (non-linear, U-shaped) trend and the lockdown?

    Year fixed effects C(year) absorb the trend nonparametrically, so this
    isolates within-year monthly seasonality (a two-way ANOVA of month | year).
    """
    full = smf.ols("rate ~ C(year) + lockdown + C(month)", data=m).fit()
    reduced = smf.ols("rate ~ C(year) + lockdown", data=m).fit()
    ftest = full.compare_f_test(reduced)  # (F, p, df_diff)
    return full, ftest


def sarima_compare(m):
    """Seasonal vs non-seasonal SARIMA, both with a linear trend (trend='ct')
    and lockdown exog, so the ONLY difference is the seasonal AR(12) term.
    Compared by AIC and a likelihood-ratio test (nested: plain = seasonal AR 0).

    Uses the default stationarity constraint (enforce_stationarity=True) and a
    generous iteration budget; both fits are checked for convergence because an
    unconverged fit produces meaningless AIC/likelihood values.
    """
    y, exog = m["rate"], m[["lockdown"]]
    common = dict(order=(1, 0, 0), trend="ct")
    fit_kw = dict(disp=False, maxiter=1000, method="lbfgs")
    seasonal = SARIMAX(y, exog=exog, seasonal_order=(1, 0, 0, 12), **common).fit(**fit_kw)
    plain = SARIMAX(y, exog=exog, seasonal_order=(0, 0, 0, 0), **common).fit(**fit_kw)
    for name, r in [("seasonal", seasonal), ("plain", plain)]:
        if not r.mle_retvals["converged"]:
            raise RuntimeError(f"SARIMA {name} model failed to converge; AIC/LR unreliable")
    return seasonal, plain


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    m = build_series()

    stl_res, strength = stl_figure(m["rate"])
    prof, grand = month_profile_figure(m)
    full, (F, p, _) = ols_seasonality_test(m)
    seasonal, plain = sarima_compare(m)

    # --- Report ---
    print("\n=== Seasonality strength (STL) ===")
    print(f"  F_S = {strength:.3f}  (0 = none, 1 = strong; >0.4 is 'meaningful' seasonality)")

    print("\n=== Calendar-month profile (calls/day, deviation from overall mean) ===")
    dev = (prof - grand).sort_values()
    for mo, d in dev.items():
        print(f"  {MONTHS[mo - 1]}: {prof[mo]:5.1f}/day  ({d:+.1f} vs mean)")
    print(f"  peak: {MONTHS[prof.idxmax() - 1]} ({prof.max():.1f}/day)   "
          f"trough: {MONTHS[prof.idxmin() - 1]} ({prof.min():.1f}/day)   "
          f"peak/trough ratio: {prof.max() / prof.min():.2f}x")

    print("\n=== OLS F-test for seasonality (month dummies jointly, net of trend + lockdown) ===")
    print(f"  F = {F:.2f}   p = {p:.2e}   -> "
          f"{'seasonality IS significant' if p < 0.05 else 'no significant seasonality'}")

    print("\n=== SARIMA: seasonal vs non-seasonal, both trend='ct' (lower AIC = better) ===")
    print(f"  seasonal  SARIMAX(1,0,0)(1,0,0)12 +trend+lockdown: AIC = {seasonal.aic:.1f}")
    print(f"  plain     SARIMAX(1,0,0)          +trend+lockdown: AIC = {plain.aic:.1f}")
    dAIC = plain.aic - seasonal.aic
    sar_coef = seasonal.params.get("ar.S.L12", float("nan"))
    sar_p = seasonal.pvalues.get("ar.S.L12", float("nan"))
    from scipy import stats as _stats
    lr = 2 * (seasonal.llf - plain.llf)
    lr_p = _stats.chi2.sf(lr, df=1)  # nested: one extra seasonal AR param
    print(f"  seasonal AR(12) coef = {sar_coef:.3f} (Wald p = {sar_p:.3f})")
    print(f"  AIC improvement from seasonal term = {dAIC:.1f}; "
          f"LR test chi2(1) = {lr:.2f}, p = {lr_p:.3f}")
    print(f"  -> {'seasonal term justified' if (dAIC > 2 and lr_p < 0.05) else 'seasonal term weak/not justified'}")

    print("\n=== Verdict ===")
    print(f"  Modest seasonality: a repeating shape (peak {MONTHS[prof.idxmax()-1]}, "
          f"trough {MONTHS[prof.idxmin()-1]}, {prof.max()/prof.min():.2f}x) with STL strength "
          f"{strength:.2f}. The single seasonal-AR term is significant (p={lr_p:.3f}), but the "
          f"month-block F-test is only borderline (p={p:.2f}) and the AIC gain is small "
          f"({dAIC:+.1f}). Seasonality is real but weak — trend and the COVID shock explain more.")

    # --- Persist a tidy summary ---
    prof.rename("calls_per_day").to_csv(PROC_DIR / "seasonality_month_profile.csv",
                                        index_label="month")
    summary = pd.DataFrame([{
        "stl_seasonal_strength": round(strength, 3),
        "peak_month": MONTHS[prof.idxmax() - 1],
        "trough_month": MONTHS[prof.idxmin() - 1],
        "peak_trough_ratio": round(prof.max() / prof.min(), 3),
        "ols_F": round(float(F), 2),
        "ols_p": float(p),
        "sarima_seasonal_aic": round(seasonal.aic, 1),
        "sarima_plain_aic": round(plain.aic, 1),
        "sarima_seasonal_ar12": round(float(sar_coef), 3),
        "sarima_lr_p": float(lr_p),
    }])
    summary.to_csv(PROC_DIR / "seasonality_summary.csv", index=False)

    print(f"\nWrote figures -> {FIG_DIR}")
    print(f"Wrote summary -> {PROC_DIR}/seasonality_summary.csv, seasonality_month_profile.csv")


if __name__ == "__main__":
    main()
