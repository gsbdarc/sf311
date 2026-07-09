"""Experiment: does location interact with COVID to drive noise complaints?

This repo already found SF noise 311 complaints surged ~+81% in the Apr-Jun 2020
lockdown vs 2019 (scripts/noise_analysis.py). This asks whether that average
hides spatial heterogeneity: if the mechanism is "people sheltering at home,"
the surge should concentrate in dense *residential* neighborhoods and be muted
or reversed in *commercial/office/transient* districts that emptied out.

Design:
  - Monthly noise counts per analysis_neighborhood, Apr-Jun window, 2019 vs 2020.
  - Descriptive: per-neighborhood % change (2020 vs 2019), ranked.
  - Formal test: a difference-in-differences Poisson GLM on the monthly counts,
        calls ~ C(neighborhood) * covid   with a log(days) exposure offset.
    A likelihood-ratio test of the interaction block (neighborhood:covid) vs the
    additive model asks whether location significantly moderates the COVID effect.
    The pooled `covid` coefficient gives the average lockdown surge (IRR); each
    neighborhood's own IRR is recovered from the interaction terms and ranked.
  - Cross-check the same DiD at supervisor_district resolution (fewer, larger units).

Outputs:
  - data/processed/noise_location_change.csv     -- per-neighborhood Apr-Jun 2019
    vs 2020 counts, % change, and model incidence-rate ratio (IRR).
  - data/processed/noise_location_interaction.csv -- one-row LR-test scorecard.
  - figures/noise_location_covid.png             -- per-neighborhood surge, and
    baseline-volume vs %-change scatter (residential vs commercial contrast).

Caveat: a 2019-vs-2020 Apr-Jun DiD does not separate the lockdown shock from
pre-existing neighborhood-specific trends (e.g. new construction). It answers
"did the lockdown-window change differ by location," not "was every difference
caused by COVID." ~6.6% of noise rows have a null neighborhood and are dropped.

Usage:
    python scripts/noise_location_covid.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from sf311lib import FIG_DIR, PROC_DIR, load_requests

LOCKDOWN_MONTHS = [4, 5, 6]  # Apr-Jun 2020 strict shelter-in-place, matching the repo
YEARS = (2019, 2020)
MIN_BASELINE = 20  # drop neighborhoods with < this many 2019 Apr-Jun calls (unstable %)

# Days in Apr-Jun by year (exposure offset); both non-leap-relevant (Apr30+May31+Jun30=91).
WINDOW_DAYS = {2019: 91, 2020: 91}


def monthly_panel():
    """Neighborhood x (year, month) noise counts for the Apr-Jun window, 2019 & 2020."""
    df = load_requests("noise").dropna(subset=["analysis_neighborhood"]).copy()
    df["year"] = df["requested_datetime"].dt.year
    df["month"] = df["requested_datetime"].dt.month
    df = df[df["year"].isin(YEARS) & df["month"].isin(LOCKDOWN_MONTHS)]

    g = (df.groupby(["analysis_neighborhood", "year", "month"])
           .size().rename("calls").reset_index())
    # Fill implicit zeros: every kept neighborhood should have all 6 (year,month) cells.
    hoods = g["analysis_neighborhood"].unique()
    full = pd.MultiIndex.from_product(
        [hoods, YEARS, LOCKDOWN_MONTHS],
        names=["analysis_neighborhood", "year", "month"]).to_frame(index=False)
    g = full.merge(g, how="left").fillna({"calls": 0})
    g["calls"] = g["calls"].astype(int)
    g["covid"] = (g["year"] == 2020).astype(int)
    g["days"] = g["month"].map({4: 30, 5: 31, 6: 30})
    return g


def window_change(panel):
    """Per-neighborhood Apr-Jun total, both years, and % change."""
    w = (panel.groupby(["analysis_neighborhood", "year"])["calls"].sum()
              .unstack("year").rename(columns={2019: "y2019", 2020: "y2020"}))
    w["pct_change"] = (w["y2020"] - w["y2019"]) / w["y2019"].replace(0, np.nan) * 100
    return w.sort_values("pct_change", ascending=False)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    panel = monthly_panel()
    change = window_change(panel)

    # Keep neighborhoods with a stable baseline for both the model and the ranking.
    keep = change.index[change["y2019"] >= MIN_BASELINE]
    dropped = len(change) - len(keep)
    model_df = panel[panel["analysis_neighborhood"].isin(keep)].copy()
    change = change.loc[keep]

    # --- Validation ---
    assert model_df["calls"].ge(0).all(), "negative counts"
    assert model_df["analysis_neighborhood"].notna().all(), "null neighborhood in model frame"
    assert len(model_df) == len(keep) * len(YEARS) * len(LOCKDOWN_MONTHS), \
        "panel is not balanced (missing neighborhood-month cells)"
    # Descriptive window totals must reconcile with the panel.
    assert change["y2020"].sum() == model_df[model_df.covid == 1]["calls"].sum(), \
        "window change and panel disagree on 2020 total"

    print("=" * 66)
    print(f"Neighborhoods: {len(keep)} kept (baseline >= {MIN_BASELINE}), {dropped} dropped.")
    pooled = (change["y2020"].sum() - change["y2019"].sum()) / change["y2019"].sum() * 100
    print(f"Pooled Apr-Jun noise change (kept hoods): {pooled:+.1f}%")

    # --- Formal DiD: Poisson GLM with neighborhood x covid interaction ---
    # Counts are overdispersed (phi ~ 3), so the naive Poisson LR test is far too
    # liberal. We report the quasi-Poisson F-test (LR scaled by the dispersion
    # phi) as the headline inference -- the standard remedy for overdispersion.
    off = np.log(model_df["days"].values)
    full = smf.glm("calls ~ C(analysis_neighborhood) * covid", data=model_df,
                   family=sm.families.Poisson(), offset=off).fit()
    reduced = smf.glm("calls ~ C(analysis_neighborhood) + covid", data=model_df,
                      family=sm.families.Poisson(), offset=off).fit()
    lr = 2 * (full.llf - reduced.llf)
    ddf = int(reduced.df_resid - full.df_resid)
    p_poisson = stats.chi2.sf(lr, ddf)              # naive (inflated by overdispersion)
    phi = full.pearson_chi2 / full.df_resid         # dispersion of the full model
    f_stat = (lr / ddf) / phi                        # quasi-Poisson F statistic
    p_int = stats.f.sf(f_stat, ddf, full.df_resid)   # overdispersion-corrected
    irr_pooled = np.exp(reduced.params["covid"])
    print(f"\nDiD Poisson GLM (calls ~ neighborhood * covid, log-days offset):")
    print(f"  Pooled COVID effect (additive model): IRR = {irr_pooled:.2f} "
          f"({(irr_pooled - 1) * 100:+.0f}% avg)")
    print(f"  Overdispersion phi = {phi:.2f}  (naive Poisson p = {p_poisson:.1e} is too liberal)")
    print(f"  Interaction, quasi-Poisson F-test: F({ddf},{int(full.df_resid)}) = {f_stat:.2f}, "
          f"p = {p_int:.2e}")
    verdict = "significant" if p_int < 0.05 else "not significant"
    print(f"  -> location {verdict}ly moderates the COVID noise effect "
          f"(robust to {phi:.1f}x overdispersion).")

    # Per-neighborhood IRR (2020/2019) from the saturated interaction model.
    change["irr_2020_vs_2019"] = _per_hood_irr(full, keep)

    # --- Supervisor-district cross-check ---
    dist_p = _district_test()
    print(f"\nCross-check at supervisor_district resolution (quasi-Poisson): "
          f"F({dist_p['ddf']}) = {dist_p['F']:.1f}, p = {dist_p['p']:.2e} "
          f"({'significant' if dist_p['p'] < 0.05 else 'n.s.'}).")

    # --- Persist ---
    change.round(1).to_csv(PROC_DIR / "noise_location_change.csv")
    pd.DataFrame([{
        "n_neighborhoods": len(keep),
        "pooled_covid_pct": round(pooled, 1),
        "pooled_irr": round(irr_pooled, 3),
        "dispersion_phi": round(phi, 2),
        "interaction_F": round(f_stat, 2),
        "interaction_df": ddf,
        "interaction_p_quasipoisson": p_int,
        "interaction_p_naive_poisson": p_poisson,
        "interaction_significant": bool(p_int < 0.05),
        "district_interaction_p": dist_p["p"],
        "pct_change_min": round(change["pct_change"].min(), 1),
        "pct_change_max": round(change["pct_change"].max(), 1),
    }]).to_csv(PROC_DIR / "noise_location_interaction.csv", index=False)

    make_figure(change)
    print(f"\nWrote tables -> {PROC_DIR}")
    print(f"Wrote figure -> {FIG_DIR}/noise_location_covid.png")


def _per_hood_irr(full, keep):
    """Recover each neighborhood's 2020/2019 incidence-rate ratio from the model.

    In C(neighborhood)*covid the ratio for the reference neighborhood is exp(covid);
    for the others it is exp(covid + neighborhood:covid interaction term).
    """
    base = full.params["covid"]
    irr = {}
    ref = None
    for h in keep:
        key = f"C(analysis_neighborhood)[T.{h}]:covid"
        if key in full.params.index:
            irr[h] = np.exp(base + full.params[key])
        else:
            ref = h  # the reference level has no interaction term
            irr[h] = np.exp(base)
    return pd.Series(irr).reindex(keep)


def _district_test():
    """Same DiD at supervisor_district resolution."""
    df = load_requests("noise").dropna(subset=["supervisor_district"]).copy()
    df["year"] = df["requested_datetime"].dt.year
    df["month"] = df["requested_datetime"].dt.month
    df = df[df["year"].isin(YEARS) & df["month"].isin(LOCKDOWN_MONTHS)]
    df["district"] = df["supervisor_district"].astype(int).astype(str)
    g = df.groupby(["district", "year", "month"]).size().rename("calls").reset_index()
    idx = pd.MultiIndex.from_product(
        [g["district"].unique(), YEARS, LOCKDOWN_MONTHS],
        names=["district", "year", "month"]).to_frame(index=False)
    g = idx.merge(g, how="left").fillna({"calls": 0})
    g["covid"] = (g["year"] == 2020).astype(int)
    off = np.log(g["month"].map({4: 30, 5: 31, 6: 30}).values)
    full = smf.glm("calls ~ C(district) * covid", data=g,
                   family=sm.families.Poisson(), offset=off).fit()
    reduced = smf.glm("calls ~ C(district) + covid", data=g,
                      family=sm.families.Poisson(), offset=off).fit()
    lr = 2 * (full.llf - reduced.llf)
    ddf = int(reduced.df_resid - full.df_resid)
    phi = full.pearson_chi2 / full.df_resid  # same overdispersion correction as neighborhoods
    f_stat = (lr / ddf) / phi
    return {"F": f_stat, "ddf": ddf, "p": stats.f.sf(f_stat, ddf, full.df_resid)}


def make_figure(change):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Panel 1: per-neighborhood % change, colored by sign.
    c = change.sort_values("pct_change")
    colors = ["#d93f0b" if v < 0 else "#1f4e79" for v in c["pct_change"]]
    ax1.barh(c.index, c["pct_change"], color=colors)
    ax1.axvline(0, color="grey", lw=0.8)
    ax1.set_xlabel("% change in noise complaints, Apr-Jun 2020 vs 2019")
    ax1.set_title("Noise surge by neighborhood\nresidential ↑, commercial/downtown ↓")
    ax1.tick_params(axis="y", labelsize=8)

    # Panel 2: baseline volume vs % change (the two decliners are the emptied-out cores).
    ax2.scatter(change["y2019"], change["pct_change"], color="#1f4e79", alpha=0.75)
    ax2.axhline(0, color="grey", lw=0.8)
    for h, r in change.iterrows():
        if r["pct_change"] < 0 or r["pct_change"] > 250 or r["y2019"] > 120:
            ax2.annotate(h, (r["y2019"], r["pct_change"]), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
    ax2.set_xlabel("2019 Apr-Jun noise complaints (baseline volume)")
    ax2.set_ylabel("% change 2020 vs 2019")
    ax2.set_title("Only the commercial/office cores fell\n(FiDi/South Beach, Mission Bay, Chinatown)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_location_covid.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
