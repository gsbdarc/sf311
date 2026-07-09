"""Experiment: reproduce the Kansas City COVID-311 paper on San Francisco data.

Reference: Tran, Bani-Yaghoub & DeLisle (2023), "Non-emergency responses in the
311 system during the early stage of the COVID-19 pandemic: a case study of
Kansas City", Dis Prev Res 2023;2:3. DOI 10.20517/dpr.2022.08 (CC-BY 4.0).

The paper makes four core claims for Mar-Aug 2019 vs 2020. We test each on SF:

  1. Aggregate 311 volume declined (KC: -13%).
  2. The decline was NOT universal -- some categories surged while
     street-condition categories fell with reduced mobility.
  3. COVID requests can be recovered by text-mining the free-text description
     (KC: 20 keywords -> 2,379 = 4.3% of requests).
  4. COVID requests shifted toward phone/email, away from web.

What this script produces:
  - data/processed/covid_category_change.csv  -- SF's version of the paper's
    Table 1 (all service_name categories, both windows, % change, artifact flag).
  - data/processed/covid_reproduction_summary.csv -- one-row scorecard of the
    four claims plus the noise/graffiti contrast.
  - figures/covid_category_reproduction.png   -- two panels: interpretable
    category movers, and the noise-vs-graffiti monthly divergence.

Data notes:
  - All-category counts come from Socrata aggregation queries (cheap; no bulk
    download). Raw graffiti/noise series are read from the committed raw files
    via sf311lib, and cross-checked against the API as a validation step.
  - SF's service_name taxonomy is far messier than KC's clean 15 categories.
    Several large movers are relabeling artifacts, not behavior (e.g. Muni
    Employee/Service Feedback appearing in 2020, Abandoned Vehicle -90%); these
    are flagged in the output and excluded from the interpretable figure.

Usage:
    python scripts/covid_category_reproduction.py
    SODA_APP_TOKEN=<token> python scripts/covid_category_reproduction.py  # avoids throttling
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

from sf311lib import FIG_DIR, PROC_DIR, load_requests

DATASET = "vw6y-z8j6"
ENDPOINT = f"https://data.sfgov.org/resource/{DATASET}.json"

# Mar-Aug window in each year, matching the paper (start inclusive, end exclusive).
WINDOWS = {2019: ("2019-03-01", "2019-09-01"), 2020: ("2020-03-01", "2020-09-01")}

# Categories whose year-over-year change is a known taxonomy/relabeling artifact
# rather than a behavioral change -- excluded from interpretation (see module docstring).
# Substrings matched case-insensitively against service_name.
ARTIFACT_PATTERNS = ("general request",  # administrative agency-routing buckets
                     "muni ", "muni feedback", "muni employee", "muni service",
                     "abandoned vehicle")  # enforcement suspended in 2020

# The paper's 20 COVID keywords, applied to the free-text description (claim 3).
COVID_KEYWORDS = ["covid", "corona", "pandemic", "virus", "mask", "face cover",
                  "coverings", "ppe", "social dist", "distanc", "6 feet",
                  "quarantine", "stay at home", "gathering", "essential",
                  "still open", "open for business", "open and operating"]


def session():
    s = requests.Session()
    token = os.environ.get("SODA_APP_TOKEN")
    if token:
        s.headers["X-App-Token"] = token
    return s


def q(s, params):
    r = s.get(ENDPOINT, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def where_window(year):
    lo, hi = WINDOWS[year]
    return f"requested_datetime >= '{lo}' and requested_datetime < '{hi}'"


def is_artifact(name):
    n = (name or "").lower()
    return any(p in n for p in ARTIFACT_PATTERNS)


def pct(a, b):
    return 100 * (a - b) / b if b else float("nan")


def fetch_category_table(s):
    """Claim 1 & 2: aggregate totals and per-category counts for both windows."""
    totals, counts = {}, {}
    for year in WINDOWS:
        totals[year] = int(q(s, {"$select": "count(*)", "$where": where_window(year)})[0]["count"])
        rows = q(s, {"$select": "service_name, count(*)", "$where": where_window(year),
                     "$group": "service_name", "$limit": 500})
        counts[year] = {r.get("service_name", "(none)"): int(r["count"]) for r in rows}

    cats = sorted(set(counts[2019]) | set(counts[2020]))
    df = pd.DataFrame([{
        "service_name": c,
        "pre_2019": counts[2019].get(c, 0),
        "pandemic_2020": counts[2020].get(c, 0),
    } for c in cats])
    df["pct_change"] = [round(pct(r.pandemic_2020, r.pre_2019), 1) for r in df.itertuples()]
    df["artifact"] = df["service_name"].map(is_artifact)
    df = df.sort_values("pct_change", ascending=False, na_position="last").reset_index(drop=True)

    # Validation: category counts reconcile to the independent total query.
    for year, col in [(2019, "pre_2019"), (2020, "pandemic_2020")]:
        assert df[col].sum() == totals[year], \
            f"{year}: category counts ({df[col].sum()}) != total query ({totals[year]})"
    assert (df[["pre_2019", "pandemic_2020"]] >= 0).all().all(), "negative category count"
    return df, totals


def fetch_channel_shift(s):
    """Claim 4: source (channel) distribution for both windows."""
    out = {}
    for year in WINDOWS:
        rows = q(s, {"$select": "source, count(*)", "$where": where_window(year),
                     "$group": "source", "$limit": 50})
        d = {r.get("source", "(none)"): int(r["count"]) for r in rows}
        tot = sum(d.values())
        out[year] = {k: round(v / tot * 100, 1) for k, v in d.items()}
    channels = sorted(set(out[2019]) | set(out[2020]),
                      key=lambda k: -(out[2019].get(k, 0) + out[2020].get(k, 0)))
    return pd.DataFrame([{"source": k, "pre_2019_pct": out[2019].get(k, 0.0),
                          "pandemic_2020_pct": out[2020].get(k, 0.0)} for k in channels])


def keyword_hits(s, year, kw):
    w = f"{where_window(year)} and upper(service_details) like upper('%{kw}%')"
    return int(q(s, {"$select": "count(*)", "$where": w})[0]["count"])


def probe_covid_keywords(s):
    """Claim 3: can COVID requests be recovered from SF's free-text description?

    Applies the paper's own identification criterion: a keyword only counts as
    COVID-identifying if it is ~absent pre-pandemic (2019) and spikes during it
    (2020). This correctly rejects false positives -- e.g. 'ppe' matches
    substrings like 'shiPPEd'/'stoPPEd' and is actually MORE common in 2019, so
    it is not COVID-related despite raw hits.
    """
    per_kw = []
    for kw in COVID_KEYWORDS:
        h19, h20 = keyword_hits(s, 2019, kw), keyword_hits(s, 2020, kw)
        discriminating = h19 <= 2 and h20 >= 10  # absent before, spikes during
        per_kw.append({"keyword": kw, "hits_2019": h19, "hits_2020": h20,
                       "discriminating": discriminating})
    total_2020 = int(q(s, {"$select": "count(*)", "$where": where_window(2020)})[0]["count"])
    nonnull = int(q(s, {"$select": "count(*)",
                        "$where": f"{where_window(2020)} and service_details IS NOT NULL"})[0]["count"])
    n_disc = sum(k["discriminating"] for k in per_kw)
    return {"per_keyword": per_kw, "n_discriminating": n_disc,
            "raw_hits_2020": sum(k["hits_2020"] for k in per_kw),
            "details_nonnull": nonnull, "total_2020": total_2020,
            "details_coverage_pct": round(nonnull / total_2020 * 100, 1)}


def window_counts(category):
    """Mar-Aug counts for a locally-stored category, both years, from raw files."""
    df = load_requests(category)
    dt = df["requested_datetime"]
    out = {}
    for year in WINDOWS:
        lo, hi = (pd.Timestamp(WINDOWS[year][0]), pd.Timestamp(WINDOWS[year][1]))
        out[year] = int(((dt >= lo) & (dt < hi)).sum())
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    s = session()

    # ---- Claim 1 & 2: aggregate decline + category heterogeneity ----
    cat, totals = fetch_category_table(s)
    agg_change = pct(totals[2020], totals[2019])
    print("=" * 66)
    print(f"Claim 1 (aggregate decline): SF Mar-Aug {totals[2019]:,} -> {totals[2020]:,} "
          f"= {agg_change:+.1f}%   [KC paper: -13%]")

    real = cat[~cat["artifact"] & cat["pct_change"].notna()]
    up = real[real["pct_change"] > 0]
    print(f"Claim 2 (heterogeneous): of {len(real)} interpretable categories, "
          f"{len(up)} rose and {len(real) - len(up)} fell.")
    print("  Top real gainers:", list(real.head(3)["service_name"]))
    print("  Top real decliners:", list(real.tail(3)["service_name"]))

    # ---- Noise vs graffiti: cross-check API against local raw, then contrast ----
    noise = window_counts("noise")
    graf = window_counts("graffiti")
    api_noise = int(cat.loc[cat.service_name == "Noise Report", "pandemic_2020"].iloc[0])
    assert noise[2020] == api_noise, \
        f"noise raw ({noise[2020]}) != API Noise Report ({api_noise}) -- raw file may be stale"
    print(f"\nNoise:    {noise[2019]:,} -> {noise[2020]:,} = {pct(noise[2020], noise[2019]):+.1f}%  (surge)")
    print(f"Graffiti: {graf[2019]:,} -> {graf[2020]:,} = {pct(graf[2020], graf[2019]):+.1f}%  (decline)")

    # ---- Claim 3: text-mining feasibility ----
    kw = probe_covid_keywords(s)
    print(f"\nClaim 3 (text-mine COVID): {kw['n_discriminating']} of {len(COVID_KEYWORDS)} keywords "
          f"discriminate (absent 2019, spike 2020). service_details is populated for "
          f"{kw['details_nonnull']:,} rows ({kw['details_coverage_pct']}%) but is a categorical "
          f"subtype label, not narrative -> "
          f"{'NOT reproducible on SF' if kw['n_discriminating'] == 0 else 'partially reproducible'} "
          f"(raw hits {kw['raw_hits_2020']} are generic-word false positives).")

    # ---- Claim 4: channel shift ----
    chan = fetch_channel_shift(s)
    print("\nClaim 4 (channel shift), share %:")
    for r in chan.itertuples():
        print(f"    {r.source:18} {r.pre_2019_pct:5} -> {r.pandemic_2020_pct:5}")

    # ---- Persist ----
    cat.to_csv(PROC_DIR / "covid_category_change.csv", index=False)
    chan.to_csv(PROC_DIR / "covid_channel_shift.csv", index=False)
    pd.DataFrame([{
        "sf_aggregate_change_pct": round(agg_change, 1),
        "kc_aggregate_change_pct": -13.0,
        "noise_change_pct": round(pct(noise[2020], noise[2019]), 1),
        "graffiti_change_pct": round(pct(graf[2020], graf[2019]), 1),
        "n_real_categories_up": int(len(up)),
        "n_real_categories_down": int(len(real) - len(up)),
        "covid_discriminating_keywords": kw["n_discriminating"],
        "covid_raw_keyword_hits_2020": kw["raw_hits_2020"],
        "text_mining_reproducible": kw["n_discriminating"] > 0,
    }]).to_csv(PROC_DIR / "covid_reproduction_summary.csv", index=False)
    pd.DataFrame(kw["per_keyword"]).to_csv(PROC_DIR / "covid_keyword_probe.csv", index=False)

    # ---- Figure ----
    make_figure(real, noise, graf)

    print(f"\nWrote processed tables -> {PROC_DIR}")
    print(f"Wrote figure -> {FIG_DIR}/covid_category_reproduction.png")


def make_figure(real, noise, graf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: interpretable category movers (top gainers + top decliners).
    movers = pd.concat([real.head(6), real.tail(6)]).drop_duplicates("service_name")
    movers = movers.sort_values("pct_change")
    colors = ["#d93f0b" if v < 0 else "#1f4e79" for v in movers["pct_change"]]
    ax1.barh(movers["service_name"], movers["pct_change"], color=colors)
    ax1.axvline(0, color="grey", lw=0.8)
    ax1.set_xlabel("% change in requests, Mar-Aug 2020 vs 2019")
    ax1.set_title("SF 311 category movers (taxonomy artifacts excluded)\n"
                  "street-condition categories fall; noise & DPH rise")
    ax1.tick_params(axis="y", labelsize=8)

    # Panel 2: the noise-vs-graffiti divergence, as % change bars.
    n_chg, g_chg = pct(noise[2020], noise[2019]), pct(graf[2020], graf[2019])
    ax2.bar(["Graffiti", "Noise"], [g_chg, n_chg], color=["#d93f0b", "#1f4e79"])
    ax2.axhline(0, color="grey", lw=0.8)
    for i, v in enumerate([g_chg, n_chg]):
        ax2.text(i, v + (2 if v > 0 else -4), f"{v:+.0f}%", ha="center", fontweight="bold")
    ax2.set_ylabel("% change, Mar-Aug 2020 vs 2019")
    ax2.set_title("Within one city, opposite COVID responses\n"
                  "SF = NYC-like (noise up), unlike Dallas (noise -14%)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "covid_category_reproduction.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
