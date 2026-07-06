# SF 311 Graffiti & COVID

San Francisco 311 service-request data used to evaluate one hypothesis:

> **Did graffiti calls decrease in 2020 as a result of COVID?**

Data source: [DataSF "311 Cases"](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6),
Socrata dataset `vw6y-z8j6`. Graffiti requests are identified by `service_name` in
(`Graffiti`, `Graffiti Public`, `Graffiti Private`). This project analyzes calls from **2017–2022**.

## Findings

**Short answer: nuanced — yes for the lockdown, no for the year.**

Graffiti calls fell sharply during San Francisco's strict spring-2020 shelter-in-place, but
rebounded so strongly afterward that the full-year total ended *above* 2019.

| Window | 2020 | 2019 | Change |
|---|---|---|---|
| **Apr–Jun** (strict lockdown) | 9,828 | 14,362 | **−31.6%** |
| **Jul–Dec** (rebound) | 32,174 | 26,785 | **+20.1%** |
| **Full year** | 59,051 | 57,035 | **+3.5%** |

![Monthly graffiti calls, 2017–2022](figures/graffiti_monthly_2017_2022.png)

The shaded lockdown window (Mar 17 – Jun 2020) is the deepest trough in the entire 2017–2022
series — calls dropped to ~3,150/month, well below the usual 4,500–6,500 — then recovered to
above-2019 levels by late 2020.

![Yearly graffiti calls, 2017–2022](figures/graffiti_yearly_2017_2022.png)

**Conclusion:** COVID produced a clear but *short-lived* decline in graffiti 311 calls during the
strict lockdown; it did **not** reduce the full-year total. So the hypothesis holds only for the
lockdown window, not for 2020 as a whole.

**Caveat:** 311 counts measure *reports*, not graffiti *incidence*. With far fewer people out during
the strict lockdown, the drop plausibly reflects reduced **reporting** (and less-frequented public
space) rather than less graffiti being created. These data show *when* calls fell, not *why*.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py     # -> data/raw/sf311_graffiti_2017_2022_<date>.csv
python scripts/analyze.py           # -> data/processed/*.csv, figures/*.png
jupyter notebook notebooks/graffiti_covid.ipynb
```

An optional DataSF app token avoids API throttling: `export SODA_APP_TOKEN=...`

## Layout

```
data/raw/          NOT committed — regenerate with download_data.py (git-ignored)
data/processed/    aggregated monthly/yearly counts (committed)
scripts/           download_data.py, analyze.py
notebooks/         graffiti_covid.ipynb (narrative)
figures/           generated plots
```

**Raw data is not stored in git.** It is fully regenerable from DataSF (`vw6y-z8j6`) — run
`python scripts/download_data.py` to recreate `data/raw/sf311_graffiti_2017_2022_<date>.csv`
before running the analysis. This keeps the repo lean (git already compresses the CSV, so there's
nothing to gain by committing or zipping it) and treats DataSF as the authoritative source.

## Notes on the data

- 311 counts measure **reports**, not graffiti incidence. A drop in calls can reflect fewer people
  out reporting rather than less graffiti — an important caveat for the COVID question.
- Analysis starts in 2017 to avoid the 2008–2012 reporting ramp-up (mobile/Open311 adoption) that
  distorts long-run trend comparisons.
