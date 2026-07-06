# SF 311 Graffiti & COVID

San Francisco 311 service-request data used to evaluate one hypothesis:

> **Did graffiti calls decrease in 2020 as a result of COVID?**

Data source: [DataSF "311 Cases"](https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6),
Socrata dataset `vw6y-z8j6`. Graffiti requests are identified by `service_name` in
(`Graffiti`, `Graffiti Public`, `Graffiti Private`). This project analyzes calls from **2017–2022**.

_Findings and figures are filled in once the analysis has run — see the branch/PR for the
`experiment/graffiti-covid-analysis` work._

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
data/raw/          downloaded once, never edited (read-only)
data/processed/    aggregated monthly/yearly counts (committed)
scripts/           download_data.py, analyze.py
notebooks/         graffiti_covid.ipynb (narrative)
figures/           generated plots
```

## Notes on the data

- 311 counts measure **reports**, not graffiti incidence. A drop in calls can reflect fewer people
  out reporting rather than less graffiti — an important caveat for the COVID question.
- Analysis starts in 2017 to avoid the 2008–2012 reporting ramp-up (mobile/Open311 adoption) that
  distorts long-run trend comparisons.
