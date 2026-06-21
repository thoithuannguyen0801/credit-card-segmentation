# Credit Card Customer Segmentation

Unsupervised segmentation of **8,950 anonymised credit-card holders** into three
behavioural personas, turned into actionable retention strategy. Built end to end in
Python with a self-contained interactive dashboard and a written report.

## Live outputs

| Output | File |
|--------|------|
| Interactive dashboard | [`dashboard/index.html`](dashboard/index.html) — open in any browser |
| Written report | [`report/index.html`](report/index.html) |

Both are self-contained (data embedded inline) — just double-click to open, or host
free on GitHub Pages.

## What's inside

```
src/analysis.py            # end-to-end, reproducible pipeline
notebooks/analysis.ipynb   # narrated notebook version
data/raw/                  # original dataset + data dictionary
data/processed/            # cleaned data, per-customer cluster labels, segment profiles
figures/                   # exported charts (PNG)
dashboard/index.html       # interactive dashboard (Chart.js)
dashboard/data.json        # summary metrics produced by analysis.py
report/index.html          # written analytical report
```

## Method

Drop ID → median-impute 314 missing values → log-transform skewed monetary fields →
`StandardScaler` → **K-Means** (k swept 2–8, scored by silhouette). `k = 3` chosen for
three distinct, actionable personas. PCA (2 components, ≈52% variance) used for
visualisation only.

## Key findings

- **Three segments:** Power Transactors (34%), Revolving Debtors (34%), Marginal Spenders (32%).
- **The revolver paradox:** the highest-value customers also carry the highest balances — most profitable yet most interest-sensitive.
- **The 12-month tenure cliff:** ~85% of customers sit at exactly 12 months — the prime moment for retention intervention.
- Full-space silhouette ≈ 0.21 — modest, as expected for overlapping real-world behaviour (see report's *Limitations*).

## Reproduce

```bash
pip install -r requirements.txt
python src/analysis.py     # regenerates data/processed, figures and dashboard/data.json
```

## Data

Anonymised credit-card usage over a 6-month window (8,950 holders, 18 fields). See
`data/raw/data_dictionary.txt`.
