# Project 1: A-Share Stock Forecasting for Shenghong Technology (300476.SZ)

This repository turns the approved Project 1 summary plan into a reproducible course project package for `STATGR5243 Applied Data Science`.

## Project Goal

Build a supervised regression model that predicts the next trading day's closing price for `Shenghong Technology (300476.SZ)` using historical market data and engineered technical indicators. The project focuses on quantitative decision support, not trading advice.

## What Is Included

- `src/project1_pipeline.py`: end-to-end training and evaluation pipeline
- `src/tushare_client.py`: lightweight Tushare HTTP client using `requests`
- `slides/project1_presentation.md`: slide-by-slide content for the final presentation
- `docs/project1_submission_notes.md`: concise write-up aligned with the instructor prompts
- `requirements.txt`: minimal Python dependencies

## Repository Layout

```text
5243project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── project1_submission_notes.md
├── outputs/
│   ├── demo/
│   └── real_run/
├── slides/
│   └── project1_presentation.md
└── src/
    ├── project1_pipeline.py
    └── tushare_client.py
```

## Environment Setup

```bash
python -m pip install -r requirements.txt
```

## Running With Real Data

The pipeline supports either a local CSV file or direct download from the Tushare HTTP API.

### Option 1: Fetch from Tushare

1. Create a Tushare account and get your token.
2. Export the token:

```bash
export TUSHARE_TOKEN="your_token_here"
```

3. Run:

```bash
python src/project1_pipeline.py \
  --fetch-tushare \
  --ts-code 300476.SZ \
  --start-date 20180101 \
  --end-date 20260301 \
  --output-dir outputs/real_run
```

The raw data will be saved to `data/raw/300476_SZ_daily.csv`.

### Option 2: Use a Local CSV

If you already exported historical data, place it at `data/raw/300476_SZ_daily.csv` with these columns:

- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `vol`
- `amount`

Then run:

```bash
python src/project1_pipeline.py \
  --input-csv data/raw/300476_SZ_daily.csv \
  --output-dir outputs/real_run
```

## Offline Demo Run

The current environment cannot reach external APIs, so the pipeline includes a synthetic data mode for smoke testing.

```bash
python src/project1_pipeline.py --demo --output-dir outputs/demo
```

This verifies that the feature engineering, walk-forward validation, chart generation, and metrics export work end to end. Do not present demo results as real stock results.

## Expected Outputs

After a successful run, the pipeline writes:

- `metrics_by_fold.csv`
- `metrics_summary.csv`
- `feature_importance.csv`
- `predictions_last_fold.csv`
- `run_summary.json`
- figures under `figures/`

## Notes

- The project uses `RandomForestRegressor` as the main model, matching the approved plan.
- The baseline is the naive forecast `close_{t+1} = close_t`.
- Validation uses an expanding-window time split to avoid leakage.
- If time allows, you can extend the model with `CSI300` index returns as a control variable.
