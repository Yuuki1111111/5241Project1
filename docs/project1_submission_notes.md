# Project 1 Submission Notes

## Direct Answers to the Approved Summary Plan Questions

### 1. What is the business problem?

Retail investors in the A-share market often make timing decisions based on intuition, short-term sentiment, or fragmented news. This project studies whether daily historical market data can support a more disciplined decision process by forecasting the next trading day's closing price for `Shenghong Technology (300476.SZ)`. The purpose is not to claim guaranteed profit, but to measure whether a supervised learning model can improve on a simple naive forecast and reveal which technical factors appear most informative.

### 2. What data will be used?

The core dataset is daily stock market data from the `Tushare Pro API`, specifically `open`, `high`, `low`, `close`, `pre_close`, `vol`, and `amount` for `300476.SZ`. The plan targets roughly `5-10 years` of daily observations so the final dataset remains large enough for modeling while still manageable for course work. Optional extension data includes a broad market control variable such as `CSI300` daily return.

### 3. Is the data sufficient?

Yes, daily OHLCV data is sufficient for a baseline forecasting project because it supports lag features, rolling statistics, and common technical indicators. Additional data is optional rather than required. If available and time-aligned correctly, a market index feature can be added to capture broader market movement without changing the overall project design.

### 4. What modeling methodology will be used?

The main model is `RandomForestRegressor`, which is appropriate because the task is a supervised regression problem and the engineered predictors are tabular, nonlinear, and potentially interacting. The project will use an expanding-window walk-forward validation scheme instead of random shuffling so that training always happens before testing in time. The primary benchmark is a naive baseline where the next day's closing price is predicted to equal today's close. A simple linear regression benchmark is included for reference.

### 5. What results are expected?

The final result shows that the random forest did not outperform the naive baseline under proper time-series validation. Average performance was `RMSE 0.810`, `MAE 0.617`, and `MAPE 3.62%` for Random Forest, compared with `RMSE 0.542`, `MAE 0.397`, and `MAPE 2.35%` for the naive benchmark. This means the project conclusion is more cautious than the original expectation: daily OHLCV features alone were not sufficient to beat a simple persistence forecast for this stock. The most influential variables were still price-level features such as `close`, `low`, `high`, and `pre_close`, which suggests the model mostly captured persistence rather than a strong incremental signal.

## Final Conclusion Summary

- The pipeline was implemented successfully on `1,976` real daily observations from `2018-01-02` to `2026-02-27`.
- `Random Forest` outperformed linear regression but did not beat the naive benchmark.
- The strongest insight is methodological: proper benchmark comparison prevents overstating model usefulness.
- A better next step would be to predict return or direction, and to add market or news variables.

## Instructor Requirement Checklist

- Business problem is explicitly defined.
- Dataset is not from Kaggle or UCI; it comes from an API source.
- Model choice is one main algorithm: `Random Forest`.
- Validation avoids leakage by respecting time order.
- Evaluation includes `RMSE`, `MAE`, and optional `MAPE`.
- Slides can be built directly from the generated figures and `slides/project1_presentation.md`.
