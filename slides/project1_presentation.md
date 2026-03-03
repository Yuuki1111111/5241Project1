# Project 1 Presentation

## Slide 1. Title

**Forecasting Next-Day Closing Price for Shenghong Technology (300476.SZ)**  
Course: STATGR5243 Applied Data Science  
Team member: Zhongyao Yu  
Presentation due: March 12, 2026 at 6:00 PM

Speaker note: This project tests whether a machine learning model can improve on a simple next-day price persistence rule for one A-share stock.

## Slide 2. Business Motivation

- A-share stocks are volatile and retail investors often rely on subjective timing.
- A simple data-driven forecast can support entry and exit decisions.
- The project goal is decision support and risk awareness, not guaranteed profit.

## Slide 3. Business Question

- Can we predict the next trading day's closing price for `300476.SZ`?
- Can a `Random Forest` beat a naive forecast `close(t+1) = close(t)`?
- Which historical price and volume patterns matter most?

## Slide 4. Data Source

- Source: `Tushare Pro API`
- Security: `Shenghong Technology (300476.SZ)`
- Frequency: daily
- Actual sample: `2018-01-02` to `2026-02-27`
- Total rows: `1,976`
- Raw fields: `open`, `high`, `low`, `close`, `pre_close`, `vol`, `amount`

Insert figure: `outputs/real_run/figures/price_history.png`

## Slide 5. Data Quality Checks

- Sort by trade date and remove duplicates.
- Check for missing or halted trading days.
- Flag abnormal zeros or suspicious outliers.
- Keep time ordering unchanged to avoid leakage.

## Slide 6. Feature Engineering

- Lagged prices: `close_lag_1` to `close_lag_5`
- Lagged returns and volume
- Moving averages: `MA5`, `MA10`, `MA20`
- Rolling volatility and rolling volume mean
- Technical indicators: `RSI`, `MACD`, `Bollinger Band width`
- Optional extension: market index return such as `CSI300`

## Slide 7. Target Variable

- Main target: `next trading day close`
- Equivalent view: next-day price change or return
- This is a supervised regression problem with time-dependent predictors

## Slide 8. Modeling Strategy

- Main model: `RandomForestRegressor`
- Why: captures nonlinear relationships and works well on engineered tabular data
- Benchmarks:
  - naive baseline
  - linear regression reference

Speaker note: The naive benchmark predicts tomorrow's closing price as today's close. This is hard to beat when stock prices are persistent.

## Slide 9. Validation Design

- Use `walk-forward / expanding-window` evaluation
- No random train/test shuffle
- Each test block happens strictly after its training block
- Prevent look-ahead bias

Insert figure: `outputs/real_run/figures/rmse_by_fold.png`

## Slide 10. Evaluation Metrics

- `RMSE`
- `MAE`
- `MAPE`
- Compare every model against the same folds

Actual average results:

- Random Forest: `RMSE 0.810`, `MAE 0.617`, `MAPE 3.62%`
- Naive baseline: `RMSE 0.542`, `MAE 0.397`, `MAPE 2.35%`
- Linear regression: `RMSE 0.889`, `MAE 0.684`, `MAPE 4.25%`

## Slide 11. Main Results

- The `Random Forest` did **not** beat the naive baseline on this task.
- Relative to naive, Random Forest was:
  - `49.4%` worse on RMSE
  - `55.2%` worse on MAE
  - `54.3%` worse on MAPE
- Random Forest beat naive in only `1 of 5` folds.
- Random Forest still outperformed linear regression in `3 of 5` folds and on all average metrics.

Speaker note: The main hypothesis was not supported. That is still a valid result because the evaluation was leakage-safe and benchmarked correctly.

## Slide 12. Prediction Plot

Insert figure: `outputs/real_run/figures/actual_vs_predicted_last_fold.png`

Talking point:

- In the last fold, the Random Forest often stayed too high during sharp declines.
- This suggests lagged price features were too slow to adapt to regime changes.
- The naive baseline remained competitive because daily closing prices are highly autocorrelated.

## Slide 13. Feature Importance

Insert figure: `outputs/real_run/figures/feature_importance.png`

Top features from the final fold:

- `close`: `45.0%`
- `low`: `20.0%`
- `high`: `9.1%`
- `pre_close`: `7.6%`
- `open`: `5.2%`
- `close_lag_2`: `4.1%`
- `macd_hist`: `3.6%`

Speaker note: Most predictive power came from current-day price levels rather than complex technical indicators.

## Slide 14. Key Insights

- Historical OHLCV data alone was not enough to beat a strong persistence baseline.
- Price-level variables dominated feature importance; advanced indicators added limited extra value.
- Errors increased sharply in more volatile folds, especially around early 2020.
- For this stock and target, correct benchmarking matters more than model complexity.

## Slide 15. Limitations and Future Work

- No direct news or sentiment features
- No transaction cost modeling
- A single-stock setup may overfit regime-specific behavior
- Possible next steps:
  - add index controls and industry factors
  - predict next-day return or direction instead of raw close level
  - use rolling retraining with shorter windows during high-volatility regimes
  - compare against gradient boosting if allowed

## Slide 16. Conclusion

- The project successfully built a reproducible forecasting pipeline on real A-share data.
- Under proper walk-forward validation, `Random Forest` did not outperform the naive baseline.
- The final conclusion is that short-term closing-price prediction for this stock is difficult using OHLCV data alone.
- The most defensible business takeaway is to use this model as an exploratory signal tool, not as a standalone trading rule.

Closing line: A negative modeling result is still useful because it prevents overclaiming predictive power and clarifies what additional data would be needed.
