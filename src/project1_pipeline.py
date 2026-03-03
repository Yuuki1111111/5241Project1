from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tushare_client import TushareClient


DEFAULT_TS_CODE = "300476.SZ"
DEFAULT_START_DATE = "20180101"
DEFAULT_END_DATE = "20260301"


@dataclass
class SplitResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rmse_model: float
    mae_model: float
    mape_model: float
    rmse_naive: float
    mae_naive: float
    mape_naive: float
    rmse_linear: float
    mae_linear: float
    mape_linear: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 1 stock forecasting pipeline")
    parser.add_argument("--input-csv", type=str, help="Path to a local raw data csv")
    parser.add_argument("--fetch-tushare", action="store_true", help="Fetch raw data from Tushare")
    parser.add_argument("--demo", action="store_true", help="Run the pipeline on synthetic data")
    parser.add_argument("--ts-code", type=str, default=DEFAULT_TS_CODE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=str, default="outputs/real_run")
    parser.add_argument("--raw-output-csv", type=str, default="")
    parser.add_argument("--index-code", type=str, default="", help="Optional control index, e.g. 000300.SH")
    return parser.parse_args()


def generate_demo_data(n_rows: int = 1400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_rows)

    base_returns = rng.normal(loc=0.0008, scale=0.025, size=n_rows)
    close = 20 * np.cumprod(1 + base_returns)
    open_ = close * (1 + rng.normal(0, 0.006, n_rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.02, n_rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.02, n_rows))
    pre_close = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_500_000, 9_000_000, size=n_rows).astype(float)
    amount = vol * close * rng.uniform(0.95, 1.05, n_rows)

    return pd.DataFrame(
        {
            "ts_code": DEFAULT_TS_CODE,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre_close,
            "vol": vol,
            "amount": amount,
        }
    )


def fetch_from_tushare(ts_code: str, start_date: str, end_date: str, index_code: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    client = TushareClient.from_env()
    stock = client.daily_stock(ts_code=ts_code, start_date=start_date, end_date=end_date)
    index_df = None
    if index_code:
        index_df = client.daily_index(ts_code=index_code, start_date=start_date, end_date=end_date)
    return stock, index_df


def load_raw_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    if args.demo:
        return generate_demo_data(), None, "demo"
    if args.fetch_tushare:
        stock, index_df = fetch_from_tushare(args.ts_code, args.start_date, args.end_date, args.index_code)
        return stock, index_df, "tushare"
    if args.input_csv:
        stock = pd.read_csv(args.input_csv)
        return stock, None, "csv"
    raise ValueError("Choose one input mode: --demo, --fetch-tushare, or --input-csv")


def normalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str))
    numeric_cols = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("trade_date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["trade_date"])
    return out


def prepare_optional_index(index_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if index_df is None or index_df.empty:
        return None
    idx = normalize_raw_frame(index_df)
    idx["index_return_1"] = idx["close"].pct_change()
    return idx[["trade_date", "index_return_1"]]


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(raw_df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = normalize_raw_frame(raw_df)
    df["return_1"] = df["close"].pct_change()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_spread"] = (df["open"] - df["close"]) / df["close"]

    for lag in range(1, 6):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)
        df[f"vol_lag_{lag}"] = df["vol"].shift(lag)

    for window in (5, 10, 20):
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
        df[f"volatility_{window}"] = df["return_1"].rolling(window).std()
        df[f"volume_mean_{window}"] = df["vol"].rolling(window).mean()

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["rsi_14"] = compute_rsi(df["close"], 14)

    rolling_mean_20 = df["close"].rolling(20).mean()
    rolling_std_20 = df["close"].rolling(20).std()
    df["bb_mid_20"] = rolling_mean_20
    df["bb_upper_20"] = rolling_mean_20 + 2 * rolling_std_20
    df["bb_lower_20"] = rolling_mean_20 - 2 * rolling_std_20
    df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["bb_mid_20"]

    if index_df is not None:
        df = df.merge(index_df, on="trade_date", how="left")

    df["target_next_close"] = df["close"].shift(-1)
    df["target_next_return"] = df["target_next_close"] / df["close"] - 1
    return df


def build_splits(n_rows: int, n_splits: int = 5, min_train_size: int = 252, test_size: int = 63) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start_test = min_train_size
    for _ in range(n_splits):
        end_test = start_test + test_size
        if end_test >= n_rows:
            break
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, end_test)
        splits.append((train_idx, test_idx))
        start_test = end_test
    if not splits:
        raise ValueError("Not enough rows after feature engineering to build walk-forward splits.")
    return splits


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = y_true.replace(0, np.nan)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return rmse, mae, mape


def run_modeling(feature_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = feature_df.dropna().reset_index(drop=True)
    feature_cols = [col for col in model_df.columns if col not in {"trade_date", "target_next_close", "target_next_return", "ts_code"}]
    X = model_df[feature_cols]
    y = model_df["target_next_close"]

    splits = build_splits(len(model_df))
    metrics: list[SplitResult] = []
    last_fold_predictions: pd.DataFrame | None = None
    last_model: RandomForestRegressor | None = None

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        linear = LinearRegression()

        rf.fit(X_train, y_train)
        linear.fit(X_train, y_train)

        pred_rf = rf.predict(X_test)
        pred_linear = linear.predict(X_test)
        pred_naive = model_df.iloc[test_idx]["close"].to_numpy()

        rmse_model, mae_model, mape_model = compute_metrics(y_test, pred_rf)
        rmse_naive, mae_naive, mape_naive = compute_metrics(y_test, pred_naive)
        rmse_linear, mae_linear, mape_linear = compute_metrics(y_test, pred_linear)

        metrics.append(
            SplitResult(
                fold=fold,
                train_end=model_df.iloc[train_idx[-1]]["trade_date"],
                test_start=model_df.iloc[test_idx[0]]["trade_date"],
                test_end=model_df.iloc[test_idx[-1]]["trade_date"],
                rmse_model=rmse_model,
                mae_model=mae_model,
                mape_model=mape_model,
                rmse_naive=rmse_naive,
                mae_naive=mae_naive,
                mape_naive=mape_naive,
                rmse_linear=rmse_linear,
                mae_linear=mae_linear,
                mape_linear=mape_linear,
            )
        )

        last_fold_predictions = pd.DataFrame(
            {
                "trade_date": model_df.iloc[test_idx]["trade_date"].to_numpy(),
                "actual_next_close": y_test.to_numpy(),
                "pred_rf": pred_rf,
                "pred_naive": pred_naive,
                "pred_linear": pred_linear,
            }
        )
        last_model = rf

    metrics_df = pd.DataFrame([vars(item) for item in metrics])
    summary_df = pd.DataFrame(
        {
            "metric": ["rmse", "mae", "mape"],
            "random_forest_mean": [
                metrics_df["rmse_model"].mean(),
                metrics_df["mae_model"].mean(),
                metrics_df["mape_model"].mean(),
            ],
            "naive_mean": [
                metrics_df["rmse_naive"].mean(),
                metrics_df["mae_naive"].mean(),
                metrics_df["mape_naive"].mean(),
            ],
            "linear_mean": [
                metrics_df["rmse_linear"].mean(),
                metrics_df["mae_linear"].mean(),
                metrics_df["mape_linear"].mean(),
            ],
        }
    )
    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": last_model.feature_importances_ if last_model is not None else np.nan,
        }
    ).sort_values("importance", ascending=False)

    metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    if last_fold_predictions is not None:
        last_fold_predictions.to_csv(output_dir / "predictions_last_fold.csv", index=False)

    return metrics_df, summary_df, feature_importance_df


def plot_price_history(raw_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(raw_df["trade_date"], raw_df["close"], color="#0B6E4F", linewidth=1.5)
    ax.set_title("Shenghong Technology Closing Price History")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Close Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "price_history.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(feature_df: pd.DataFrame, figure_dir: Path) -> None:
    corr_cols = [
        "close",
        "return_1",
        "close_lag_1",
        "close_lag_2",
        "return_lag_1",
        "ma_5",
        "ma_20",
        "volatility_10",
        "rsi_14",
        "macd",
        "bb_width_20",
        "target_next_close",
    ]
    existing = [col for col in corr_cols if col in feature_df.columns]
    corr = feature_df[existing].dropna().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="YlGnBu", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(feature_importance_df: pd.DataFrame, figure_dir: Path) -> None:
    top = feature_importance_df.head(12).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"], color="#1F7A8C")
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=180)
    plt.close(fig)


def plot_last_fold_predictions(output_dir: Path, figure_dir: Path) -> None:
    pred_path = output_dir / "predictions_last_fold.csv"
    pred_df = pd.read_csv(pred_path, parse_dates=["trade_date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pred_df["trade_date"], pred_df["actual_next_close"], label="Actual", linewidth=2, color="#0B6E4F")
    ax.plot(pred_df["trade_date"], pred_df["pred_rf"], label="Random Forest", linewidth=1.5, color="#CC5500")
    ax.plot(pred_df["trade_date"], pred_df["pred_naive"], label="Naive", linewidth=1.2, color="#5C677D")
    ax.set_title("Last Fold: Actual vs Predicted Next-Day Close")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Next-Day Close")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "actual_vs_predicted_last_fold.png", dpi=180)
    plt.close(fig)


def plot_fold_metrics(metrics_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics_df["fold"], metrics_df["rmse_model"], marker="o", label="Random Forest")
    ax.plot(metrics_df["fold"], metrics_df["rmse_naive"], marker="o", label="Naive")
    ax.plot(metrics_df["fold"], metrics_df["rmse_linear"], marker="o", label="Linear")
    ax.set_title("RMSE by Walk-Forward Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "rmse_by_fold.png", dpi=180)
    plt.close(fig)


def save_run_summary(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    input_mode: str,
    args: argparse.Namespace,
) -> None:
    summary = {
        "input_mode": input_mode,
        "ts_code": args.ts_code,
        "start_date": str(raw_df["trade_date"].min().date()),
        "end_date": str(raw_df["trade_date"].max().date()),
        "raw_rows": int(len(raw_df)),
        "model_rows": int(len(feature_df.dropna())),
        "folds": int(len(metrics_df)),
        "average_metrics": summary_df.to_dict(orient="records"),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def save_raw_data_if_needed(raw_df: pd.DataFrame, args: argparse.Namespace) -> Path | None:
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{args.ts_code.replace('.', '_')}_daily.csv"
    output_path = Path(args.raw_output_csv) if args.raw_output_csv else data_dir / default_name
    if args.fetch_tushare or args.demo:
        raw_df.to_csv(output_path, index=False)
        return output_path
    return None


def main() -> None:
    sns.set_theme(style="whitegrid")
    args = parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw_df, raw_index_df, input_mode = load_raw_data(args)
    raw_df = normalize_raw_frame(raw_df)
    save_raw_data_if_needed(raw_df, args)

    index_df = prepare_optional_index(raw_index_df)
    feature_df = engineer_features(raw_df, index_df=index_df)
    metrics_df, summary_df, feature_importance_df = run_modeling(feature_df, output_dir)

    plot_price_history(raw_df, figure_dir)
    plot_correlation_heatmap(feature_df, figure_dir)
    plot_feature_importance(feature_importance_df, figure_dir)
    plot_last_fold_predictions(output_dir, figure_dir)
    plot_fold_metrics(metrics_df, figure_dir)
    save_run_summary(raw_df, feature_df, metrics_df, summary_df, output_dir, input_mode, args)

    print(f"Run complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
