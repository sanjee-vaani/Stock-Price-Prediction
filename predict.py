import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ── 1. Load Model ─────────────────────────────────────────────────────────────
def load_model() -> dict:
    """Load trained ARIMA model + metadata from model/arima_model.pkl."""
    if not os.path.exists("model/arima_model.pkl"):
        raise FileNotFoundError(
            "No trained model found. Please run 'python train.py' first."
        )
    payload = joblib.load("model/arima_model.pkl")
    print(f"[predict] ✓ Model loaded  ARIMA{payload['order']}  ticker={payload['ticker']}")
    return payload

# ── 2. Load Data ──────────────────────────────────────────────────────────────
def load_data() -> pd.Series:
    """Load historical stock data saved by train.py."""
    if not os.path.exists("data/stock_data.csv"):
        raise FileNotFoundError(
            "No data found. Please run 'python train.py' first."
        )
    df = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)
    # Fix: force to plain float64 to avoid numpy isnan errors
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    series = df["Close"].squeeze().dropna().astype("float64")
    print(f"[predict] ✓ Data loaded  {len(series)} rows")
    return series

# ── 3. Generate Forecast ──────────────────────────────────────────────────────
def generate_forecast(fitted_model, series: pd.Series, days: int) -> pd.DataFrame:
    """
    Forecast the next `days` business days.
    Returns a DataFrame with columns: date, forecast, lower_95, upper_95
    """
    print(f"[predict] Generating {days}-day forecast...")

    forecast_obj = fitted_model.get_forecast(steps=days)
    mean_forecast = forecast_obj.predicted_mean
    conf_int      = forecast_obj.conf_int(alpha=0.05)   # 95% CI

    # Generate future business day dates
    last_date    = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.BDay(1),
        periods=days,
        freq="B"
    )

    df_forecast = pd.DataFrame({
        "date":      future_dates,
        "forecast":  mean_forecast.values,
        "lower_95":  conf_int.iloc[:, 0].values,
        "upper_95":  conf_int.iloc[:, 1].values,
    })
    df_forecast["date"] = df_forecast["date"].dt.strftime("%Y-%m-%d")

    print(f"[predict] ✓ Forecast ready  "
          f"({df_forecast['date'].iloc[0]} → {df_forecast['date'].iloc[-1]})")
    return df_forecast

# ── 4. Evaluate on Test Data ──────────────────────────────────────────────────
def evaluate(series: pd.Series, order: tuple, test_ratio: float = 0.1) -> dict:
    """
    Walk-forward validation on last 10% of data.
    Shows how well the model would have predicted known values.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    n_test  = max(int(len(series) * test_ratio), 10)
    train   = series.iloc[:-n_test]
    test    = series.iloc[-n_test:]

    preds = []
    history = list(train)
    for i in range(len(test)):
        m   = SARIMAX(history, order=order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        res = m.fit(disp=False)
        preds.append(res.forecast(1)[0])
        history.append(test.iloc[i])

    preds  = np.array(preds)
    actual = test.values
    rmse   = np.sqrt(mean_squared_error(actual, preds))
    mae    = mean_absolute_error(actual, preds)
    mape   = np.mean(np.abs((actual - preds) / actual)) * 100

    metrics = {"RMSE": round(rmse, 3),
               "MAE":  round(mae, 3),
               "MAPE": round(mape, 2)}
    print(f"[predict] Evaluation → RMSE={metrics['RMSE']}  "
          f"MAE={metrics['MAE']}  MAPE={metrics['MAPE']}%")
    return metrics

# ── 5. Save Results ───────────────────────────────────────────────────────────
def save_results(df_forecast: pd.DataFrame):
    os.makedirs("results", exist_ok=True)
    df_forecast.to_csv("results/predictions.csv", index=False)
    print("[predict] ✓ Saved to results/predictions.csv")

# ── Main ──────────────────────────────────────────────────────────────────────
def run(days: int = 30) -> tuple[pd.DataFrame, dict]:
    """
    Full predict pipeline. Called by app.py.
    Returns (forecast_df, metrics_dict)
    """
    payload      = load_model()
    fitted_model = payload["model"]
    order        = payload["order"]

    series       = load_data()
    metrics      = evaluate(series, order)
    df_forecast  = generate_forecast(fitted_model, series, days)
    save_results(df_forecast)

    return df_forecast, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", default=30, type=int,
                        help="Number of days to forecast")
    args = parser.parse_args()

    df, metrics = run(days=args.days)
    print("\n  Forecast Preview:")
    print(df.to_string(index=False))
