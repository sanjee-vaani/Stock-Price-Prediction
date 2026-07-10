import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# ── Folders ───────────────────────────────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("model",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── 1. Fetch Data ─────────────────────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download stock closing prices from Yahoo Finance."""
    import yfinance as yf
    print(f"[train] Downloading {ticker} data ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Fix: yfinance sometimes returns MultiIndex columns like ("Close", "RELIANCE.NS")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    # Fix: force Close column to plain float64 (avoids numpy isnan error)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.to_csv("data/stock_data.csv")
    print(f"[train] ✓ {len(df)} rows saved to data/stock_data.csv")
    return df

# ── 2. Stationarity Check ─────────────────────────────────────────────────────
def check_stationarity(series: pd.Series) -> bool:
    """
    ADF Test: if p-value < 0.05, series is stationary (good for ARIMA).
    If not stationary, ARIMA will apply differencing automatically (d > 0).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    is_stationary = p_value < 0.05
    print(f"[train] ADF p-value: {p_value:.4f} → "
          f"{'Stationary ✓' if is_stationary else 'Not Stationary (ARIMA will difference)'}")
    return is_stationary

# ── 3. Auto-select ARIMA Order ────────────────────────────────────────────────
def find_best_order(series: pd.Series) -> tuple:
    """
    Uses pmdarima's auto_arima to test combinations of (p,d,q)
    and pick the one with the lowest AIC score.
    """
    print("[train] Searching for best ARIMA order (this may take ~30 seconds)...")
    model = auto_arima(
        series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        trace=False,
    )
    order = model.order
    print(f"[train] ✓ Best order found: ARIMA{order}  AIC={model.aic():.2f}")
    return order

# ── 4. Train ARIMA ────────────────────────────────────────────────────────────
def train_model(series: pd.Series, order: tuple):
    """Fit SARIMAX (statsmodels) with the selected order."""
    print(f"[train] Training ARIMA{order}...")
    model = SARIMAX(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    print(f"[train] ✓ Training complete  AIC={fitted.aic:.2f}")
    return fitted

# ── 5. Save Model ─────────────────────────────────────────────────────────────
def save_model(fitted_model, order: tuple, ticker: str):
    """Save model + metadata to model/arima_model.pkl using joblib."""
    payload = {
        "model":  fitted_model,
        "order":  order,
        "ticker": ticker,
    }
    joblib.dump(payload, "model/arima_model.pkl")
    print("[train] ✓ Model saved to model/arima_model.pkl")

# ── Main ──────────────────────────────────────────────────────────────────────
def main(ticker: str, start: str, end: str):
    print(f"\n{'─'*50}")
    print(f"  ML-22 · Training Pipeline  [{ticker}]")
    print(f"{'─'*50}")

    df     = fetch_data(ticker, start, end)
    series = df["Close"].squeeze()

    check_stationarity(series)
    order  = find_best_order(series)
    fitted = train_model(series, order)
    save_model(fitted, order, ticker)

    print(f"\n  Done! Now run:  python app.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL",       help="Stock ticker symbol")
    parser.add_argument("--start",  default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default="2024-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(args.ticker, args.start, args.end)



