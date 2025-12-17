#!/usr/bin/env python3
# ============================================================
# history_fetch.py
# Fetch multiple timeframes (1min–1month) directly from Kite,
# compute indicators, and save as CSV (no market-hour filtering).
# ============================================================

import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pytz

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

CONFIG_PATH = "config.json"  # format: {"stocks": [{"stock_code": "FORCEMOT", "instrument_token": 128008452}, ...]}
OUTPUT_DIR = "history_data"

INDIA_TZ = pytz.timezone("Asia/Kolkata")
MAX_CANDLES = 1000

# Kite API supports: minute, 3minute, 5minute, 10minute, 15minute,
# 30minute, 60minute, day, week, month
BASE_TIMEFRAMES = {
    # "1min": "minute",
    "5min": "5minute",
    "15min": "15minute",
    "30min": "30minute",
    "45min": "15minute",  # derived
    "1hour": "60minute",
    "4hour": "60minute",  # derived
    # "1day": "day",
    # "1month": "week",  # ✅ use week interval to simulate monthly-like candles
}


# ============================================================
# HELPERS
# ============================================================
def make_kite():
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise RuntimeError("❌ Missing Kite credentials in .env file.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _kite_fetch_hist(kite, token, from_date, to_date, interval):
    """Fetch historical data in 60-day chunks (Kite API limit)."""
    all_rows = []
    start = from_date
    while start <= to_date:
        end = min(start + timedelta(days=60), to_date)
        try:
            rows = kite.historical_data(
                instrument_token=token,
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
                interval=interval,
                continuous=False,
                oi=False,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"❌ Error fetching {interval} for token {token}: {e}")
        start = end + timedelta(days=1)
        time.sleep(0.25)
    return all_rows


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(INDIA_TZ)
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    return df


# ============================================================
# INDICATORS
# ============================================================
def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_ema = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_dm_ema = (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )
    minus_dm_ema = (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )

    plus_di = 100.0 * (plus_dm_ema / tr_ema.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_ema / tr_ema.replace(0, np.nan))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["+DI"] = plus_di
    out["-DI"] = minus_di
    out["ADX"] = adx
    return out


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["MA_Fast"] = df["Close"].rolling(window=9, min_periods=1).mean()
    df["MA_Slow"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # Bollinger Bands
    bb_mid = df["Close"].rolling(window=20, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=1).std(ddof=0)
    df["BB_Upper"] = bb_mid + (bb_std * 2)
    df["BB_Lower"] = bb_mid - (bb_std * 2)

    # MACD
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ADX
    adx_df = _adx_wilder(df)
    df["+DI"] = adx_df["+DI"]
    df["-DI"] = adx_df["-DI"]
    df["ADX"] = adx_df["ADX"]

    # RSI
    df["RSI14"] = _rsi(df)

    # ATR %
    df["ATR14"] = _atr_wilder(df)
    df["atr_pct"] = (df["ATR14"] / df["Close"].replace(0, np.nan)) * 100.0
    return df


# ============================================================
# SAVE HELPERS
# ============================================================
def _prepare_stock_dir(stock_code: str):
    out_dir = os.path.join(OUTPUT_DIR, f"history_data_{stock_code}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_df(
    df: pd.DataFrame, stock_code: str, tf: str, from_date: datetime, to_date: datetime
):
    if df.empty:
        print(f"⚠️ No data for {stock_code} {tf}")
        return
    out_dir = _prepare_stock_dir(stock_code)
    out_path = os.path.join(
        out_dir, f"{stock_code}_{tf}_{from_date.date()}_to_{to_date.date()}.csv"
    )
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].round(4)
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
    df.to_csv(out_path, index=False)
    print(f"✅ {stock_code} {tf}: {len(df)} candles saved → {out_path}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Generic OHLCV resampling for derived timeframes."""
    if df.empty:
        return df
    df = df.set_index("Timestamp")
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = (
        df.resample(rule, label="right", closed="right").agg(agg).dropna().reset_index()
    )
    return out


# ============================================================
# MAIN FETCH PIPELINE
# ============================================================
def fetch_all_timeframes(config_path: str):
    kite = make_kite()
    now = datetime.now(INDIA_TZ)
    from_date = now - timedelta(days=365)

    cfg = json.load(open(config_path, "r"))
    for stock in cfg.get("stocks", []):
        stock_code = stock["stock_code"]
        token = stock["instrument_token"]
        print(f"\n📥 {stock_code} ({token}) — fetching all timeframes …")

        df_cache = {}

        # Directly fetch base frames
        for tf, interval in BASE_TIMEFRAMES.items():
            if tf in ["45min", "4hour"]:
                continue  # handle later as derived

            rows = _kite_fetch_hist(kite, token, from_date, now, interval)
            if not rows:
                print(f"⚠️ No data for {stock_code} {tf}")
                continue
            df = _standardize_df(pd.DataFrame(rows))
            df = calculate_indicators(df)
            df = df.tail(MAX_CANDLES)
            df_cache[tf] = df
            _save_df(df, stock_code, tf, from_date, now)

        # Derived 45min from 15min
        if "15min" in df_cache:
            df_45 = _resample_ohlcv(df_cache["15min"], "45T")
            df_45 = calculate_indicators(df_45).tail(MAX_CANDLES)
            _save_df(df_45, stock_code, "45min", from_date, now)

        # Derived 4hour from 1hour
        if "1hour" in df_cache:
            df_4h = _resample_ohlcv(df_cache["1hour"], "4H")
            df_4h = calculate_indicators(df_4h).tail(MAX_CANDLES)
            _save_df(df_4h, stock_code, "4hour", from_date, now)


# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    fetch_all_timeframes(CONFIG_PATH)
