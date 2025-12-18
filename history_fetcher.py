#!/usr/bin/env python3
# ============================================================
# history_fetcher.py — fetch historical candles (1min → 1month)
# Includes derived 45min & 4hour candle generation.
# Saves to Redis with indicators.
# ============================================================

import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pytz

from indicator import add_indicators, normalize_config
from redis_util import get_redis, save_to_redis

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
CONFIG_PATH = "config.json"
INDIA_TZ = pytz.timezone("Asia/Kolkata")
MAX_CANDLES = 100

BASE_TIMEFRAMES = {
    "1min": "minute",
    "5min": "5minute",
    "15min": "15minute",
    "30min": "30minute",
    "1hour": "60minute",
    "1day": "day",
    "1month": "month",  # built from 1day later
}


# ============================================================
# HELPERS
# ============================================================
def make_kite():
    kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))
    return kite


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
    return (
        df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    )


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cfg = {
        "moving_averages": {"ma_fast": 9, "ma_slow": 20},
        "bollinger": {"period": 20, "std_dev": 2},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "adx": {"period": 14, "threshold": 20},
        "rsi": {"period": 14},
    }
    return add_indicators(df, normalize_config(cfg))


def _kite_fetch_hist(kite, token, from_date, to_date, interval):
    """Fetches historical data in 60-day windows (Kite API limit)."""
    all_rows = []
    start = from_date
    while start <= to_date:
        end = min(start + timedelta(days=60), to_date)
        try:
            rows = kite.historical_data(
                token, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"⚠️ Error fetching {interval} for {token}: {e}")
        start = end + timedelta(days=1)
        time.sleep(0.25)
    return all_rows


def _resample_market_hours(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV while keeping only market hours (9:15–15:30)."""
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

    # Align to Indian market start (09:15)
    out = (
        df.resample(
            rule,
            label="right",
            closed="right",
            origin="start_day",
            offset="15min",
        )
        .agg(agg)
        .dropna()
        .reset_index()
    )

    # Trim to valid market hours (ending before 15:30)
    out = out[out["Timestamp"].dt.time <= datetime.strptime("15:30", "%H:%M").time()]
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True).dt.tz_convert(
        INDIA_TZ
    )
    return out


def _resample_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert 1day data → 1month candles, excluding incomplete current month."""
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    today = pd.Timestamp.now(tz=INDIA_TZ)
    df = df[df["Timestamp"].dt.to_period("M") < today.to_period("M")]

    if df.empty:
        return pd.DataFrame()

    df = (
        df.set_index("Timestamp")
        .resample("M")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(INDIA_TZ)
    df = calculate_indicators(df)
    df = df.tail(MAX_CANDLES)
    print(f"✅ Resampled {len(df)} monthly candles up to {df['Timestamp'].max()}")
    return df


# ============================================================
# MAIN
# ============================================================
def fetch_all_frames(config_path: str):
    kite = make_kite()
    r = get_redis()

    now = datetime.now(INDIA_TZ)
    from_date = now - timedelta(days=365 * 5)

    cfg = json.load(open(config_path, "r"))
    for stock in cfg.get("stocks", []):
        stock_code, token = stock["stock_code"], stock["instrument_token"]
        print(f"\n📥 {stock_code} ({token}) — Fetching all timeframes")

        df_cache = {}

        # Fetch base timeframes
        for tf, interval in BASE_TIMEFRAMES.items():
            try:
                if tf == "1month":
                    continue  # build later

                if tf == "1day":
                    end_date = now - timedelta(days=1)
                    rows = _kite_fetch_hist(kite, token, from_date, end_date, interval)
                else:
                    rows = _kite_fetch_hist(kite, token, from_date, now, interval)

                if not rows:
                    print(f"⚠️ No data for {stock_code} {tf}")
                    continue

                df = _standardize_df(pd.DataFrame(rows))
                df = calculate_indicators(df)
                df = df.tail(MAX_CANDLES)
                save_to_redis(r, stock_code, tf, df)
                df_cache[tf] = df
                print(f"✅ {stock_code} {tf}: {len(df)} candles saved")

            except Exception as e:
                print(f"❌ Error {stock_code} {tf}: {e}")

        # =====================================================
        # Build derived 45min candles (from 15min)
        # =====================================================
        try:
            if "15min" in df_cache:
                df_15 = df_cache["15min"]
                df_45 = _resample_market_hours(df_15, "45T")
                df_45 = calculate_indicators(df_45).tail(MAX_CANDLES)
                save_to_redis(r, stock_code, "45min", df_45)
                print(f"🕓 Built 45min candles for {stock_code}: {len(df_45)} saved")
        except Exception as e:
            print(f"⚠️ Failed 45min build for {stock_code}: {e}")

        # =====================================================
        # Build derived 4hour candles (from 1hour)
        # =====================================================
        try:
            if "1hour" in df_cache:
                df_1h = df_cache["1hour"]
                df_4h = _resample_market_hours(df_1h, "4H")
                df_4h = calculate_indicators(df_4h).tail(MAX_CANDLES)
                save_to_redis(r, stock_code, "4hour", df_4h)
                print(f"🕓 Built 4hour candles for {stock_code}: {len(df_4h)} saved")
        except Exception as e:
            print(f"⚠️ Failed 4hour build for {stock_code}: {e}")

        # =====================================================
        # Build monthly candles (from daily)
        # =====================================================
        try:
            if "1day" in df_cache:
                df_month = _resample_daily_to_monthly(df_cache["1day"])
                if not df_month.empty:
                    save_to_redis(r, stock_code, "1month", df_month)
                    print(
                        f"✅ {stock_code} 1month: {len(df_month)} candles saved (from daily)"
                    )
        except Exception as e:
            print(f"⚠️ Failed 1month build for {stock_code}: {e}")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    fetch_all_frames(CONFIG_PATH)
