#!/usr/bin/env python3
# collector.py — smart incremental candle updater with derived TF support (1min, 15min added)
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
from dotenv import load_dotenv
import pytz

from indicator import add_indicators, normalize_config
from redis_util import get_redis, save_to_redis, get_last_timestamp, load_from_redis

# ===================================================
# CONFIG
# ===================================================
load_dotenv()
CONFIG_PATH = "config.json"
INDIA_TZ = pytz.timezone("Asia/Kolkata")

TIMEFRAMES = {
    "1min": {"interval": "minute", "delta": timedelta(minutes=1)},
    "5min": {"interval": "5minute", "delta": timedelta(minutes=5)},
    "15min": {"interval": "15minute", "delta": timedelta(minutes=15)},
    "30min": {"interval": "30minute", "delta": timedelta(minutes=30)},
    "1hour": {"interval": "60minute", "delta": timedelta(hours=1)},
}

MAX_CANDLES = 500


# ===================================================
# HELPERS
# ===================================================
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


def resample_candles(df: pd.DataFrame, new_tf: str) -> pd.DataFrame:
    """Build custom candles (45min, 4hour) from smaller TFs aligned to 9:15 market start."""
    if df.empty:
        return df

    rule_map = {"45min": "45T", "4hour": "4H"}
    if new_tf not in rule_map:
        raise ValueError(f"Unsupported custom timeframe: {new_tf}")

    rule = rule_map[new_tf]

    if new_tf == "45min":
        agg_df = (
            df.resample(
                rule,
                on="Timestamp",
                label="right",
                closed="right",
                origin="start_day",
                offset="15min",
            )
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

    elif new_tf == "4hour":
        agg_df = (
            df.resample(
                rule,
                on="Timestamp",
                label="right",
                closed="right",
                origin="start_day",
                offset="15min",
            )
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
        agg_df = agg_df[
            agg_df["Timestamp"].dt.time <= datetime.strptime("15:30", "%H:%M").time()
        ]

    agg_df["Timestamp"] = pd.to_datetime(agg_df["Timestamp"], utc=True).dt.tz_convert(
        INDIA_TZ
    )
    return agg_df


def get_next_expected_close(last_ts: datetime, step: timedelta) -> datetime:
    return None if last_ts is None else last_ts + step


def should_fetch(now: datetime, next_close: datetime) -> bool:
    if not next_close:
        return True
    return now >= next_close + timedelta(seconds=30)


# ===================================================
# MAIN COLLECTOR LOOP
# ===================================================
def run_collector():
    kite = make_kite()
    r = get_redis()
    cfg = json.load(open(CONFIG_PATH, "r"))

    print("🚀 Smart Collector started. Timeframes:", list(TIMEFRAMES.keys()))

    seen, stocks = set(), []
    for s in cfg.get("stocks", []):
        code = s.get("stock_code")
        if code not in seen:
            seen.add(code)
            stocks.append(s)
    cfg["stocks"] = stocks

    while True:
        now = datetime.now(INDIA_TZ)
        print(f"\n⏱ Sync at {now:%Y-%m-%d %H:%M:%S}")

        for stock in cfg.get("stocks", []):
            stock_code = stock["stock_code"]
            token = stock["instrument_token"]

            for tf, info in TIMEFRAMES.items():
                interval, step = info["interval"], info["delta"]

                try:
                    last_ts = get_last_timestamp(r, stock_code, tf)
                    next_close = get_next_expected_close(last_ts, step)
                    next_close_str = (
                        next_close.strftime("%H:%M") if next_close else "INIT"
                    )

                    if not should_fetch(now, next_close):
                        print(
                            f"[⏩ WAIT] {stock_code} {tf}: waiting for {next_close_str}"
                        )
                        continue

                    from_date = (
                        now - timedelta(days=2) if not last_ts else last_ts + step
                    )
                    to_date = now

                    rows = kite.historical_data(
                        token,
                        from_date.strftime("%Y-%m-%d %H:%M:%S"),
                        to_date.strftime("%Y-%m-%d %H:%M:%S"),
                        interval,
                    )
                    if not rows:
                        continue

                    df_new = _standardize_df(pd.DataFrame(rows))
                    if last_ts:
                        df_new = df_new[df_new["Timestamp"] > last_ts]
                        # ensure we only store fully closed candles
                        # df_new = df_new[
                        #     df_new["Timestamp"] < datetime.now(INDIA_TZ) - step
                        # ]

                    if df_new.empty:
                        continue

                    df_old = load_from_redis(r, stock_code, tf, limit=100)
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    df_combined = calculate_indicators(df_combined).tail(MAX_CANDLES)

                    for col in [
                        "RSI14",
                        "ADX",
                        "MACD",
                        "MACD_Signal",
                        "MACD_Hist",
                        "macd_hist_delta",
                        "atr_pct",
                    ]:
                        if col in df_combined.columns:
                            df_combined[col] = df_combined[col].round(4)

                    save_to_redis(r, stock_code, tf, df_combined.tail(len(df_new)))
                    print(
                        f"✅ {stock_code} {tf}: saved {len(df_new)} → latest {df_new['Timestamp'].max().strftime('%H:%M')}"
                    )

                    # Derived TF builds only where applicable
                    if tf == "5min":
                        df_5 = load_from_redis(r, stock_code, "5min", limit=500)
                        df_45 = resample_candles(df_5, "45min")
                        df_45 = calculate_indicators(df_45).tail(MAX_CANDLES)

                        now = datetime.now(INDIA_TZ)
                        latest_ts = df_45["Timestamp"].max()
                        if latest_ts and now < latest_ts:
                            df_45 = df_45.iloc[:-1]

                        df_old_45 = load_from_redis(r, stock_code, "45min", limit=10)
                        last_saved_ts = (
                            df_old_45["Timestamp"].max()
                            if not df_old_45.empty
                            else None
                        )
                        df_to_save = (
                            df_45[df_45["Timestamp"] > last_saved_ts]
                            if last_saved_ts
                            else df_45
                        )

                        if not df_to_save.empty:
                            save_to_redis(r, stock_code, "45min", df_to_save)
                            print(
                                f"🕓 Built 45min candles for {stock_code}: {len(df_to_save)} new closed"
                            )

                    if tf == "1hour":
                        df_1h = load_from_redis(r, stock_code, "1hour", limit=500)
                        df_4h = resample_candles(df_1h, "4hour")
                        df_4h = calculate_indicators(df_4h).tail(MAX_CANDLES)
                        save_to_redis(r, stock_code, "4hour", df_4h.tail(5))
                        print(f"🕓 Built 4hour candles for {stock_code}")

                except KiteException as e:
                    print(f"❌ Kite error {stock_code} {tf}: {e}")
                except Exception as e:
                    print(
                        f"[⚠️ Unexpected error] {stock_code} {tf}: {type(e).__name__}: {e}"
                    )

        time.sleep(10)


if __name__ == "__main__":
    run_collector()
