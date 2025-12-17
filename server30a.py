#!/usr/bin/env python3
# server30a.py — Smart Real-Time GPT Forecast Server (Collector-aligned)
# Runs GPT only after confirming new candle data is present in Redis.
# Prevents premature GPT runs and duplicate processing.

import os
import time
import json
import random
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# === Local imports ===
from telegram_alert30a import (
    send_telegram_message,
    send_error_alert,
    send_server_feedback,
    send_buy_alert,
)
from gsheet_logger import log_config_update
from gpretty_logger import write_pretty_to_sheet_from_sheets
from llm_predict4 import forecast_config_update
from redis_util import get_redis, save_gpt_buy_signal_to_redis, get_last_timestamp
from llm_news_sentiment import update_news_sentiment
from pattern_narrator import run_pattern_narration_realtime

load_dotenv()

CONFIG_PATH = "config30a.json"
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
TAB_NAME = os.getenv("TAB_NAME_30A", "stocks_30_gpt")

MAX_WORKERS = int(os.getenv("MAX_WORKERS", 6))
_config_file_lock = threading.Lock()
INDIA_TZ = ZoneInfo("Asia/Kolkata")

# --------------------------------------------------------------------
# Timeframe settings
# --------------------------------------------------------------------
TF_INTERVALS = {
    "5min": timedelta(minutes=5),
    "15min": timedelta(minutes=15),
    "30min": timedelta(minutes=30),
    "45min": timedelta(minutes=45),
    "1hour": timedelta(hours=1),
    "4hour": timedelta(hours=4),
}


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def load_config(path=CONFIG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config Error] {e}")
        return {"stocks": []}


def get_next_expected_close(last_ts, step):
    return None if not last_ts else last_ts + step


def redis_has_new_candle(r, stock_code, tf_label, last_forecast_dt):
    """Check Redis to see if a newer closed candle exists than last GPT run."""
    last_ts = get_last_timestamp(r, stock_code, tf_label)
    if not last_ts:
        return False, None

    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=INDIA_TZ)

    if not last_forecast_dt:
        return True, last_ts  # first run

    return (last_ts > last_forecast_dt), last_ts


def get_last_forecast_time(r, stock_code):
    key = f"FORECASTGPT:{stock_code}"
    val = r.hget(key, "last_updated")
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(val.decode())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=INDIA_TZ)
        return dt
    except Exception:
        return None


def gpt_tf_cooldown_active(r, stock_code, tf_label):
    """Avoid duplicate BUY alerts within timeframe cooldown."""
    key = f"BUY_SIGNAL_GPT:{stock_code}"
    val = r.hget(key, f"{tf_label}_last_alert_time")
    if not val:
        return False

    try:
        last_dt = datetime.fromisoformat(val.decode())
    except Exception:
        return False

    elapsed = (datetime.now(INDIA_TZ) - last_dt).total_seconds() / 60.0
    limits = {
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "45min": 45,
        "1hour": 60,
        "4hour": 240,
    }
    limit = limits.get(tf_label, 5)
    if elapsed < limit:
        print(
            f"[⏸️ COOLDOWN ACTIVE] {stock_code} {tf_label} ({limit - elapsed:.1f}m left)"
        )
        return True
    return False


# --------------------------------------------------------------------
# Forecast Worker
# --------------------------------------------------------------------
def _process_stock_forecast(stock_cfg, tf_label):
    stock_code = stock_cfg["stock_code"]
    buy_signals = []

    try:
        print(
            f"[🧩 Forecast Start] {stock_code} {tf_label} @ {datetime.now(INDIA_TZ):%H:%M:%S}"
        )

        # ✅ NEW: Run pattern narrator before GPT
        try:
            pattern_result = run_pattern_narration_realtime(stock_code, save_json=True)
            print(f"[🪶 Pattern Narration Done] {stock_code} {tf_label}")
        except Exception as e:
            print(f"[⚠️ Pattern Narration Error] {stock_code}: {e}")

        updated_cfg = forecast_config_update(
            stock_cfg, verbose=False, selected_tf=tf_label
        )
        if not updated_cfg:
            return stock_code, None, []

        # ✅ Immediately mark GPT processed time in Redis
        r = get_redis()
        now_str = datetime.now(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S")
        r.hset(f"FORECASTGPT:{stock_code}", mapping={"last_updated": now_str})

        TF_INDEX_MAP = {
            1: "5min",
            2: "15min",
            3: "30min",
            4: "45min",
            5: "1hour",
            6: "4hour",
            7: "1day",
        }

        for idx, tf in TF_INDEX_MAP.items():
            if tf != tf_label:
                continue
            sig_val = updated_cfg.get(f"signal{idx}")
            if sig_val != "BUY":
                continue

            if gpt_tf_cooldown_active(r, stock_code, tf_label):
                continue

            payload = {
                "stock_code": stock_code,
                "timeframe": tf_label,
                "tf_index": idx,
                "signal": "BUY",
                "entry": updated_cfg.get(f"entry{idx}"),
                "target": updated_cfg.get(f"target{idx}"),
                "stoploss": updated_cfg.get(f"stoploss{idx}"),
                "support": updated_cfg.get(f"support{idx}"),
                "resistance": updated_cfg.get(f"resistance{idx}"),
                "entry_target_pct": updated_cfg.get(f"entry_target_pct{idx}"),
                "last_updated": updated_cfg.get("last_updated"),
            }

            if save_gpt_buy_signal_to_redis(r, stock_code, tf_label, payload):
                r.hset(
                    f"BUY_SIGNAL_GPT:{stock_code}",
                    mapping={
                        f"{tf_label}_last_alert_time": datetime.now(INDIA_TZ).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    },
                )
                buy_signals.append((tf_label, payload))
                print(f"[🚀 GPT BUY ACCEPTED] {stock_code} {tf_label}")

        return stock_code, updated_cfg, buy_signals
    except Exception as e:
        print(f"[❌ Forecast Worker Error] {stock_code}: {e}")
        return stock_code, None, []


# --------------------------------------------------------------------
# Forecast Manager (Smart Collector-aligned loop)
# --------------------------------------------------------------------
def forecast_manager():
    print("🧠 Smart Forecast Manager running...")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    r = get_redis()
    processed_candles = {}  # Track last processed candle timestamps

    while True:
        try:
            config = load_config()
            stocks = config.get("stocks", [])
            if not stocks:
                print("[ℹ️] No stocks found; sleeping 15s.")
                time.sleep(15)
                continue

            now = datetime.now(INDIA_TZ)
            print(f"\n🕐 Forecast Cycle @ {now:%Y-%m-%d %H:%M:%S}")
            futures = {}

            for stock_cfg in stocks:
                stock_code = stock_cfg["stock_code"]
                last_forecast_dt = get_last_forecast_time(r, stock_code)

                for tf_label, step in TF_INTERVALS.items():
                    try:
                        last_ts = get_last_timestamp(r, stock_code, tf_label)
                        next_close = get_next_expected_close(last_ts, step)
                        if not next_close:
                            continue

                        has_new, new_ts = redis_has_new_candle(
                            r, stock_code, tf_label, last_forecast_dt
                        )

                        # 🕒 Debug log to confirm candle presence
                        if new_ts:
                            print(
                                f"[🕒 Candle Confirmed in Redis] {stock_code} {tf_label} → {new_ts:%H:%M}"
                            )

                        # ✅ Skip if already processed same candle timestamp
                        key = f"{stock_code}:{tf_label}"
                        if processed_candles.get(key) == str(new_ts):
                            print(
                                f"[⏩ SKIP] {stock_code} {tf_label}: already processed {new_ts:%H:%M}"
                            )
                            continue

                        if not has_new:
                            # Wait up to 2 min for collector to push candle
                            if now >= next_close and now < next_close + timedelta(
                                minutes=2
                            ):
                                print(
                                    f"[⏳ WAIT] {stock_code} {tf_label}: waiting for candle (expected {next_close:%H:%M})"
                                )
                                continue
                            else:
                                continue

                        # ✅ Confirmed new candle present → run GPT once
                        print(
                            f"[🧩 NEW CANDLE READY] {stock_code} {tf_label}: {new_ts:%H:%M} → running GPT."
                        )
                        processed_candles[key] = str(new_ts)

                        fut = executor.submit(
                            _process_stock_forecast, stock_cfg, tf_label
                        )
                        futures[fut] = (stock_code, tf_label)
                        time.sleep(random.uniform(0.3, 0.6))

                    except Exception as e:
                        print(f"[⚠️ TF Loop Error] {stock_code} {tf_label}: {e}")

            # Handle GPT results
            batched_updates, all_buy_signals = [], []

            for fut in as_completed(futures):
                stock_code, tf_label = futures[fut]
                stock_code, updated_cfg, buy_signals = fut.result()
                if updated_cfg:
                    batched_updates.append((stock_code, updated_cfg))
                if buy_signals:
                    all_buy_signals.extend((stock_code, tf, p) for tf, p in buy_signals)

            # Google Sheets sync
            if batched_updates:
                try:
                    for stock_code, _ in batched_updates:
                        redis_data = r.hgetall(f"FORECASTGPT:{stock_code}")
                        if not redis_data:
                            continue
                        live_cfg = {
                            k.decode(): v.decode() for k, v in redis_data.items()
                        }
                        live_cfg["stock_code"] = stock_code
                        live_cfg["last_updated"] = live_cfg.get(
                            "last_updated",
                            datetime.now(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        log_config_update(
                            live_cfg, GOOGLE_SHEET_NAME, tab_name=TAB_NAME
                        )
                    write_pretty_to_sheet_from_sheets(
                        spreadsheet_name=GOOGLE_SHEET_NAME,
                        gpt_tab=os.getenv("TAB_NAME_30A", "stocks_30_gpt"),
                        algo_tab=os.getenv("TAB_NAME_30B", "stocks_30_algo"),
                        pretty_tab=os.getenv("TAB_NAME_30_ALL", "stocks_30_all"),
                        service_account_json="cred.json",
                    )
                    print(f"[✅ Synced {len(batched_updates)} stocks to Google Sheets]")
                except Exception as e:
                    print(f"[⚠️ Sheet Sync Error] {e}")

            # Telegram Alerts
            if all_buy_signals:
                for stock_code, tf_label, payload in all_buy_signals:
                    try:
                        send_buy_alert(
                            code=stock_code,
                            tf_index=payload.get("tf_index"),
                            ts=payload.get("last_updated"),
                            reason="GPT BUY signal",
                            entry=payload.get("entry"),
                            target=payload.get("target"),
                            stop=payload.get("stoploss"),
                            support=payload.get("support"),
                            resistance=payload.get("resistance"),
                            live_price=None,
                            stock_cfg=payload,
                        )
                        print(f"[📢 Alert Sent] {stock_code} {tf_label}")
                    except Exception as e:
                        send_error_alert(
                            f"[Telegram Error] {stock_code} {tf_label}: {e}"
                        )

            time.sleep(60)

        except Exception as e:
            print(f"[Forecast Manager Error] {e}")
            send_error_alert(f"[Forecast Manager Error] {e}")
            time.sleep(10)


# --------------------------------------------------------------------
# Target Monitor
# --------------------------------------------------------------------
def target_monitor():
    print("🎯 GPT Target Monitor started.")
    r = get_redis()
    tz = INDIA_TZ
    MONITORED_TFS = ["30min", "45min", "1hour", "4hour"]

    while True:
        try:
            keys = r.keys("BUY_SIGNALS:*")
            for key in keys:
                stock_code = key.split(":")[1]
                fields = r.hkeys(key)
                for f in fields:
                    if f.endswith("_target_status") and r.hget(key, f) == b"pending":
                        tf = f.replace("_target_status", "")
                        if tf not in MONITORED_TFS:
                            continue
                        payload_raw = r.hget(key, tf)
                        if not payload_raw:
                            continue
                        data = json.loads(payload_raw)
                        target = float(data.get("target") or 0)
                        entry = float(data.get("entry") or 0)
                        stop = float(data.get("stoploss") or 0)
                        live_key = f"MARKETDATA:{stock_code}:{tf}"
                        candles = [json.loads(v) for v in r.hgetall(live_key).values()]
                        candles.sort(key=lambda c: c.get("Timestamp", ""))
                        latest = candles[-1]
                        price_check = float(
                            latest.get("High") or latest.get("Close") or 0
                        )
                        if price_check >= target:
                            r.hset(
                                key,
                                mapping={
                                    f"{tf}_target_status": "sent",
                                    f"{tf}_target_hit_time": datetime.now(tz).strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                },
                            )
                            msg = (
                                f"🎯 *TARGET ACHIEVED (GPT)*\n"
                                f"Symbol: `{stock_code}`\n"
                                f"Timeframe: `{tf}`\n"
                                f"Entry: `{entry}` | Target: `{target}` | Stoploss: `{stop}`\n"
                                f"Hit Time: `{datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}`"
                            )
                            asyncio.run(send_telegram_message(msg))
                            print(f"[🎯 Target Hit Alert Sent] {stock_code} {tf}")

            time.sleep(60)
        except Exception as e:
            print(f"[⚠️ Target Monitor Error] {e}")
            time.sleep(10)


def run_news_fetcher_loop():
    while True:
        try:
            print("[📰] Running news sentiment updater...")
            update_news_sentiment()
            print("[✅] News sentiment refresh complete.\n")
        except Exception as e:
            print(f"[⚠️] News updater error: {e}")
        time.sleep(5 * 60)


# --------------------------------------------------------------------
# Server Runner
# --------------------------------------------------------------------
def run():
    processes = {}
    print(f"🚀 Starting GPT Forecast Server — {TAB_NAME}")
    send_server_feedback()

    while True:
        try:
            if "news_fetcher" not in processes:
                t = threading.Thread(target=run_news_fetcher_loop, daemon=True)
                t.start()
                processes["news_fetcher"] = t
                print("📰 News Fetcher thread started.")

            if "forecaster" not in processes:
                p1 = Process(target=forecast_manager)
                p1.start()
                processes["forecaster"] = p1
                print("🧠 Forecast Manager started.")

            if "target_monitor" not in processes:
                p2 = Process(target=target_monitor)
                p2.start()
                processes["target_monitor"] = p2
                print("🎯 Target Monitor started.")

            for name, proc in list(processes.items()):
                if not proc.is_alive():
                    print(f"[⚠️ Restarting {name}]")
                    new_proc = Process(
                        target=(
                            forecast_manager if name == "forecaster" else target_monitor
                        )
                    )
                    new_proc.start()
                    processes[name] = new_proc

            time.sleep(10)
        except KeyboardInterrupt:
            print("⛔️ Server stopped manually.")
            break
        except Exception as e:
            send_error_alert(f"[Server Error] {e}")
            time.sleep(10)


if __name__ == "__main__":
    run()
