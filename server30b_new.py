# server30b.py (updated)
import os
import time
import json
import random
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Manager
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import threading
import asyncio
from datetime import datetime, timedelta


load_dotenv()

# === Local imports ===
from telegram_alert30b import (
    send_telegram_message,
    send_error_alert,
    send_server_feedback,
    trigger_buy_alert_from_config,
    send_buy_alert,  # your function to send formatted buy alerts
)
from gsheet_logger import log_config_update
from gpretty_logger import write_pretty_to_sheet_from_sheets

# Forecast imports
# from llm_predict3 import forecast_config_update, route_model
# from basic_algo4 import basic_forecast_update

from basic_algo5a import basic_forecast_update

# Redis helpers (ensure redis_util has save_forecast_to_redis and save_buy_signal_to_redis)
from redis_util import get_redis, save_forecast_to_redis, save_buy_signal_to_redis

# === GLOBAL CONFIG ===
CONFIG_PATH = "config.json"
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
TAB_NAME = os.getenv("TAB_NAME_30B")
USE_LLM = False

# MONITOR_TFS = ["5min", "30min", "1hour"]

# Thread pool size (tune per machine). For 2-core VPS, keep small.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 6))

# Lock for atomic config.json writes
_config_file_lock = threading.Lock()


def load_config(path=CONFIG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config Error] Failed to read config: {e}")
        return {"stocks": []}


def atomic_write_config(cfg_obj, path=CONFIG_PATH):
    """
    Atomically write config JSON to disk (write temp -> replace).
    Protect with a thread lock so only one thread writes at a time.
    """
    tmp = f"{path}.tmp"
    try:
        with _config_file_lock:
            with open(tmp, "w") as f:
                json.dump(cfg_obj, f, indent=2)
            os.replace(tmp, path)
    except Exception as e:
        print(f"[Config Write Error] {e}")


def fetch_latest_config_for_stock(stock_code):
    cfg = load_config()
    for s in cfg.get("stocks", []):
        if s.get("stock_code") == stock_code:
            return s
    return None


def load_stocks_from_redis():
    """
    Fetch stock configs directly from Redis FORECAST:* hashes.
    Returns a list of stock dicts (same schema as config.json['stocks']).
    """
    try:
        r = get_redis()
        keys = r.keys("FORECAST:*")
        if not keys:
            print("[ℹ️] No forecast keys found in Redis — falling back to config.json")
            return []

        stocks = []
        for key in keys:
            stock_code = key.split(":")[1]
            try:
                # Try full JSON first (if you saved via FORECAST_JSON:{stock})
                full_json = r.get(f"FORECAST_JSON:{stock_code}")
                if full_json:
                    stocks.append(json.loads(full_json))
                    continue

                # Otherwise, rebuild dict from hash
                data = r.hgetall(key)
                if not data:
                    continue
                # Decode JSON fields if needed
                decoded = {}
                for k, v in data.items():
                    try:
                        decoded[k] = json.loads(v)
                    except Exception:
                        decoded[k] = v
                decoded["stock_code"] = stock_code
                stocks.append(decoded)
            except Exception as e:
                print(f"[⚠️] Redis load failed for {stock_code}: {e}")
        print(f"[✅] Loaded {len(stocks)} stock configs from Redis.")
        return stocks

    except Exception as e:
        print(f"[⚠️] Redis not reachable ({e}) — falling back to config.json")
        return []


# -------------------------
# Threaded forecast worker
# -------------------------


def _process_stock_forecast(stock_cfg, use_llm=False, verbose=True):
    """
    Runs forecast for a single stock.
    BUY signals are accepted ONLY if algo exists.
    """

    stock_code = stock_cfg.get("stock_code")
    buy_signals = []

    try:
        updated_cfg = basic_forecast_update(stock_cfg, verbose=verbose)

        # Engine identity (informational only)
        updated_cfg["forecast"] = "basic_algo5b"
        updated_cfg["algo_engine"] = "basic_algo5b"

        # Save forecast
        r = get_redis()
        save_forecast_to_redis(r, stock_code, updated_cfg)

        TF_INDEX_MAP = {
            0: "1min",
            1: "5min",
            2: "15min",
            3: "30min",
            4: "45min",
            5: "1hour",
            6: "4hour",
            7: "1day",
            8: "1month",
        }

        for idx, tf_label in TF_INDEX_MAP.items():
            sig_key = "signal" if idx == 0 else f"signal{idx}"
            if updated_cfg.get(sig_key) != "BUY":
                continue

            tf_key = tf_label.lower().strip()
            algo_list = updated_cfg.get("algo_signals", {}).get(tf_key, [])

            if not algo_list:
                print(f"[DROP BUY] {stock_code} {tf_label} BUY present but no algo")
                continue

            algo_name = algo_list[0]

            payload = {
                "ALGO": algo_name,
                "stock_code": stock_code,
                "timeframe": tf_label,
                "tf_index": idx,
                "signal": "BUY",
                "entry": updated_cfg.get(f"entry{idx}" if idx != 0 else "entry"),
                "target": updated_cfg.get(f"target{idx}" if idx != 0 else "target"),
                "stoploss": updated_cfg.get(
                    f"stoploss{idx}" if idx != 0 else "stoploss"
                ),
                "support": updated_cfg.get(f"support{idx}" if idx != 0 else "support"),
                "resistance": updated_cfg.get(
                    f"resistance{idx}" if idx != 0 else "resistance"
                ),
                "entry_target_pct": updated_cfg.get(
                    f"entry_target_pct{idx}" if idx != 0 else "entry_target_pct"
                ),
                "last_updated": updated_cfg.get("last_updated"),
            }

            print(
                f"[SERVER BUY PIPE] {stock_code} {tf_label} "
                f"algo={algo_name} entry={payload['entry']} target={payload['target']}"
            )

            should_alert = save_buy_signal_to_redis(r, stock_code, tf_label, payload)

            print(f"[DEBUG BUY] {stock_code} {tf_label} alert={should_alert}")

            if should_alert:
                buy_signals.append((tf_label, payload))

    except Exception as e:
        print(f"[❌ Forecast worker error] {stock_code}: {type(e).__name__}: {e}")

    return stock_code, updated_cfg, buy_signals


# ============================================================
# Forecast manager (uses ThreadPoolExecutor)
# ============================================================
def forecast_manager():
    print("🧠 Forecast Manager (threadpool) started.")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while True:
        try:
            # Try Redis first
            #stocks = load_stocks_from_redis()
            stocks = None
            if not stocks:
                # fallback to local config.json
                config_data = load_config(CONFIG_PATH)
                stocks = config_data.get("stocks", [])

            if not stocks:
                print("[ℹ️] No stocks in config; sleeping.")
                time.sleep(5)
                continue

            # Submit all stocks to threadpool
            futures = {}
            for i, stock_cfg in enumerate(stocks):
                # small jitter to spread Redis load slightly
                time.sleep(random.uniform(0, 0.01))
                fut = executor.submit(
                    _process_stock_forecast, stock_cfg, USE_LLM, False
                )
                futures[fut] = i

            # --- Collect all results in one pass ---
            batched_updates = []  # for Google Sheets
            all_buy_signals = []  # for Telegram alerts
            any_update = False

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    stock_code, updated_cfg, buy_signals = fut.result()
                except Exception as e:
                    print(f"[Forecast Future Error] {e}")
                    continue

                if updated_cfg and isinstance(updated_cfg, dict):
                    old_stock = stocks[i]
                    if updated_cfg != old_stock:
                        stocks[i] = updated_cfg
                        any_update = True
                        print(f"[✅ Forecast Updated - {stock_code}]")

                    batched_updates.append((stock_code, updated_cfg))

                if buy_signals:
                    for tf_label, payload in buy_signals:
                        all_buy_signals.append((stock_code, tf_label, payload))

            # --- Google Sheets batch sync (once per cycle) ---
            if batched_updates:
                try:
                    for stock_code, _ in batched_updates:
                        r = get_redis()
                        redis_data = r.hgetall(f"FORECAST:{stock_code}")
                        if not redis_data:
                            print(f"[⚠️ Redis data missing for {stock_code}]")
                            continue

                        live_cfg = {}
                        for k, v in redis_data.items():
                            k = k.decode() if isinstance(k, bytes) else k
                            v = v.decode() if isinstance(v, bytes) else v
                            try:
                                if v and (v.startswith("{") or v.startswith("[")):
                                    v = json.loads(v)
                            except Exception:
                                pass
                            live_cfg[k] = v

                        live_cfg["stock_code"] = stock_code
                        live_cfg["last_updated"] = live_cfg.get(
                            "last_updated"
                        ) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                #         try:
                #             log_config_update(
                #                 live_cfg, GOOGLE_SHEET_NAME, tab_name=TAB_NAME
                #             )
                #         except Exception as e:
                #             print(
                #                 f"[⚠️ Google Sheets logging failed for {stock_code}]: {e}"
                #             )

                #     # Write pretty sheet once at the end
                #     try:
                #         write_pretty_to_sheet_from_sheets(
                #             spreadsheet_name=GOOGLE_SHEET_NAME,
                #             gpt_tab=os.getenv("TAB_NAME_30A", "stocks_30_gpt"),
                #             algo_tab=os.getenv("TAB_NAME_30B", "stocks_30_algo"),
                #             pretty_tab=os.getenv("TAB_NAME_30_ALL", "stocks_30_all"),
                #             service_account_json="cred.json",
                #         )
                #         print(
                #             f"[✅ Batched Redis→Sheets sync completed for {len(batched_updates)} stocks]"
                #         )
                #     except Exception as e:
                #         print(f"[⚠️ Google Sheets pretty writer failed]: {e}")

                except Exception as e_gs:
                    print(f"[⚠️ Google Sheets batch logging failed]: {e_gs}")

            # --- Telegram BUY alerts (with cooldown + dedup) ---
            if all_buy_signals:
                r = get_redis()
                for stock_code, tf_label, payload in all_buy_signals:
                    try:
                        ts = payload.get("last_updated") or datetime.now(
                            ZoneInfo("Asia/Kolkata")
                        ).strftime("%Y-%m-%d %H:%M:%S")

                        key = f"BUY_SIGNALS:{stock_code}"

                        # Cooldown: skip same alert within 3 minutes
                        last_alert_time = r.hget(key, f"{tf_label}_last_alert_time")
                        if last_alert_time:
                            try:
                                last_dt = datetime.fromisoformat(last_alert_time)
                                if (datetime.now() - last_dt).total_seconds() < 180:
                                    print(
                                        f"[⏩ COOLDOWN] {stock_code} {tf_label}: skip duplicate within 3min"
                                    )
                                    continue
                            except Exception:
                                pass

                        # Mark as sent + update last alert timestamp
                        r.hset(
                            key,
                            mapping={
                                f"{tf_label}_entry_status": "sent",
                                f"{tf_label}_last_alert_time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            },
                        )

                        # Send formatted Telegram alert
                        # send_buy_alert(
                        #     code=stock_code,
                        #     tf_index=payload.get("tf_index"),
                        #     ts=ts,
                        #     reason="Realtime BUY signal",
                        #     entry=payload.get("entry"),
                        #     target=payload.get("target"),
                        #     stop=payload.get("stoploss"),
                        #     support=payload.get("support"),
                        #     resistance=payload.get("resistance"),
                        #     live_price=None,
                        #     stock_cfg=payload,
                        # )

                        send_buy_alert(
                            code=stock_code,
                            tf_index=payload.get("tf_index"),
                            ts=ts,
                            reason=f"Realtime BUY signal — ALGO: {payload.get('ALGO', 'NA')}",
                            entry=payload.get("entry"),
                            target=payload.get("target"),
                            stop=payload.get("stoploss"),
                            support=payload.get("support"),
                            resistance=payload.get("resistance"),
                            live_price=None,
                            stock_cfg=payload,
                        )

                        print(f"[🚀 BUY ALERT SENT] {stock_code} {tf_label}")

                    except Exception as e_alert:
                        print(
                            f"[⚠️] Failed to send buy alert for {stock_code} {tf_label}: {e_alert}"
                        )
                        send_error_alert(
                            f"[Alert Error] {stock_code} {tf_label}: {e_alert}"
                        )

            # Sleep before next forecast cycle (aligned with collector)
            time.sleep(10)

        except Exception as e:
            print(f"[Forecast Manager Error] {type(e).__name__}: {e}")
            send_error_alert(f"[Forecast Manager Error] {type(e).__name__}: {e}")
            time.sleep(5)


def target_monitor():
    print("🎯 Target Monitor started.")
    r = get_redis()
    tz = ZoneInfo("Asia/Kolkata")

    MONITORED_TFS = ["30min", "45min", "1hour", "4hour"]

    while True:
        try:
            keys = r.keys("BUY_SIGNALS:*")
            for key in keys:
                stock_code = key.split(":")[1]
                fields = r.hkeys(key)

                for f in fields:
                    if f.endswith("_target_status") and r.hget(key, f) == "pending":
                        tf = f.replace("_target_status", "")
                        if tf not in MONITORED_TFS:
                            continue  # only monitor selected timeframes

                        payload_raw = r.hget(key, tf)
                        if not payload_raw:
                            continue
                        data = json.loads(payload_raw)
                        target = float(data.get("target") or 0)
                        entry = float(data.get("entry") or 0)
                        stop = float(data.get("stoploss") or 0)

                        # ---- Parse entry time ----
                        ts_entry = r.hget(key, f"{tf}_entry_time")
                        try:
                            entry_time = datetime.strptime(
                                ts_entry, "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=tz)
                        except Exception:
                            entry_time = datetime.now(tz) - timedelta(days=1)

                        # ---- Load last candle ----
                        live_price_key = f"MARKETDATA:{stock_code}:{tf}"
                        last_candle = r.hgetall(live_price_key)
                        if not last_candle:
                            continue

                        try:
                            candles = [json.loads(v) for v in last_candle.values()]
                            candles = sorted(
                                candles, key=lambda c: c.get("Timestamp", "")
                            )
                            latest = candles[-1]
                        except Exception as e:
                            print(f"[⚠️ Parse error in {stock_code} {tf}: {e}]")
                            continue

                        # ---- Parse candle timestamp safely ----
                        try:
                            latest_ts = datetime.fromisoformat(str(latest["Timestamp"]))
                            if latest_ts.tzinfo is None:
                                latest_ts = latest_ts.replace(tzinfo=tz)
                        except Exception:
                            continue

                        # ---- Skip same candle ----
                        if latest_ts <= entry_time:
                            continue

                        # ---- Check if target hit ----
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
                                f"🎯 *TARGET ACHIEVED*\n"
                                f"Symbol: `{stock_code}`\n"
                                f"Timeframe: `{tf}`\n"
                                f"Entry: `{entry}` | Target: `{target}` | Stoploss: `{stop}`\n"
                                f"Entry Time: `{ts_entry}`\n"
                                f"Hit Time: `{datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}`\n"
                            )
                            asyncio.run(send_telegram_message(msg))
                            print(f"[🎯 Target Alert Sent] {stock_code} {tf}")

            time.sleep(60)  # once per minute is enough

        except Exception as e:
            print(f"[⚠️ Target Monitor Error] {e}")
            time.sleep(10)


# ============================================================
# SERVER RUNNER
# ============================================================
def run():
    processes = {}
    manager = Manager()
    stats = manager.dict()

    print(f"🚀 Real-Time Stock Server starting — {TAB_NAME}")
    send_server_feedback()

    while True:
        try:
            config = load_config(CONFIG_PATH)
            stock_list = config.get("stocks", [])

            # Start forecast manager process (single process will run threadpool inside)
            if "forecaster" not in processes:
                forecast_proc = Process(target=forecast_manager)
                forecast_proc.start()
                processes["forecaster"] = forecast_proc
                print("🧠 Forecast Manager process started.")

            if "target_monitor" not in processes:
                tmon_proc = Process(target=target_monitor)
                tmon_proc.start()
                processes["target_monitor"] = tmon_proc
                print("🎯 Target Monitor process started.")

            # Restart if needed
            for name, proc in list(processes.items()):
                if not proc.is_alive():
                    print(f"[⚠️] Process {name} stopped. Restarting...")

                    if name == "forecaster":
                        forecast_proc = Process(target=forecast_manager)
                        forecast_proc.start()
                        processes[name] = forecast_proc
                        print("♻️ Restarted Forecast Manager")

            time.sleep(10)

        except KeyboardInterrupt:
            print("⛔️ Server stopped manually.")
            break
        except Exception as e:
            send_error_alert(f"[Server Error] {type(e).__name__}: {e}")
            print(f"[❌ Server Error] {e}")
            time.sleep(10)


if __name__ == "__main__":
    run()
