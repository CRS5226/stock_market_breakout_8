# #!/usr/bin/env python3
# import redis
# import json
# import pandas as pd
# from datetime import datetime

# # ==========================================================
# # CONFIGURATION
# # ==========================================================
# REDIS_HOST = "localhost"  # Change if Redis is remote
# REDIS_PORT = 6379
# REDIS_DB = 0

# STOCK_CODES = [
#     "RRKABEL",
#     "PARADEEP",
#     "GMDCLTD",
#     "GOKEX",
#     "SHAKTIPUMP",
#     "FORCEMOT",
# ]

# OUTPUT_CSV = "buy_signals_export.csv"

# # ==========================================================
# # REDIS CONNECTION
# # ==========================================================


# def get_redis_connection():
#     """Initialize and return Redis connection."""
#     return redis.Redis(
#         host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
#     )


# # ==========================================================
# # FETCH LOGIC
# # ==========================================================


# def fetch_buy_signals(r, stock_code):
#     """
#     Fetch all buy signal data for a given stock from Redis.
#     Returns a list of dicts (one per timeframe).
#     """
#     key = f"BUY_SIGNALS:{stock_code}"
#     if not r.exists(key):
#         print(f"[ℹ️] No BUY_SIGNALS for {stock_code}")
#         return []

#     data = r.hgetall(key)
#     results = []

#     # Iterate through all possible timeframes
#     for tf in ["5min", "30min", "45min", "1hour", "4hour"]:
#         payload_json = data.get(tf)
#         if not payload_json:
#             continue

#         try:
#             payload = json.loads(payload_json)
#         except Exception as e:
#             print(f"[⚠️] Failed to parse JSON for {stock_code} {tf}: {e}")
#             continue

#         # Enrich payload with extra Redis metadata
#         payload.update(
#             {
#                 "entry_status": data.get(f"{tf}_entry_status", ""),
#                 "target_status": data.get(f"{tf}_target_status", ""),
#                 "entry_time": data.get(f"{tf}_entry_time", ""),
#                 "target_hit_time": data.get(f"{tf}_target_hit_time", ""),
#             }
#         )

#         results.append(payload)

#     return results


# # ==========================================================
# # EXPORT LOGIC
# # ==========================================================


# def export_all_buy_signals():
#     """Fetch all BUY_SIGNALS:* from Redis and save as a CSV."""
#     r = get_redis_connection()
#     all_records = []

#     print("📡 Fetching BUY_SIGNALS from Redis...")

#     # Loop through your tracked stock codes
#     for stock_code in STOCK_CODES:
#         signals = fetch_buy_signals(r, stock_code)
#         all_records.extend(signals)

#     if not all_records:
#         print("⚠️ No BUY signal data found in Redis.")
#         return

#     df = pd.DataFrame(all_records)

#     # --- Ensure last_updated is datetime for sorting ---
#     if "last_updated" in df.columns:
#         df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

#     # --- Sort by last_updated descending ---
#     df = df.sort_values(by="last_updated", ascending=False, na_position="last")

#     # --- Reorder columns ---
#     preferred_cols = [
#         "last_updated",  # moved to first
#         "stock_code",
#         "timeframe",
#         "signal",
#         "entry",
#         "target",
#         "stoploss",
#         "support",
#         "resistance",
#         "entry_target_pct",
#         "entry_status",
#         "target_status",
#         "entry_time",
#         "target_hit_time",
#     ]
#     df = df[[c for c in preferred_cols if c in df.columns]]

#     # --- Export to CSV ---
#     df.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
#     print(f"✅ Saved {len(df)} rows to {OUTPUT_CSV} (sorted by last_updated ↓)")


# # ==========================================================
# # MAIN
# # ==========================================================

# if __name__ == "__main__":
#     export_all_buy_signals()


# purge_buy_signals.py
# ----------------------------------------------------
# Deletes ALL BUY signals from Redis:
# Keys pattern: BUY_SIGNALS:{STOCK_CODE}
# ----------------------------------------------------

import sys
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379


def purge_all_buy_signals():
    """
    Deletes all BUY signals saved by the forecaster.
    Keys affected: BUY_SIGNALS:*
    """
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        print("🔍 Searching for BUY_SIGNALS:* keys...")
        keys = r.keys("BUY_SIGNALS:*")

        if not keys:
            print("ℹ️ No BUY_SIGNALS keys found in Redis.")
            return

        print(f"⚠️ Found {len(keys)} BUY signal keys. Deleting...")

        for key in keys:
            r.delete(key)
            print(f"✔ Deleted: {key}")

        print("\n🎉 All BUY signals removed successfully.")

    except Exception as e:
        print(f"❌ Error while deleting BUY signals: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("----------------------------------------------------")
    print("🧹  BUY SIGNAL PURGE TOOL")
    print("----------------------------------------------------\n")
    purge_all_buy_signals()
