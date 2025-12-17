import redis, json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
key = "MARKETDATA:FORCEMOT:30min"

data = r.hgetall(key)
if not data:
    print("No data found for PARADEEP.")
else:
    last_ts = max(data.keys())  # most recent timestamp
    print(f"Latest timestamp: {last_ts}")
    candle = json.loads(data[last_ts])
    print(json.dumps(candle, indent=2))


#!/usr/bin/env python3
# init_forecast_keys.py — Create missing FORECAST:* Redis keys from config.json

# import json
# from redis_util import get_redis

# CONFIG_PATH = "config30b.json"
# r = get_redis()
# cfg = json.load(open(CONFIG_PATH, "r"))

# existing = {k.split(":")[1] for k in r.keys("FORECAST:*")}
# new_count = 0

# for s in cfg.get("stocks", []):
#     code, token = s["stock_code"], s["instrument_token"]
#     if code not in existing:
#         r.hset(
#             f"FORECAST:{code}",
#             mapping={
#                 "stock_code": code,
#                 "instrument_token": token,
#                 "forecast": "basic_algo",
#                 "last_updated": "",
#             },
#         )
#         print(f"[🆕 Added missing forecast key for {code}]")
#         new_count += 1

# print(f"✅ Initialization complete — {new_count} new forecast keys created.")


######################################################################################################
