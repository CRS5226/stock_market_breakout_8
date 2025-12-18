# redis_util.py
import os
import json
import redis
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

REDIS_PREFIX = "MARKETDATA"
MAX_CANDLES = 100


def get_redis():
    """Create and return a Redis connection."""
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
    )


def redis_key(stock_code: str, timeframe: str) -> str:
    """Builds a Redis key for given stock & timeframe."""
    return f"{REDIS_PREFIX}:{stock_code}:{timeframe}"


def save_to_redis(r, stock_code: str, tf: str, df: pd.DataFrame):
    """Save dataframe rows into Redis (with cap MAX_CANDLES)."""
    if df.empty:
        return
    key = redis_key(stock_code, tf)
    records = df.to_dict(orient="records")

    pipe = r.pipeline()
    for row in records:
        # Expect 'Timestamp' is a pandas.Timestamp or ISO string; store as ISO
        ts = (
            row["Timestamp"].isoformat()
            if hasattr(row["Timestamp"], "isoformat")
            else str(row["Timestamp"])
        )
        pipe.hset(key, ts, json.dumps(row, default=str))
    pipe.execute()

    # Trim old data if beyond cap
    all_keys = r.hkeys(key)
    if len(all_keys) > MAX_CANDLES:
        to_delete = sorted(all_keys)[: len(all_keys) - MAX_CANDLES]
        if to_delete:
            r.hdel(key, *to_delete)
    print(f"✅ Redis updated {len(records)} → {key} (kept last {MAX_CANDLES})")


def get_last_timestamp(r, stock_code: str, tf: str):
    """Get last stored timestamp for a stock/timeframe (returns pandas Timestamp)."""
    key = redis_key(stock_code, tf)
    if not r.exists(key):
        return None
    all_keys = r.hkeys(key)
    if not all_keys:
        return None
    last_ts = max(all_keys)
    return pd.to_datetime(last_ts)


def load_from_redis(r, stock_code: str, tf: str, limit: int = None) -> pd.DataFrame:
    """Load stored candles from Redis as DataFrame (ascending chronological order)."""
    key = redis_key(stock_code, tf)
    if not r.exists(key):
        return pd.DataFrame()

    raw = r.hgetall(key)
    rows = [json.loads(val) for _, val in raw.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure Timestamp is parsed and sorted ascending (oldest -> newest)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    if limit:
        df = df.tail(limit)
    return df


# ---------------------------------------------------------------------
# New helper APIs: these return **lists of dicts ordered newest-first**
# (this matches other parts of the code which call `reversed(rows)` to get
# chronological order). Keeping this contract avoids subtle bugs.
# ---------------------------------------------------------------------
def get_recent_indicators_tf(r, stock_code: str, tf: str, n: int = 200):
    """
    Return recent n rows (indicators/candles) for stock_code/timeframe
    as a list of dicts in newest-first order (most recent first).
    """
    df = load_from_redis(r, stock_code, tf, limit=n)
    if df.empty:
        return []
    # df is ascending (oldest->newest); convert to newest-first list
    df_desc = df.sort_values("Timestamp", ascending=False).head(n)
    return df_desc.to_dict(orient="records")


def get_recent_candles_tf(r, stock_code: str, tf: str, n: int = 200):
    """
    Alias / semantic wrapper: recent raw candles (newest-first).
    Collector stores indicators inside the same TF keys so this will return
    indicator-inclusive rows if they exist.
    """
    return get_recent_indicators_tf(r, stock_code, tf, n=n)


def get_recent_candles(r, stock_code: str, n: int = 200):
    """
    Legacy wrapper that returns recent 1min candles (newest-first).
    """
    return get_recent_candles_tf(r, stock_code, "1min", n=n)


def save_forecast_to_redis(r, stock_code, forecast_dict):
    """Save both flat and full JSON forms for full visibility."""
    key = f"FORECAST:{stock_code}"
    try:
        flat = {}
        for k, v in forecast_dict.items():
            if v is None:
                flat[k] = "null"  # mark None safely
            elif isinstance(v, (dict, list)):
                flat[k] = json.dumps(v)
            elif isinstance(v, (int, float, str)):
                flat[k] = str(v)
            else:
                flat[k] = json.dumps(str(v))
        r.hset(key, mapping=flat)
        r.set(
            f"FORECAST_JSON:{stock_code}",
            json.dumps(forecast_dict, indent=2, default=str),
        )
    except Exception as e:
        print(f"[⚠️ Redis forecast save error] {stock_code}: {e}")


## GPT BUY SIGNAL
def save_gpt_buy_signal_to_redis(r, stock_code: str, tf: str, signal_data: dict):
    """
    Save GPT BUY signal in a separate Redis namespace.
    Uses deduplication via signal hash, same logic as the basic algo version.
    Key pattern: BUY_SIGNALS_GPT:{stock_code}
    """
    key = f"BUY_SIGNALS_GPT:{stock_code}"
    tf_name = (signal_data.get("timeframe", tf) or tf).strip().lower()

    entry_status_field = f"{tf_name}_entry_status"
    target_status_field = f"{tf_name}_target_status"
    hash_field = f"{tf_name}_last_signal_hash"

    # Compute new signal hash (deduplication)
    new_hash = compute_signal_hash(signal_data)
    old_hash = r.hget(key, hash_field)
    if isinstance(old_hash, bytes):
        old_hash = old_hash.decode()

    # Deduplication check
    if old_hash == new_hash and r.hget(key, entry_status_field) == "sent":
        print(f"[⏩ GPT DUP SKIP] {stock_code} {tf_name}: identical GPT signal hash")
        return False

    # New or changed signal → store
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    r.hset(
        key,
        mapping={
            tf_name: json.dumps(signal_data, default=str),
            entry_status_field: "pending",
            target_status_field: "pending",
            f"{tf_name}_entry_time": signal_data.get("last_updated", now),
            hash_field: new_hash,
        },
    )

    print(f"[✅ GPT NEW SIGNAL STORED] {stock_code} {tf_name} (hash={new_hash})")
    return True


## ALGO BUY SIGNAL
import json


# def save_buy_signal_to_redis(r, stock_code: str, tf: str, signal_data: dict):
#     """
#     Dedup-safe signal save logic:
#       - Compares hashes of entry/target/stoploss
#       - Only triggers alert if hash changed or status was reset
#     """
#     key = f"BUY_SIGNALS:{stock_code}"
#     tf_name = (signal_data.get("timeframe", tf) or tf).strip().lower()

#     entry_status_field = f"{tf_name}_entry_status"
#     target_status_field = f"{tf_name}_target_status"
#     hash_field = f"{tf_name}_last_signal_hash"

#     # Compute new signal hash
#     new_hash = compute_signal_hash(signal_data)
#     old_hash = r.hget(key, hash_field)
#     if isinstance(old_hash, bytes):
#         old_hash = old_hash.decode()

#     # If same hash and already sent → skip
#     if old_hash == new_hash and r.hget(key, entry_status_field) == "sent":
#         print(f"[⏩ DUP SKIP] {stock_code} {tf_name}: identical signal hash")
#         return False

#     # Else new or changed signal → update Redis and alert
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     r.hset(
#         key,
#         mapping={
#             tf_name: json.dumps(signal_data),
#             entry_status_field: "pending",
#             target_status_field: "pending",
#             # f"{tf_name}_entry_time": now,
#             f"{tf_name}_entry_time": signal_data.get("last_updated", now),
#             hash_field: new_hash,
#         },
#     )
#     print(f"[✅ NEW SIGNAL STORED] {stock_code} {tf_name} (hash changed)")
#     return True


def save_buy_signal_to_redis(r, stock_code: str, tf: str, signal_data: dict):
    """
    Dedup-safe BUY signal save.
    HARD RULE: BUY without ALGO is illegal.
    """

    import json
    from datetime import datetime

    key = f"BUY_SIGNALS:{stock_code}"
    tf_name = (signal_data.get("timeframe", tf) or tf).strip().lower()

    entry_status_field = f"{tf_name}_entry_status"
    target_status_field = f"{tf_name}_target_status"
    hash_field = f"{tf_name}_last_signal_hash"

    # 🚨 HARD INVARIANT
    algo = signal_data.get("ALGO")
    if not algo:
        raise RuntimeError(f"[FATAL] BUY without ALGO: {stock_code} {tf_name}")

    # -----------------------------------------
    # HASH
    # -----------------------------------------
    new_hash = compute_signal_hash(signal_data)
    old_hash = r.hget(key, hash_field)
    old_status = r.hget(key, entry_status_field)

    if old_hash == new_hash and old_status == "sent":
        print(f"[SKIP] {stock_code} {tf_name} same hash, already sent")
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    r.hset(
        key,
        mapping={
            tf_name: json.dumps(signal_data),
            entry_status_field: old_status or "pending",
            target_status_field: r.hget(key, target_status_field) or "pending",
            f"{tf_name}_entry_time": signal_data.get("last_updated", now),
            hash_field: new_hash,
        },
    )

    print(f"[SAVE BUY] {stock_code} {tf_name} " f"algo={algo} hash={new_hash}")

    return True


def save_gpt_buy_signal_to_redis_can(r, stock_code: str, tf: str, signal_data: dict):
    """
    Save GPT BUY signal for the candle-only (CAN) model in a separate Redis namespace.
    Key pattern: BUY_SIGNALS_CAN:{stock_code}
    Same logic as save_gpt_buy_signal_to_redis() but with isolated keys.
    """
    key = f"BUY_SIGNALS_CAN:{stock_code}"
    tf_name = (signal_data.get("timeframe", tf) or tf).strip().lower()

    entry_status_field = f"{tf_name}_entry_status"
    target_status_field = f"{tf_name}_target_status"
    hash_field = f"{tf_name}_last_signal_hash"

    # Compute deduplication hash
    new_hash = compute_signal_hash(signal_data)
    old_hash = r.hget(key, hash_field)
    if isinstance(old_hash, bytes):
        old_hash = old_hash.decode()

    # Skip if same signal already sent
    if old_hash == new_hash and r.hget(key, entry_status_field) == "sent":
        print(f"[⏩ CAN DUP SKIP] {stock_code} {tf_name}: identical signal hash")
        return False

    # Store signal data
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    r.hset(
        key,
        mapping={
            tf_name: json.dumps(signal_data, default=str),
            entry_status_field: "pending",
            target_status_field: "pending",
            f"{tf_name}_entry_time": signal_data.get("last_updated", now),
            hash_field: new_hash,
        },
    )

    print(f"[✅ CAN GPT SIGNAL STORED] {stock_code} {tf_name} (hash={new_hash})")
    return True


def check_target_hit(r, stock_code: str, tf: str, live_price: float) -> bool:
    """Check if live_price has reached or crossed target. Return True if target newly hit."""
    key = f"BUY_SIGNALS:{stock_code}"
    tf_name = tf.strip().lower()

    payload_raw = r.hget(key, tf_name)
    if not payload_raw:
        return False

    try:
        data = json.loads(payload_raw)
    except Exception:
        return False

    target = float(data.get("target", 0))
    entry = float(data.get("entry", 0))
    target_status = r.hget(key, f"{tf_name}_target_status")

    if target_status == "sent":
        return False  # already sent

    if live_price >= target and target > 0:
        # Mark as hit
        r.hset(
            key,
            mapping={
                f"{tf_name}_target_status": "sent",
                f"{tf_name}_target_hit_time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
        )
        return True

    return False


import hashlib


# def compute_signal_hash(signal_data: dict) -> str:
#     """Stable hash — ignore volatile fields like last_updated"""

#     def _round(v):
#         try:
#             return round(float(v), 2)
#         except:
#             return v

#     relevant = {
#         "entry": _round(signal_data.get("entry")),
#         "target": _round(signal_data.get("target")),
#         "stoploss": _round(signal_data.get("stoploss")),
#         "signal": signal_data.get("signal"),
#         "timeframe": (signal_data.get("timeframe") or "").lower().strip(),
#         # ❌ remove last_updated — it changes every forecast even if same trade
#     }
#     raw = json.dumps(relevant, sort_keys=True)
#     return hashlib.md5(raw.encode()).hexdigest()


def compute_signal_hash(signal_data: dict) -> str:
    """
    Stable hash for BUY signal.
    MUST include ALGO to prevent UNKNOWN poisoning.
    """

    import json, hashlib

    def _round(v):
        try:
            return round(float(v), 2)
        except Exception:
            return v

    relevant = {
        "entry": _round(signal_data.get("entry")),
        "target": _round(signal_data.get("target")),
        "stoploss": _round(signal_data.get("stoploss")),
        "signal": signal_data.get("signal"),
        "timeframe": (signal_data.get("timeframe") or "").lower().strip(),
        "algo": signal_data.get("ALGO"),  # 🔑 REQUIRED
    }

    raw = json.dumps(relevant, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()

    print(f"[HASH] tf={relevant['timeframe']} algo={relevant['algo']} hash={h}")
    return h
