# ============================================================
# basic_algo5b.py
# PHASE 1 — Imports, Globals, Constants
# ============================================================

# -------------------------
# Standard library imports
# -------------------------
import math
import json
import datetime as dt
from typing import Dict, List, Optional, Any
import datetime as dt

# -------------------------
# Third-party imports
# -------------------------
import pandas as pd
import numpy as np

# -------------------------
# Timezone (India)
# -------------------------
from zoneinfo import ZoneInfo
from redis_util import get_redis

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# STRATEGY ACTIVATION CONFIG
# ============================================================

# Enable / disable BUY strategies here
# Order DOES NOT imply priority (resolution handled later)
ACTIVE_BUY_STRATEGIES = [
    "EMA_RSI_CONFLUENCE",
    "BASE_BREAKOUT",
    "BREAKOUT_RETEST",
    "HL_BOS",
]

import logging
import os

# LOG_PATH = os.path.abspath("algo_debug.log")

# algo_logger = logging.getLogger("ALGO")
# algo_logger.setLevel(logging.INFO)

# # 🚨 IMPORTANT: prevent duplicate handlers
# if not algo_logger.handlers:
#     file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
#     formatter = logging.Formatter("%(asctime)s | %(message)s")
#     file_handler.setFormatter(formatter)
#     algo_logger.addHandler(file_handler)

#     # optional but useful
#     algo_logger.propagate = False

DEBUG_LOG_PATH = os.path.abspath("algo_debug.log")


def debug_log(msg: str):
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{_now_ist_str()} | {msg}\n")
            f.flush()
    except Exception:
        pass


# ============================================================
# TIMEFRAME INDEX MAP (MUST MATCH SERVER)
# ============================================================

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

# Reverse lookup if needed
TF_LABEL_TO_INDEX = {v: k for k, v in TF_INDEX_MAP.items()}


# ============================================================
# NUMERIC CONSTANTS (GLOBAL, STRATEGY-AGNOSTIC)
# ============================================================

# Small buffer to avoid exact SR touches
BUF_FRAC = 0.001  # 0.10%

# Safety epsilon for divisions
EPS = 1e-9

# Minimum acceptable Entry→Target percentage (global guardrail)
GLOBAL_MIN_ETP_PCT = 0.20


# ============================================================
# REQUIRED DATA COLUMNS (VALIDATION ONLY)
# ============================================================

REQUIRED_OHLCV_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

REQUIRED_INDICATOR_COLS = [
    "MA_Fast",  # EMA20
    "RSI14",
    "ATR14",
    "atr_pct_smooth",
]


# ============================================================
# INTERNAL KEYS USED IN UPDATED_CFG
# (Server depends on these — DO NOT RENAME)
# ============================================================

SIGNAL_KEY_BASE = "signal"
ENTRY_KEY_BASE = "entry"
TARGET_KEY_BASE = "target"
STOPLOSS_KEY_BASE = "stoploss"
SUPPORT_KEY_BASE = "support"
RESISTANCE_KEY_BASE = "resistance"
ETP_KEY_BASE = "entry_target_pct"

# Extra metadata (safe to add)
ALGO_SIGNAL_KEY = "algo_signals"


# ============================================================
# SAFE HELPERS (NO LOGIC, NO STATE)
# ============================================================


def _safe_float(v) -> Optional[float]:
    """
    Safely convert value to float.
    Returns None if invalid.
    """
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def _now_ist_str() -> str:
    """
    Current timestamp string in IST.
    """
    return dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# PHASE 2 — Data Validation & Indicator Preparation
# ============================================================


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Validate that required OHLCV columns exist.
    This is a hard gate — strategies must never run on bad data.
    """
    if df is None or df.empty:
        return False

    for col in REQUIRED_OHLCV_COLS:
        if col not in df.columns:
            return False

    return True


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common OHLCV column name variants to standard names.
    This function is idempotent.
    """
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "close_price": "close",
        "vol": "volume",
    }

    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required indicators exist.
    Computes ONLY what is missing.
    Indicators:
      - EMA20  -> MA_Fast
      - RSI14
      - ATR14
      - atr_pct_smooth
    """

    # -------------------------
    # EMA20 (MA_Fast)
    # -------------------------
    if "MA_Fast" not in df.columns:
        df["MA_Fast"] = df["close"].ewm(span=20, adjust=False).mean()

    # -------------------------
    # RSI14
    # -------------------------
    if "RSI14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + EPS)
        df["RSI14"] = 100 - (100 / (1 + rs))

    # -------------------------
    # ATR14 + atr_pct_smooth
    # -------------------------
    if "ATR14" not in df.columns or "atr_pct_smooth" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR14"] = tr.rolling(14).mean()

        # ATR as % of price
        atr_pct = (df["ATR14"] / (df["close"] + EPS)) * 100.0

        # Smooth ATR% for stability
        df["atr_pct_smooth"] = atr_pct.ewm(span=10, adjust=False).mean()

    return df


def prepare_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Full preparation pipeline for a timeframe dataframe.
    Returns cleaned dataframe or None if invalid.
    """

    if df is None or df.empty:
        return None

    df = normalize_ohlcv_columns(df)

    if not validate_ohlcv(df):
        return None

    df = ensure_indicators(df)

    # Final sanity check
    for col in REQUIRED_INDICATOR_COLS:
        if col not in df.columns:
            return None

    return df


def get_latest_index(df: pd.DataFrame) -> Optional[int]:
    """
    Decide which candle index to evaluate for real-time logic.
    Uses the LAST completed candle.
    """
    if df is None or len(df) < 30:
        return None

    # Always use last row (collector guarantees closed candles)
    return len(df) - 1


# ============================================================
# PHASE 3 — Support / Resistance Engine (Shared)
# ============================================================


def compute_support_resistance(df: pd.DataFrame, i: int) -> Dict[str, Optional[float]]:
    """
    Compute shared Support / Resistance for index i.

    Uses:
      - Pivot + S1/R1
      - HH20 / LL20 (swing-based)
    This function is:
      - Strategy-agnostic
      - Read-only
      - Deterministic

    Returns:
      {
        "support": float | None,
        "resistance": float | None
      }
    """

    if df is None or i is None or i <= 0 or i >= len(df):
        return {"support": None, "resistance": None}

    # -------------------------
    # Safe price fetch
    # -------------------------
    high = _safe_float(df["high"].iloc[i])
    low = _safe_float(df["low"].iloc[i])
    close = _safe_float(df["close"].iloc[i])

    if high is None or low is None or close is None:
        return {"support": None, "resistance": None}

    # -------------------------
    # Pivot-based levels
    # -------------------------
    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high

    # -------------------------
    # Swing High / Low (HH20 / LL20)
    # -------------------------
    lookback = 20
    start = max(0, i - lookback + 1)

    try:
        hh20 = float(df["high"].iloc[start : i + 1].max())
    except Exception:
        hh20 = None

    try:
        ll20 = float(df["low"].iloc[start : i + 1].min())
    except Exception:
        ll20 = None

    # -------------------------
    # Candidate pools
    # -------------------------
    res_candidates = [x for x in (r1, hh20) if x is not None]
    sup_candidates = [x for x in (s1, ll20) if x is not None]

    resistance = max(res_candidates) if res_candidates else None
    support = min(sup_candidates) if sup_candidates else None

    # -------------------------
    # Sanity enforcement
    # -------------------------
    if support is not None and resistance is not None:
        if support >= resistance:
            # fallback to swing range if pivot failed
            if hh20 is not None and ll20 is not None and ll20 < hh20:
                support = ll20
                resistance = hh20
            else:
                support = None
                resistance = None

    # Clamp negative values
    if support is not None and support < 0:
        support = 0.0

    return {
        "support": round(support, 6) if support is not None else None,
        "resistance": round(resistance, 6) if resistance is not None else None,
    }


def run_ema_rsi_confluence(
    df: pd.DataFrame, i: int, sr: Dict[str, Optional[float]]
) -> Optional[Dict[str, Any]]:

    if i < 5:
        return None

    close = _safe_float(df["close"].iloc[i])
    open_ = _safe_float(df["open"].iloc[i])
    high = _safe_float(df["high"].iloc[i])
    low_prev = _safe_float(df["low"].iloc[i - 1])

    ema = _safe_float(df["MA_Fast"].iloc[i])
    ema_prev3 = _safe_float(df["MA_Fast"].iloc[i - 3])
    rsi = _safe_float(df["RSI14"].iloc[i])
    rsi_prev = _safe_float(df["RSI14"].iloc[i - 1])
    atr = _safe_float(df["ATR14"].iloc[i])
    atr_pct = _safe_float(df["atr_pct_smooth"].iloc[i])

    if None in (
        close,
        open_,
        high,
        low_prev,
        ema,
        ema_prev3,
        rsi,
        rsi_prev,
        atr,
        atr_pct,
    ):
        return None

    # ---- BUY CONDITIONS ----
    if close <= ema:
        return None
    if (ema - ema_prev3) < (0.002 * close):
        return None
    if not (rsi > 50 and rsi > rsi_prev and rsi <= 80):
        return None
    if atr_pct > 1.8:
        return None
    if (close - open_) < 0.25 * atr:
        return None
    if high <= df["high"].iloc[i - 1]:
        return None

    # ---- ENTRY ----
    body_mid = (close + open_) / 2.0
    entry = max(body_mid, ema + 0.2 * atr)
    entry = min(entry, high * 1.002)

    # ---- STOPLOSS ----
    swing_low = df["low"].iloc[i - 3 : i].min()
    stoploss = swing_low - 0.60 * atr

    sl_pct = ((entry - stoploss) / entry) * 100
    if sl_pct < 0.45 or sl_pct > 2.2:
        return None

    # ---- RR ----
    if atr_pct < 1.0:
        rr = 1.0
    elif atr_pct < 1.3:
        rr = 1.3
    elif atr_pct < 1.5:
        rr = 1.6
    else:
        rr = 1.9

    if rsi > 65:
        rr += 0.3

    target = entry + rr * (entry - stoploss)

    if sr.get("resistance"):
        target = min(target, sr["resistance"] * 0.998)

    etp_pct = ((target - entry) / entry) * 100
    if target <= entry or etp_pct < 0.40:
        return None

    return {
        "algo": "EMA_RSI_CONFLUENCE",
        "signal": "BUY",
        "entry": round(entry, 6),
        "target": round(target, 6),
        "stoploss": round(stoploss, 6),
        "etp_pct": round(etp_pct, 6),
    }


def run_base_breakout(
    df: pd.DataFrame, i: int, sr: Dict[str, Optional[float]]
) -> Optional[Dict[str, Any]]:

    if i < 15:
        return None

    close = _safe_float(df["close"].iloc[i])
    ema = _safe_float(df["MA_Fast"].iloc[i])
    atr = _safe_float(df["ATR14"].iloc[i])
    atr_pct = _safe_float(df["atr_pct_smooth"].iloc[i])
    vol = _safe_float(df["volume"].iloc[i])
    vol_avg = df["volume"].iloc[i - 25 : i].mean()

    if None in (close, ema, atr, atr_pct, vol, vol_avg):
        return None

    if close <= ema * 1.0004:
        return None
    if atr_pct > 9.0:
        return None
    if vol < 1.05 * vol_avg:
        return None

    base_high = df["high"].iloc[i - 10 : i].max()
    base_low = df["low"].iloc[i - 10 : i].min()

    base_range_pct = ((base_high - base_low) / base_low) * 100
    if base_range_pct < 0.20 or base_range_pct > 3.50:
        return None

    if close < base_high * 1.0006:
        return None

    entry = base_high * 1.0009
    stoploss = base_low - 0.15 * atr
    stoploss = min(stoploss, entry * 0.995)

    sl_pct = ((entry - stoploss) / entry) * 100
    if sl_pct < 0.20 or sl_pct > 2.80:
        return None

    rr = 1.5 + min(0.6, atr_pct / 10.0)
    target = entry + rr * (entry - stoploss)

    if sr.get("resistance"):
        target = min(target, sr["resistance"] * 0.999)

    etp_pct = ((target - entry) / entry) * 100
    if target <= entry or etp_pct < 0.20:
        return None

    return {
        "algo": "BASE_BREAKOUT",
        "signal": "BUY",
        "entry": round(entry, 6),
        "target": round(target, 6),
        "stoploss": round(stoploss, 6),
        "etp_pct": round(etp_pct, 6),
    }


def run_breakout_retest(
    df: pd.DataFrame, i: int, sr: Dict[str, Optional[float]]
) -> Optional[Dict[str, Any]]:

    if i < 20:
        return None

    close = _safe_float(df["close"].iloc[i])
    ema = _safe_float(df["MA_Fast"].iloc[i])
    atr = _safe_float(df["ATR14"].iloc[i])
    atr_pct = _safe_float(df["atr_pct_smooth"].iloc[i])
    vol = _safe_float(df["volume"].iloc[i])
    vol_avg = df["volume"].iloc[i - 20 : i].mean()

    if None in (close, ema, atr, atr_pct, vol, vol_avg):
        return None

    if close < ema * 1.00015:
        return None
    if atr_pct > 9.5:
        return None
    if vol < 0.80 * vol_avg:
        return None

    resistance = df["high"].iloc[i - 18 : i - 5].max()
    if close < resistance * 1.00055:
        return None

    retest_low = df["low"].iloc[i - 7 : i].min()
    if not (resistance * 0.998 <= retest_low <= resistance * 1.002):
        return None

    entry = resistance * 1.00055
    stoploss = retest_low - 0.35 * atr
    stoploss = min(stoploss, entry * 0.9935)

    sl_pct = ((entry - stoploss) / entry) * 100
    if sl_pct < 0.22 or sl_pct > 3.10:
        return None

    rr = 1.56 + min(0.82, atr_pct / 6.5)
    target = entry + rr * (entry - stoploss)

    if sr.get("resistance"):
        target = min(target, sr["resistance"] * 0.999)

    etp_pct = ((target - entry) / entry) * 100
    if etp_pct < max(0.28, sl_pct * 1.40):
        return None

    return {
        "algo": "BREAKOUT_RETEST",
        "signal": "BUY",
        "entry": round(entry, 6),
        "target": round(target, 6),
        "stoploss": round(stoploss, 6),
        "etp_pct": round(etp_pct, 6),
    }


def run_hl_bos(
    df: pd.DataFrame, i: int, sr: Dict[str, Optional[float]]
) -> Optional[Dict[str, Any]]:

    if i < 20:
        return None

    close = _safe_float(df["close"].iloc[i])
    open_ = _safe_float(df["open"].iloc[i])
    ema = _safe_float(df["MA_Fast"].iloc[i])
    atr = _safe_float(df["ATR14"].iloc[i])
    atr_pct = _safe_float(df["atr_pct_smooth"].iloc[i])

    if None in (close, open_, ema, atr, atr_pct):
        return None

    if close <= ema:
        return None
    if atr_pct > 9.0:
        return None

    swing_low = df["low"].iloc[i - 6 : i].min()
    prev_swing_low = df["low"].iloc[i - 16 : i - 6].min()
    if swing_low < prev_swing_low * 0.96:
        return None

    prev_swing_high = df["high"].iloc[i - 16 : i].max()
    if close <= prev_swing_high * 1.00002:
        return None

    body = abs(close - open_)
    rng = df["high"].iloc[i] - df["low"].iloc[i]
    if rng <= 0 or body < 0.08 * rng:
        return None

    entry = prev_swing_high * 1.00015
    stoploss = swing_low - 0.10 * atr
    stoploss = min(stoploss, entry * 0.995)

    sl_pct = ((entry - stoploss) / entry) * 100
    if sl_pct < 0.20 or sl_pct > 6.0:
        return None

    rr = min(2.2, 1.4 + sl_pct / 100.0)
    target = entry + rr * (entry - stoploss)

    if sr.get("resistance"):
        target = min(target, sr["resistance"] * 0.999)

    etp_pct = ((target - entry) / entry) * 100
    if etp_pct < max(0.20, 1.3 * sl_pct):
        return None

    return {
        "algo": "HL_BOS",
        "signal": "BUY",
        "entry": round(entry, 6),
        "target": round(target, 6),
        "stoploss": round(stoploss, 6),
        "etp_pct": round(etp_pct, 6),
    }


# ============================================================
# STRATEGY REGISTRY
# ============================================================

BUY_STRATEGY_REGISTRY = {
    "EMA_RSI_CONFLUENCE": run_ema_rsi_confluence,
    "BASE_BREAKOUT": run_base_breakout,
    "BREAKOUT_RETEST": run_breakout_retest,
    "HL_BOS": run_hl_bos,
}


# def run_all_buy_strategies(
#     df: pd.DataFrame,
#     i: int,
#     sr: Dict[str, Optional[float]],
# ) -> List[Dict[str, Any]]:
#     """
#     Execute all ACTIVE_BUY_STRATEGIES independently.
#     HARD isolated: one failure does not affect others.
#     """

#     results: List[Dict[str, Any]] = []

#     for algo_name in ACTIVE_BUY_STRATEGIES:
#         strat_fn = BUY_STRATEGY_REGISTRY.get(algo_name)
#         if not strat_fn:
#             continue

#         try:
#             res = strat_fn(df, i, sr)
#             if res and res.get("signal") == "BUY" and res.get("algo"):
#                 results.append(res)
#         except Exception as e:
#             print(f"[STRATEGY ERROR] {algo_name}: {type(e).__name__}: {e}")
#             continue

#     return results


def run_all_buy_strategies(
    df: pd.DataFrame,
    i: int,
    sr: Dict[str, Optional[float]],
) -> List[Dict[str, Any]]:
    """
    Execute all ACTIVE_BUY_STRATEGIES independently.

    DEBUG GUARANTEES:
    - Logs entry into EACH strategy
    - Logs reject vs pass
    - Multiprocessing-safe (manual file append)
    """

    results: List[Dict[str, Any]] = []

    # ---- Strategy cycle start ----
    try:
        close_px = float(df["close"].iloc[i])
    except Exception:
        close_px = None

    debug_log(f"[STRATEGY CYCLE START] i={i} close={close_px}")

    # ---- Run each strategy independently ----
    for algo_name in ACTIVE_BUY_STRATEGIES:

        debug_log(f"[ENTER] {algo_name}")

        strat_fn = BUY_STRATEGY_REGISTRY.get(algo_name)
        if not strat_fn:
            debug_log(f"[SKIP] {algo_name} not registered")
            continue

        try:
            res = strat_fn(df, i, sr)

            if res is None:
                debug_log(f"[REJECT] {algo_name}")
                continue

            # ---- Strategy passed ----
            debug_log(
                f"[PASS] {algo_name} "
                f"entry={res.get('entry')} "
                f"target={res.get('target')} "
                f"stop={res.get('stoploss')} "
                f"etp={res.get('etp_pct')}"
            )

            results.append(res)

        except Exception as e:
            debug_log(f"[ERROR] {algo_name}: {type(e).__name__}: {e}")

    # ---- Strategy cycle end ----
    debug_log("[STRATEGY CYCLE END]")

    return results


# ============================================================
# BUY RESOLUTION PRIORITY (top = strongest)
# ============================================================

BUY_PRIORITY_ORDER = [
    "EMA_RSI_CONFLUENCE",
    "HL_BOS",
    "BREAKOUT_RETEST",
    "BASE_BREAKOUT",
]


def resolve_buy_signals(
    buy_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Resolve multiple BUY signals into ONE final signal.

    Returns:
        Winning strategy result dict or None
    """

    if not buy_results:
        return None

    # 1️⃣ Priority sort
    def _priority_key(r):
        try:
            return BUY_PRIORITY_ORDER.index(r["algo"])
        except ValueError:
            return 999  # lowest priority

    buy_results = sorted(
        buy_results,
        key=lambda r: (_priority_key(r), -r.get("etp_pct", 0)),
    )

    return buy_results[0]


def build_forecast_update(
    resolved_signal: Optional[Dict[str, Any]],
    sr: Dict[str, Optional[float]],
    tf_index: int,
    tf_label,
    now_ts: str,
) -> Dict[str, Any]:
    """
    Convert resolved BUY signal into forecast config fields.
    DOES NOT touch algo_signals (handled globally).
    """

    out = {}

    if resolved_signal is None:
        out[f"signal{tf_index}" if tf_index != 0 else "signal"] = None
        return out

    # ---- Core signal fields ----
    out[f"signal{tf_index}" if tf_index != 0 else "signal"] = "BUY"
    out[f"entry{tf_index}" if tf_index != 0 else "entry"] = resolved_signal["entry"]
    out[f"target{tf_index}" if tf_index != 0 else "target"] = resolved_signal["target"]
    out[f"stoploss{tf_index}" if tf_index != 0 else "stoploss"] = resolved_signal[
        "stoploss"
    ]
    out[f"entry_target_pct{tf_index}" if tf_index != 0 else "entry_target_pct"] = (
        resolved_signal["etp_pct"]
    )

    # ---- Support / Resistance ----
    out[f"support{tf_index}" if tf_index != 0 else "support"] = sr.get("support")
    out[f"resistance{tf_index}" if tf_index != 0 else "resistance"] = sr.get(
        "resistance"
    )

    # ---- Timestamp ----
    out["last_updated"] = now_ts

    return out


def _load_latest_df(stock_cfg: Dict[str, Any], tf: str) -> Optional[pd.DataFrame]:
    """
    Load latest candles for stock+timeframe from Redis MARKETDATA.
    """

    stock = stock_cfg.get("stock_code")
    if not stock:
        return None

    try:
        r = get_redis()
        key = f"MARKETDATA:{stock}:{tf}"
        raw = r.hgetall(key)

        if not raw:
            debug_log(f"[REDIS EMPTY] {stock} {tf}")
            return None

        rows = []
        for v in raw.values():
            try:
                rows.append(json.loads(v))
            except Exception:
                continue

        if len(rows) < 30:
            debug_log(f"[REDIS INSUFFICIENT] {stock} {tf} rows={len(rows)}")
            return None

        df = pd.DataFrame(rows)

        # normalize column names
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Timestamp": "timestamp",
            },
            inplace=True,
        )

        return df.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        debug_log(f"[REDIS LOAD ERROR] {stock} {tf}: {e}")
        return None


# def basic_forecast_update(
#     stock_cfg: Dict[str, Any],
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     CORE forecast engine (single-winner model).

#     GUARANTEES:
#     - No stale BUY signals
#     - No UNKNOWN algo
#     - Algo + signal always born in same cycle
#     """

#     updated_cfg = dict(stock_cfg)
#     now_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # 🔥 GLOBAL HARD RESET (EVERY CYCLE)
#     updated_cfg["algo_signals"] = {}
#     updated_cfg["winner_algo"] = {}

#     for tf_index, tf_label in TF_INDEX_MAP.items():

#         tf_key = tf_label.lower().strip()

#         # 🔥 HARD RESET PER TIMEFRAME (MOST IMPORTANT FIX)
#         sig_key = f"signal{tf_index}" if tf_index != 0 else "signal"
#         entry_key = f"entry{tf_index}" if tf_index != 0 else "entry"
#         target_key = f"target{tf_index}" if tf_index != 0 else "target"
#         sl_key = f"stoploss{tf_index}" if tf_index != 0 else "stoploss"
#         etp_key = f"entry_target_pct{tf_index}" if tf_index != 0 else "entry_target_pct"

#         updated_cfg[sig_key] = None
#         updated_cfg.pop(entry_key, None)
#         updated_cfg.pop(target_key, None)
#         updated_cfg.pop(sl_key, None)
#         updated_cfg.pop(etp_key, None)

#         updated_cfg["algo_signals"][tf_key] = []
#         updated_cfg["winner_algo"][tf_key] = None

#         # ----------------------------------
#         # LOAD DATA
#         # ----------------------------------
#         df = _load_latest_df(stock_cfg, tf_label)
#         if df is None:
#             continue

#         try:
#             df = prepare_dataframe(df)
#             if df is None or len(df) < 30:
#                 continue

#             # i = len(df) - 1

#             i = len(df) - 2

#             print(
#                 f"[CANDLE] {stock_cfg.get('stock_code')} {tf_label} "
#                 f"i={i} close={df['close'].iloc[i]:.2f} "
#                 f"ema={df['MA_Fast'].iloc[i]:.2f} "
#                 f"rsi={df['RSI14'].iloc[i]:.2f} "
#                 f"atr%={df['atr_pct_smooth'].iloc[i]:.2f}"
#             )

#             sr = compute_support_resistance(df, i - 1)
#             sr_dict = {
#                 "support": sr["support"],
#                 "resistance": sr["resistance"],
#             }

#             buy_results = run_all_buy_strategies(df, i, sr_dict)

#             print(
#                 f"[DEBUG STRATS] {stock_cfg.get('stock_code')} {tf_label} "
#                 f"candidates={[r['algo'] for r in buy_results]}"
#             )

#             resolved = resolve_buy_signals(buy_results)

#             print(
#                 f"[DEBUG RESOLVE] {stock_cfg.get('stock_code')} {tf_label} "
#                 f"winner={resolved['algo'] if resolved else None}"
#             )

#             if resolved:
#                 # 🔑 SINGLE SOURCE OF TRUTH
#                 updated_cfg["algo_signals"][tf_key] = [resolved["algo"]]
#                 updated_cfg["winner_algo"][tf_key] = resolved["algo"]

#                 forecast_patch = build_forecast_update(
#                     resolved_signal=resolved,
#                     sr=sr_dict,
#                     tf_index=tf_index,
#                     tf_label=tf_label,
#                     now_ts=now_ts,
#                 )

#                 updated_cfg.update(forecast_patch)

#         except Exception as e:
#             if verbose:
#                 print(
#                     f"[Forecast Error] {updated_cfg.get('stock_code')} "
#                     f"{tf_label}: {type(e).__name__}: {e}"
#                 )

#     updated_cfg["last_updated"] = now_ts
#     return updated_cfg


def basic_forecast_update(
    stock_cfg: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    CORE forecast engine (single-winner model).

    GUARANTEES:
    - No stale BUY signals
    - No UNKNOWN algo
    - Algo + signal always born in same cycle

    DEBUG:
    - Writes evaluation + BUY confirmation to algo_debug.log
    """
    print("Running basic_forecast_update...")
    updated_cfg = dict(stock_cfg)
    now_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 🔥 GLOBAL HARD RESET (EVERY CYCLE)
    updated_cfg["algo_signals"] = {}
    updated_cfg["winner_algo"] = {}

    for tf_index, tf_label in TF_INDEX_MAP.items():

        tf_key = tf_label.lower().strip()

        # 🔥 HARD RESET PER TIMEFRAME
        sig_key = f"signal{tf_index}" if tf_index != 0 else "signal"
        entry_key = f"entry{tf_index}" if tf_index != 0 else "entry"
        target_key = f"target{tf_index}" if tf_index != 0 else "target"
        sl_key = f"stoploss{tf_index}" if tf_index != 0 else "stoploss"
        etp_key = f"entry_target_pct{tf_index}" if tf_index != 0 else "entry_target_pct"

        updated_cfg[sig_key] = None
        updated_cfg.pop(entry_key, None)
        updated_cfg.pop(target_key, None)
        updated_cfg.pop(sl_key, None)
        updated_cfg.pop(etp_key, None)

        updated_cfg["algo_signals"][tf_key] = []
        updated_cfg["winner_algo"][tf_key] = None

        # ----------------------------------
        # LOAD DATA
        # ----------------------------------
        df = _load_latest_df(stock_cfg, tf_label)
        if df is None:
            debug_log(f"[NO DATA] {stock_cfg.get('stock_code')} {tf_label}")
            continue

        debug_log(f"[DATA OK] {stock_cfg.get('stock_code')} {tf_label} rows={len(df)}")
        try:
            df = prepare_dataframe(df)
            if df is None or len(df) < 30:
                continue

            if df is None:
                debug_log(f"[DF INVALID] {stock_cfg.get('stock_code')} {tf_label}")
                continue

            debug_log(
                f"[DF READY] {stock_cfg.get('stock_code')} {tf_label} "
                f"close={df['close'].iloc[-2]:.2f}"
            )

            # ✅ USE LAST CLOSED CANDLE
            i = len(df) - 2

            # ----------------------------------
            # SUPPORT / RESISTANCE
            # ----------------------------------
            sr = compute_support_resistance(df, i - 1)
            sr_dict = {
                "support": sr["support"],
                "resistance": sr["resistance"],
            }

            # ----------------------------------
            # RUN STRATEGIES
            # ----------------------------------
            buy_results = run_all_buy_strategies(df, i, sr_dict)

            resolved = resolve_buy_signals(buy_results)

            # ----------------------------------
            # BUY CONFIRMATION (ONLY WHEN TRIGGERED)
            # ----------------------------------
            if resolved:

                updated_cfg["algo_signals"][tf_key] = [resolved["algo"]]
                updated_cfg["winner_algo"][tf_key] = resolved["algo"]

                forecast_patch = build_forecast_update(
                    resolved_signal=resolved,
                    sr=sr_dict,
                    tf_index=tf_index,
                    tf_label=tf_label,
                    now_ts=now_ts,
                )

                updated_cfg.update(forecast_patch)

        except Exception as e:
            if verbose:
                print(
                    f"[Forecast Error] {updated_cfg.get('stock_code')} "
                    f"{tf_label}: {type(e).__name__}: {e}"
                )

    updated_cfg["last_updated"] = now_ts
    return updated_cfg
