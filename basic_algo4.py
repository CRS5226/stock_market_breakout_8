# basic_algo_forecaster.py
import os
import glob
import copy
import math
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Optional, Tuple
import math
import datetime as dt


from redis_util import (
    get_recent_candles,  # legacy wrapper -> returns newest-first list
    get_redis,
    get_recent_candles_tf,  # newest-first list
    get_recent_indicators_tf,  # newest-first list
)


# -------------------- utils (unchanged) --------------------


# Toggle: use relaxed thresholds like backtest (keeps your previous behaviour if you had RELAX_MODE)
RELAX_MODE = True


def safe_number(val, default=0):
    """Robust numeric coercion used across the forecaster (unified with backtest semantics)."""
    try:
        if val is None:
            return default
        if isinstance(val, (int, float, np.floating, np.integer)):
            v = float(val)
            if math.isnan(v) or v in (float("inf"), float("-inf")):
                return default
            return v
        # attempt numeric cast
        v = float(val)
        if math.isnan(v) or v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _get(df: pd.DataFrame, *candidates, default=None):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name is None:
            continue
        key = str(name).lower()
        if key in cols:
            return df[cols[key]]
    return pd.Series([default] * len(df), index=df.index)


def _last_val(row: pd.Series, *candidates, default=None):
    """
    Robust retrieval from a Series by trying exact names then case-insensitive matches.
    Works with both pandas Series and mapping-like rows returned by Redis.
    """
    # try exact names first
    for name in candidates:
        if name is None:
            continue
        try:
            if name in row and pd.notna(row[name]):
                return row[name]
        except Exception:
            # row might be a dict-like where 'in' fails for some types; skip safely
            pass

    # try lowercase matching across columns
    lname_targets = [str(n).lower() for n in candidates if n is not None]
    for c in row.index:
        try:
            if str(c).lower() in lname_targets and pd.notna(row[c]):
                return row[c]
        except Exception:
            continue

    return default


# ---- Timezone helpers ----
LOCAL_TZ = "Asia/Kolkata"  # IST


def to_ist(series: pd.Series, keep_tz=True) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(LOCAL_TZ)
    else:
        s = s.dt.tz_localize(LOCAL_TZ)
    if not keep_tz:
        s = s.dt.tz_localize(None)
    return s


def compute_support_resistance(df: pd.DataFrame):
    """
    Backtest-style SR computation adapted for the forecaster.
    Returns (support, resistance, bb_upper, bb_lower) rounded to 2 decimals where possible.
    Logic:
      - Use HH20/LL20 if present for swing high/low.
      - Compute pivot/R1/S1 when High/Low/Close are available; use R1/S1 + BB + HH20/LL20 to decide SR.
      - Fallback robustly to available series values.
    """
    if df is None or df.empty:
        return None, None, None, None

    # helper to find columns case-insensitively
    cols = {c.lower(): c for c in df.columns}

    def col_get(name):
        return (
            df[cols[name.lower()]]
            if name.lower() in cols
            else pd.Series([None] * len(df))
        )

    # core series (case-insensitive)
    high_s = None
    low_s = None
    close_s = None
    if "high" in cols:
        high_s = df[cols["high"]]
    if "low" in cols:
        low_s = df[cols["low"]]
    if "close" in cols:
        close_s = df[cols["close"]]

    # hh20 / ll20 candidates
    hh20_s = None
    ll20_s = None
    if "hh20" in cols:
        hh20_s = df[cols["hh20"]]
    if "ll20" in cols:
        ll20_s = df[cols["ll20"]]

    # BBs
    bb_upper_s = None
    bb_lower_s = None
    if "bb_upper" in cols:
        bb_upper_s = df[cols["bb_upper"]]
    if "bb_lower" in cols:
        bb_lower_s = df[cols["bb_lower"]]

    # last row fallback values
    latest = df.iloc[-1]
    # prefer exact keys if present (works with different input shapes)
    high = _last_val(latest, "High", "high", default=None)
    low = _last_val(latest, "Low", "low", default=None)
    close = _last_val(latest, "Close", "close", default=None)

    try:
        # determine swing high/low using hh20/ll20 if present, else tail max/min
        swing_high = (
            safe_number(hh20_s.iloc[-1])
            if (hh20_s is not None and hh20_s.notna().any())
            else (safe_number(high_s.tail(20).max()) if high_s is not None else None)
        )
    except Exception:
        swing_high = None

    try:
        swing_low = (
            safe_number(ll20_s.iloc[-1])
            if (ll20_s is not None and ll20_s.notna().any())
            else (safe_number(low_s.tail(20).min()) if low_s is not None else None)
        )
    except Exception:
        swing_low = None

    # compute pivot/r1/s1 if we have high/low/close
    r1 = None
    s1 = None
    if all(v is not None for v in (high, low, close)):
        try:
            pivot = (float(high) + float(low) + float(close)) / 3.0
            r1 = 2 * pivot - float(low)
            s1 = 2 * pivot - float(high)
        except Exception:
            r1 = None
            s1 = None
    else:
        # fallback to swing levels
        r1 = swing_high
        s1 = swing_low

    # bb fallback defaults
    try:
        bb_upper = (
            safe_number(bb_upper_s.iloc[-1])
            if (bb_upper_s is not None and bb_upper_s.notna().any())
            else r1
        )
    except Exception:
        bb_upper = r1
    try:
        bb_lower = (
            safe_number(bb_lower_s.iloc[-1])
            if (bb_lower_s is not None and bb_lower_s.notna().any())
            else s1
        )
    except Exception:
        bb_lower = s1

    # choose resistance as max of available candidates, support as min
    # candidates_res = [v for v in (r1, bb_upper, swing_high) if v is not None]
    # candidates_sup = [v for v in (s1, bb_lower, swing_low) if v is not None]

    # resistance = max(candidates_res) if candidates_res else None
    # support = min(candidates_sup) if candidates_sup else None

    # --- Stabilized version ---
    candidates_res = [v for v in (r1, swing_high) if v is not None]
    candidates_sup = [v for v in (s1, swing_low) if v is not None]

    # Fallback if no valid values
    if not candidates_res and bb_upper is not None:
        candidates_res = [bb_upper]
    if not candidates_sup and bb_lower is not None:
        candidates_sup = [bb_lower]

    resistance = max(candidates_res) if candidates_res else None
    support = max(min(candidates_sup), 0) if candidates_sup else None

    # final sanity: ensure ordering
    if support is not None and resistance is not None:
        if support >= resistance:
            # if inverted, swap to keep order but log (no logging here, just ensure valid)
            support, resistance = min(support, resistance), max(support, resistance)

    # rounding to 2 decimals keeps output compact & matches sample
    try:
        support_r = round(float(support), 2) if support is not None else None
    except Exception:
        support_r = None
    try:
        resistance_r = round(float(resistance), 2) if resistance is not None else None
    except Exception:
        resistance_r = None
    try:
        bb_upper_r = round(float(bb_upper), 2) if bb_upper is not None else None
    except Exception:
        bb_upper_r = None
    try:
        bb_lower_r = round(float(bb_lower), 2) if bb_lower is not None else None
    except Exception:
        bb_lower_r = None

    return support_r, resistance_r, bb_upper_r, bb_lower_r


# ---------------------- entry/target/stoploss (tweaked) ----------------------


def _compute_pair_levels(
    support,
    resistance,
    current_price,
    tf: str = "1day",
    candle_high: float = None,
    candle_low: float = None,
):
    """
    Backtest-style Entry/Target/Stoploss computation adapted for forecaster.

    - Uses High/Low range overlap with SR (not just Close).
    - Uses TF-specific clamps (SL_BOUNDS, R_MULT_BY_TF, ETP_MIN_BY_TF, MIN_ENTRY_BUF_BY_TF).
    - Uses 2R logic capped at resistance.
    - Asymmetric SL widening allowed.
    - Hard acceptance band enforced.

    Returns (entry, target, stoploss).
    """

    # TF multipliers (aligns with backtest constants)
    SL_BOUNDS = {
        "1min": (0.003, 0.010),
        "5min": (0.005, 0.015),
        "15min": (0.008, 0.020),
        "30min": (0.010, 0.025),
        "45min": (0.010, 0.030),
        "1hour": (0.012, 0.030),
        "4hour": (0.015, 0.040),
        "1day": (0.020, 0.050),
        "1month": (0.030, 0.100),
    }

    R_MULT_BY_TF = {
        "1min": 2.0,
        "5min": 2.0,
        "15min": 2.0,
        "30min": 2.0,
        "45min": 2.0,
        "1hour": 2.0,
        "4hour": 2.0,
        "1day": 2.0,
        "1month": 2.0,
    }

    ETP_MIN_BY_TF = {
        "1min": 0.15,
        "5min": 0.20,
        "15min": 0.25,
        "30min": 0.30,
        "45min": 0.40,
        "1hour": 0.50,
        "4hour": 0.70,
        "1day": 1.00,
        "1month": 2.00,
    }

    MIN_ENTRY_BUF_BY_TF = {
        "1min": 0.02,
        "5min": 0.03,
        "15min": 0.05,
        "30min": 0.07,
        "45min": 0.10,
        "1hour": 0.15,
        "4hour": 0.20,
        "1day": 0.30,
        "1month": 0.50,
    }

    # clamp values from dicts with defaults
    sl_min, sl_max = SL_BOUNDS.get(tf, (0.01, 0.05))
    r_mult = R_MULT_BY_TF.get(tf, 2.0)
    etp_min = ETP_MIN_BY_TF.get(tf, 0.25)
    entry_buf_min = MIN_ENTRY_BUF_BY_TF.get(tf, 0.05)

    try:
        cp = float(current_price)
        sup = float(support)
        res = float(resistance)
        hi = float(candle_high) if candle_high is not None else cp
        lo = float(candle_low) if candle_low is not None else cp
    except Exception:
        return None, None, None

    if any(v is None or math.isnan(v) or v <= 0 for v in (cp, sup, res, hi, lo)):
        return None, None, None

    # --- new: allow if High/Low overlaps SR range ---
    if not (lo < res and hi > sup):
        return None, None, None

    rng = res - sup
    if rng <= 0:
        return None, None, None

    # entry just above current price, with minimum buffer
    entry = cp * (1.0 + entry_buf_min / 100.0)
    entry = min(entry, res * 0.999)

    # stoploss below support
    stoploss = sup * (1.0 - sl_min / 2.0)
    raw_sl_pct = (entry - stoploss) / entry * 100.0
    if raw_sl_pct < sl_min * 100.0:
        stoploss = entry * (1.0 - sl_min)
    if raw_sl_pct > sl_max * 100.0:
        stoploss = entry * (1.0 - sl_max)

    # target = entry + R*SL, capped at resistance
    sl_dist = entry - stoploss
    target = entry + r_mult * sl_dist
    target = min(target, res * 0.999)

    # enforce minimum ETP%
    etp_pct = ((target - entry) / entry) * 100.0
    if etp_pct < etp_min:
        if res > entry:
            target = min(res * 0.999, entry * (1.0 + etp_min / 100.0))
            etp_pct = ((target - entry) / entry) * 100.0
        if etp_pct < etp_min:
            return None, None, None

    # sanity: ordering
    tol = res * 0.001
    if not (stoploss < sup < cp < entry < target <= res + tol):
        return None, None, None

    return (
        round(float(entry), 3),
        round(float(target), 3),
        round(float(stoploss), 3),
    )


# -------------------- respect counts (unchanged) --------------------
def _colmap(df):
    cols = {c.lower(): c for c in df.columns}
    return cols.get("high"), cols.get("low"), cols.get("close")


def _count_respects(df, level, side, tol_frac=0.0015, min_sep=3):
    if level is None or df is None or df.empty:
        return 0
    H, L, C = _colmap(df)
    if not (H and L and C):
        return 0
    cnt, last_i = 0, -(10**9)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            hi = float(row[H])
            lo = float(row[L])
            cl = float(row[C])
        except Exception:
            continue
        if side == "support":
            near = abs(lo - level) / max(1e-9, level) <= tol_frac and cl >= level
            wick = (lo < level) and (cl > level)
            hit = near or wick
        else:
            near = abs(hi - level) / max(1e-9, level) <= tol_frac and cl <= level
            wick = (hi > level) and (cl < level)
            hit = near or wick
        if hit and (i - last_i) >= min_sep:
            cnt += 1
            last_i = i
    return cnt


# --------------------------- helpers ---------------------------
# --- NEW: file-based TF history discovery ---
HIST_ROOTS_DEFAULT = [
    os.path.join("historical_data_candles"),
    os.path.join("p17", "historical_data_candles"),
]

# map TF → subfolder and filename token used in your files
TF_FILE_MAP = {
    "1min": ("1min", "1min"),
    "5min": ("5min", "5min"),
    "15min": ("15min", "15min"),
    "30min": ("30min", "30min"),
    "45min": ("45min", "45min"),
    "1hour": ("1hour", "1hour"),
    "4hour": ("4hour", "4hour"),
    "1day": ("1day", "1day"),  # daily
    "1month": ("1day", "1day"),  # we'll resample this from daily later
}


def _glob_hist_csv(
    stock_code: str, tf: str, roots: list[str] | None = None
) -> Optional[str]:
    """
    Find the latest CSV for stock_code & tf under given roots.
    We match files like ANANDRATHI_1day_YYYY-MM-DD_to_YYYY-MM-DD.csv.
    """
    roots = roots or HIST_ROOTS_DEFAULT
    if tf not in TF_FILE_MAP:
        return None
    subdir, token = TF_FILE_MAP[tf]
    patterns = []
    for root in roots:
        base = os.path.join(root, subdir)
        # windows/backslash-safe glob
        patterns.append(os.path.join(base, f"{stock_code}_{token}_*.csv"))
    # collect all matches
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    if not matches:
        return None
    # pick the most recent by mtime (robust) instead of parsing dates
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _load_hist_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common columns
    cols = {c.lower(): c for c in df.columns}
    # Timestamp column
    if "timestamp" not in cols:
        for cand in ("date", "datetime", "time", "minute"):
            if cand in cols:
                df = df.rename(columns={cols[cand]: "Timestamp"})
                break
    else:
        df = df.rename(columns={cols["timestamp"]: "Timestamp"})

    # Core OHLCV
    renames = {}
    for raw, std in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        if raw in cols and std not in df.columns:
            renames[cols[raw]] = std
    if renames:
        df = df.rename(columns=renames)

    if "Timestamp" in df.columns:
        df["Timestamp"] = to_ist(df["Timestamp"], keep_tz=True)
        df = (
            df.dropna(subset=["Timestamp"])
            .drop_duplicates(subset=["Timestamp"])
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    return df


# -------------------- multi-timeframe SR + levels --------------------


def _resample_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty or "Timestamp" not in df_daily.columns:
        return pd.DataFrame()
    df = df_daily.copy()
    df = df.set_index(pd.to_datetime(df["Timestamp"]))
    agg = {}
    for k, v in [
        ("Open", "first"),
        ("High", "max"),
        ("Low", "min"),
        ("Close", "last"),
        ("Volume", "sum"),
        ("atr_pct", "last"),
        ("BB_Upper", "last"),
        ("BB_Lower", "last"),
        ("HH20", "last"),
        ("LL20", "last"),
    ]:
        if k in df.columns:
            agg[k] = v
    out = df.resample("ME").agg(agg).dropna(how="all")
    if out.empty:
        return pd.DataFrame()
    out = out.reset_index().rename(columns={"index": "Timestamp"})
    return out


def _tf_recent_from_redis(
    stock_code: str, tf: str, n: int = 200, quite=False
) -> pd.DataFrame:
    """
    Fetch recent candles when stored as Redis HASH:
      key = MARKETDATA:{stock}:{tf}
      field = timestamp (ISO string)
      value = JSON (OHLC + indicators)
    Returns chronological DataFrame (oldest→newest).
    """
    import json
    import pandas as pd

    try:
        r = get_redis()
    except Exception as e:
        print(f"[Redis conn error] {e}")
        return pd.DataFrame()

    key = f"MARKETDATA:{stock_code}:{tf}"
    try:
        if not r.exists(key):
            print(f"[ℹ️ TF fetch] {stock_code} {tf}: no Redis hash found ({key}).")
            return pd.DataFrame()

        # Redis type check — safe for both bytes and str
        key_type = r.type(key)
        if isinstance(key_type, bytes):
            key_type = key_type.decode(errors="ignore")
        if key_type != "hash":
            print(f"[⚠️] {stock_code} {tf}: expected hash but found {key_type}.")
            return pd.DataFrame()

        data_map = r.hgetall(key)
        if not data_map:
            print(f"[ℹ️ TF fetch] {stock_code} {tf}: hash empty.")
            return pd.DataFrame()

        rows = []
        for ts_raw, val_raw in data_map.items():
            try:
                ts_str = str(ts_raw)
                val_str = str(val_raw)
                row = json.loads(val_str)
                if "Timestamp" not in row:
                    row["Timestamp"] = ts_str
                rows.append(row)
            except Exception as e:
                print(f"[⚠️ Parse error] {stock_code} {tf}: {e}")
                continue

        if not rows:
            print(f"[ℹ️ TF fetch] {stock_code} {tf}: parsed 0 rows.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
        df = (
            df.dropna(subset=["Timestamp"])
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

        if len(df) > n:
            df = df.tail(n).reset_index(drop=True)

        print(
            f"[✅ TF fetch] {stock_code} {tf}: loaded {len(df)} rows from Redis hash."
        )
        return df

    except Exception as e:
        print(f"[❌ TF fetch error] {stock_code} {tf}: {e}")
        return pd.DataFrame()


def _tf_hist_source(
    df_hist_daily: pd.DataFrame,
    tf: str,
    stock_code: str,
    hist_roots: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prefer per-TF CSV from disk; for 1month we resample daily.
    If no TF CSV exists, fall back to df_hist_daily where sensible.
    """
    # daily as baseline
    if tf == "1day":
        # daily history from disk if available, else df_hist_daily
        p = _glob_hist_csv(stock_code, "1day", hist_roots)
        if p:
            return _load_hist_csv(p)
        return df_hist_daily.copy() if df_hist_daily is not None else pd.DataFrame()

    if tf == "1month":
        # build from daily
        daily = _tf_hist_source(df_hist_daily, "1day", stock_code, hist_roots)
        return _resample_daily_to_monthly(daily)

    # intraday buckets
    p = _glob_hist_csv(stock_code, tf, hist_roots)
    if p:
        return _load_hist_csv(p)

    # no intraday CSV → fall back to daily just to derive rough SR
    return df_hist_daily.copy() if df_hist_daily is not None else pd.DataFrame()


def build_all_timeframe_levels(
    stock_code: str, df_dict: dict[str, pd.DataFrame], verbose: bool = True
) -> dict:
    """
    Compute SR, entry/target/stoploss, respected counts, ETP%, and consolidation
    across TF0..TF8. Returns a flat dict of keys matching final JSON schema.

    - df_dict should be: { "1min": df, "5min": df, ..., "4hour": df }
    - If a TF is missing in df_dict, fallback to Redis/historical fetch.
    """

    # mapping TF indices to names (TF0 is 1min base)
    TF_MAP = {
        0: "1min",
        1: "5min",
        2: "15min",
        3: "30min",
        4: "45min",
        5: "1hour",
        6: "4hour",
        7: "1day",  # ✅ Add daily
        8: "1month",  # ✅ Add monthly
    }

    result = {}

    for idx, tf in TF_MAP.items():
        suffix = "" if idx == 0 else str(idx)

        # --- Pick dataframe ---
        df_tf = None
        if tf in df_dict and not df_dict[tf].empty:
            df_tf = df_dict[tf]
        else:
            try:
                if tf in ["1day", "1month"]:
                    df_tf = _tf_hist_source(None, tf, stock_code)
                else:
                    df_tf = _tf_recent_from_redis(stock_code, tf=tf, n=200)
            except Exception:
                df_tf = None

        if df_tf is None or df_tf.empty:
            if verbose:
                print(f"[{stock_code}] TF{idx} ({tf}) missing data")
            continue

        if tf != "1min" and "Volume" in df_tf.columns:
            nonzero = df_tf[df_tf["Volume"] > 0]
            if len(nonzero) < max(5, int(0.1 * len(df_tf))):
                if verbose:
                    print(
                        f"[WARN] {stock_code} {tf}: too many zero-volume bars — skipping TF"
                    )
                continue

        # --- Support/Resistance ---
        support, resistance, bb_u, bb_l = compute_support_resistance(df_tf)
        if support is None or resistance is None:
            if verbose:
                print(f"[{stock_code}] TF{idx} ({tf}) missing SR")
            continue

        # --- Latest candle values ---
        latest = df_tf.iloc[-1]
        try:
            cp = float(latest["Close"])
            hi = float(latest["High"])
            lo = float(latest["Low"])
        except Exception:
            if verbose:
                print(f"[{stock_code}] TF{idx} ({tf}) missing OHLC values")
            continue

        # --- Entry/Target/Stoploss ---
        entry, target, stoploss = _compute_pair_levels(
            support, resistance, cp, tf=tf, candle_high=hi, candle_low=lo
        )
        # if entry is None or target is None or stoploss is None:
        #     if verbose:
        #         print(
        #             f"[{stock_code}] TF{idx} ({tf}) invalid E/T/SL "
        #             f"(cp={cp}, sup={support}, res={resistance}, hi={hi}, lo={lo})"
        #         )
        #     continue

        # --- Respected counts ---
        resS = _count_respects(df_tf, support, "support")
        resR = _count_respects(df_tf, resistance, "resistance")

        # --- Entry/Target % ---
        try:
            etp_pct = round(((target - entry) / entry) * 100.0, 2)
        except Exception:
            etp_pct = None

        # --- Consolidation score ---
        new_cfg = {"ohlcv": {"close": cp}}
        new_cfg = compute_consolidation(
            new_cfg, stock_code, tf, idx, df_hist_daily=None
        )
        cons_val = new_cfg.get(
            "consolidation_score" if idx == 0 else f"consolidation_score{idx}", 0.0
        )

        # --- Store into result dict ---
        result[f"support{suffix}"] = support
        result[f"resistance{suffix}"] = resistance
        result[f"entry{suffix}"] = entry
        result[f"target{suffix}"] = target
        result[f"stoploss{suffix}"] = stoploss
        result[f"respected_S{suffix}"] = resS
        result[f"respected_R{suffix}"] = resR
        if etp_pct is not None:
            result[f"entry_target_pct{suffix}"] = etp_pct
        result[f"consolidation_score{suffix}"] = cons_val

    return result


# ---------- Phase 2.5: stricter completeness gate (_all_tf_complete) ----------
def _has_val(x) -> bool:
    return x is not None and str(x) != ""


def _is_number(x) -> bool:
    try:
        if x is None:
            return False
        v = float(x)
        return not (math.isnan(v) or v in (float("inf"), float("-inf")))
    except Exception:
        return False


def _tf_keys(i: int):
    # TF0 uses unsuffixed keys; TF1..8 use suffixed keys
    if i == 0:
        return ("support", "resistance", "entry", "target", "stoploss")
    return (f"support{i}", f"resistance{i}", f"entry{i}", f"target{i}", f"stoploss{i}")


def _all_tf_complete(cfg: dict, require_triplets: bool = True) -> tuple[bool, list]:
    """
    Stricter completeness check for TF0..TF8.

    Returns (ok, missing_list).
    - If require_triplets==True: requires for each TF:
        support{i}, resistance{i}, entry{i}, target{i}, stoploss{i}
      and also validates basic numeric ordering:
        stoploss < support < entry < target < resistance
      Per-TF failures produce readable hints (e.g. "TF3: missing entry3", "TF5: ordering invalid stop>=support").
    - If require_triplets==False: only requires support/resistance exist & numeric.

    Keeps same API as earlier but is more defensive and descriptive.
    """
    missing = []
    for i in range(0, 9):
        s_key, r_key, e_key, t_key, sl_key = _tf_keys(i)

        s_val = cfg.get(s_key)
        r_val = cfg.get(r_key)

        # check S/R presence first
        if not (
            _has_val(s_val)
            and _has_val(r_val)
            and _is_number(s_val)
            and _is_number(r_val)
        ):
            missing.append(f"TF{i}: missing/invalid {s_key}/{r_key}")
            # if S/R missing we skip deeper checks for this TF
            continue

        # If only S/R required, continue
        if not require_triplets:
            continue

        # require entry/target/stoploss present and numeric
        e_val = cfg.get(e_key)
        t_val = cfg.get(t_key)
        sl_val = cfg.get(sl_key)
        if not (_has_val(e_val) and _has_val(t_val) and _has_val(sl_val)):
            missing.append(f"TF{i}: missing {e_key}/{t_key}/{sl_key}")
            continue
        if not (_is_number(e_val) and _is_number(t_val) and _is_number(sl_val)):
            missing.append(f"TF{i}: non-numeric {e_key}/{t_key}/{sl_key}")
            continue

        # ordering invariants: stop < support < entry < target < resistance
        try:
            s_n = float(s_val)
            r_n = float(r_val)
            e_n = float(e_val)
            t_n = float(t_val)
            sl_n = float(sl_val)
        except Exception:
            missing.append(f"TF{i}: invalid numeric casting for ordering check")
            continue

        # Validate strict ordering with a tiny tolerance to avoid equality edge-cases
        tiny = 1e-9
        if not (sl_n + tiny < s_n - tiny):
            missing.append(f"TF{i}: stoploss >= support (stop={sl_n}, sup={s_n})")
        if not (s_n + tiny < e_n - tiny):
            missing.append(f"TF{i}: entry <= support (entry={e_n}, sup={s_n})")
        if not (e_n + tiny < t_n - tiny):
            missing.append(f"TF{i}: target <= entry (target={t_n}, entry={e_n})")
        if not (t_n + tiny < r_n - tiny):
            # allow target==resistance - tiny (but flag if target >= resistance)
            missing.append(f"TF{i}: target >= resistance (target={t_n}, res={r_n})")

    return (len(missing) == 0, missing)


def compute_consolidation(
    new_cfg: dict,
    stock_code: str,
    tf: str,
    idx: int,
    df_hist_daily: pd.DataFrame | None = None,
) -> dict:
    """
    Backtest-style consolidation score writer.
    Signature preserved: writes to new_cfg[out_key] where out_key is 'consolidation_score'
    for idx==0, otherwise 'consolidation_score{idx}'.

    Uses weighted BB width (60%) and ATR% (40%) with TF-dependent thresholds.
    """

    def _clamp01(x):
        try:
            x = float(x)
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return 0.0 if x < 0 else (1.0 if x > 1 else x)
        except Exception:
            return 0.0

    out_key = "consolidation_score" if idx == 0 else f"consolidation_score{idx}"

    try:
        # pick a source for indicators: monthly/daily use historical daily; intraday prefer recent TF (handled by caller)
        if tf in ["1day", "1month"]:
            df_ind = (
                _tf_hist_source(df_hist_daily, tf, stock_code)
                if "_tf_hist_source" in globals()
                else (df_hist_daily or pd.DataFrame())
            )
        else:
            # caller already has _tf_recent_from_redis; fallback to df_hist_daily if not present
            try:
                df_ind = _tf_recent_from_redis(stock_code, tf=tf, n=50, quite=True)
            except Exception:
                df_ind = pd.DataFrame()

        # current close
        close_val = (
            float(new_cfg["ohlcv"]["close"])
            if new_cfg.get("ohlcv") and new_cfg["ohlcv"].get("close") is not None
            else None
        )

        bb_upper = bb_lower = atr_pct = None
        if df_ind is not None and not df_ind.empty:
            cols = {c.lower(): c for c in df_ind.columns}
            if "bb_upper" in cols:
                bb_upper = safe_number(df_ind[cols["bb_upper"]].iloc[-1], None)
            if "bb_lower" in cols:
                bb_lower = safe_number(df_ind[cols["bb_lower"]].iloc[-1], None)
            if "atr_pct" in cols:
                atr_pct = safe_number(df_ind[cols["atr_pct"]].iloc[-1], None)

        bbw_pct = None
        if close_val and bb_upper is not None and bb_lower is not None:
            try:
                bbw_pct = ((bb_upper - bb_lower) / close_val) * 100.0
            except Exception:
                bbw_pct = None

        # thresholds by TF (mirrors backtest)
        if RELAX_MODE:
            if tf in ["1min"]:
                bbw_th, atr_th = 5.0, 2.0
            elif tf in ["5min", "15min", "30min"]:
                bbw_th, atr_th = 8.0, 4.0
            elif tf in ["45min", "1hour"]:
                bbw_th, atr_th = 20.0, 6.0
            elif tf == "4hour":
                bbw_th, atr_th = 40.0, 10.0
            elif tf == "1day":
                bbw_th, atr_th = 100.0, 15.0
            elif tf == "1month":
                bbw_th, atr_th = 200.0, 20.0
            else:
                bbw_th, atr_th = 10.0, 4.0
        else:
            if tf in ["1min", "5min", "15min", "30min", "45min", "1hour", "4hour"]:
                bbw_th, atr_th = 5.0, 2.0
            elif tf == "1day":
                bbw_th, atr_th = 25.0, 10.0
            elif tf == "1month":
                bbw_th, atr_th = 40.0, 15.0
            else:
                bbw_th, atr_th = 5.0, 2.0

        parts = []
        weights = []

        if bbw_pct is not None:
            parts.append(max(0.0, min(1.0, 1.0 - (bbw_pct / bbw_th))))
            weights.append(0.6)
        if atr_pct is not None:
            parts.append(max(0.0, min(1.0, 1.0 - (atr_pct / atr_th))))
            weights.append(0.4)

        if parts:
            wsum = sum(weights) if sum(weights) > 0 else 1.0
            score01 = sum(p * (w / wsum) for p, w in zip(parts, weights))
            new_cfg[out_key] = float(round(score01 * 100.0, 1))
        else:
            new_cfg[out_key] = 0.0

    except Exception as e:
        # keep robust: if anything fails, write 0.0
        new_cfg[out_key] = 0.0

    return new_cfg


# ---------- Phase 3: BUY signal logic (ported from backtest) ----------


def _compute_consolidation_score_for_row(row: pd.Series, tf: str) -> float:
    """Single-row consolidation score (0–100), same as backtest logic."""

    def _safe(v, d=0.0):
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return d
            return f
        except Exception:
            return d

    close = _safe(row.get("Close"))
    bb_u = _safe(row.get("BB_Upper"))
    bb_l = _safe(row.get("BB_Lower"))
    atr_pct = _safe(row.get("atr_pct"))
    bbw_pct = ((bb_u - bb_l) / close * 100.0) if close and bb_u and bb_l else None

    if tf in ["1min"]:
        bbw_th, atr_th = 5, 2
    elif tf in ["5min", "15min", "30min"]:
        bbw_th, atr_th = 8, 4
    elif tf in ["45min", "1hour"]:
        bbw_th, atr_th = 20, 6
    elif tf == "4hour":
        bbw_th, atr_th = 40, 10
    elif tf == "1day":
        bbw_th, atr_th = 500, 20
    elif tf == "1month":
        bbw_th, atr_th = 1500, 25
    else:
        bbw_th, atr_th = 10, 4

    parts, weights = [], []
    if bbw_pct is not None:
        parts.append(max(0, min(1, 1 - (bbw_pct / bbw_th))))
        weights.append(0.6)
    if atr_pct is not None:
        parts.append(max(0, min(1, 1 - (atr_pct / atr_th))))
        weights.append(0.4)
    if not parts:
        return 0.0
    score = sum(p * (w / sum(weights)) for p, w in zip(parts, weights))
    return round(score * 100.0, 1)


def _tf_thresholds(tf: str) -> dict:
    """Thresholds per TF — relaxed when RELAX_MODE=True."""
    table = {
        "1min": dict(adx_min=8, rsi_min=35, cons_min=16, etp_min=0.30),
        "5min": dict(adx_min=10, rsi_min=37, cons_min=18, etp_min=0.45),
        "15min": dict(adx_min=12, rsi_min=39, cons_min=20, etp_min=0.70),
        "30min": dict(adx_min=14, rsi_min=41, cons_min=22, etp_min=0.90),
        "45min": dict(adx_min=15, rsi_min=42, cons_min=23, etp_min=1.00),
        "1hour": dict(adx_min=16, rsi_min=43, cons_min=24, etp_min=1.10),
        "4hour": dict(adx_min=18, rsi_min=45, cons_min=26, etp_min=1.30),
        "1day": dict(adx_min=18, rsi_min=46, cons_min=28, etp_min=1.50),
        "1month": dict(adx_min=18, rsi_min=46, cons_min=28, etp_min=1.70),
    }

    return table.get(tf, dict(adx_min=14, rsi_min=41, cons_min=22, etp_min=0.9))


def is_buy_signal_forecaster(
    latest: pd.Series, tf: str, sr: dict, etp: Optional[float]
):
    reasons = []
    try:
        close = float(latest.get("Close") or latest.get("close"))
    except Exception:
        return False, ["missing close"]

    sup, res = sr.get("support"), sr.get("resistance")
    if sup is None or res is None:
        return False, ["missing support/resistance"]

    cons = _compute_consolidation_score_for_row(latest, tf)
    p = _tf_thresholds(tf)

    rsi = latest.get("RSI14") or latest.get("rsi14")
    adx = latest.get("ADX") or latest.get("adx14")
    plus_di = latest.get("+DI") or latest.get("plus_di")
    minus_di = latest.get("-DI") or latest.get("minus_di")
    macd = latest.get("MACD") or latest.get("macd")
    macd_sig = latest.get("MACD_Signal") or latest.get("macd_signal")
    macd_hist = latest.get("MACD_Hist") or latest.get("macd_hist")
    macd_d = latest.get("macd_hist_delta")

    # Consolidation
    if cons < p["cons_min"]:
        return False, [f"consolidation {cons} < {p['cons_min']}"]

    # DI trend (backtest required one of trend anchors too — keep DI required here)
    if not (plus_di and minus_di and plus_di > minus_di):
        return False, ["no DI trend"]

    # ADX check (simple)
    if adx is None or float(adx) < p["adx_min"]:
        return False, [f"ADX weak ({adx})"]
    reasons.append(f"ADX {adx} ≥ {p['adx_min']}")

    # Momentum: backtest-style pass_count acceptance
    base_tests = [
        bool(
            macd is not None and macd_sig is not None and macd > macd_sig
        ),  # MACD cross
        bool(macd_hist is not None and macd_hist > 0),  # hist > 0
        bool(macd_d is not None and macd_d >= -0.02),  # hist improving (loose)
    ]
    pass_count = sum(base_tests)
    try:
        rsi_val = float(rsi) if rsi is not None else None
    except Exception:
        rsi_val = None

    strong_rsi = rsi_val is not None and rsi_val >= (p["rsi_min"] + 6)

    if not (
        pass_count >= 2
        or (pass_count >= 1 and strong_rsi)
        or (
            macd_hist is not None
            and macd_hist >= -0.05
            and rsi_val is not None
            and rsi_val >= p["rsi_min"]
        )
    ):
        return False, [f"momentum weak (pass_count={pass_count}, rsi={rsi_val})"]

    if rsi_val is None or rsi_val < p["rsi_min"]:
        return False, [f"RSI weak ({rsi})"]
    reasons.append(f"RSI {rsi} ≥ {p['rsi_min']}")

    # ETP check
    if etp is None or float(etp) < p["etp_min"]:
        return False, [f"ETP weak ({etp})"]
    reasons.append(f"ETP {etp}% ≥ {p['etp_min']}%")

    return True, reasons


def _last_ohlcv_from_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    ts = row.get("Timestamp") if "Timestamp" in row else None
    return {
        "time": str(ts) if ts is not None else None,
        "open": safe_number(row.get("Open") or row.get("open")),
        "high": safe_number(row.get("High") or row.get("high")),
        "low": safe_number(row.get("Low") or row.get("low")),
        "close": safe_number(row.get("Close") or row.get("close")),
        "volume": safe_number(row.get("Volume") or row.get("volume")),
    }


def _enforce_schema_defaults(cfg: dict) -> dict:
    """
    Ensure cfg has all expected keys (TF0..TF8 support/resistance/entry/target/stoploss/
    respected_S/respected_R, entry_target_pct, consolidation_score).
    Fill with None/0 where missing.
    """

    TF_RANGE = range(0, 9)  # TF0..TF8

    for i in TF_RANGE:
        suffix = "" if i == 0 else str(i)

        # SR + levels
        for k in ["support", "resistance", "entry", "target", "stoploss"]:
            key = f"{k}{suffix}"
            if key not in cfg:
                cfg[key] = None

        # respected counts
        for k in ["respected_S", "respected_R"]:
            key = f"{k}{suffix}"
            if key not in cfg:
                cfg[key] = 0

        # pct & consolidation
        pct_key = f"entry_target_pct{suffix}"
        cons_key = f"consolidation_score{suffix}"
        if pct_key not in cfg:
            cfg[pct_key] = 0.0
        if cons_key not in cfg:
            cfg[cons_key] = 0.0

    # signal
    if "signal" not in cfg or not cfg["signal"]:
        cfg["signal"] = "No Action"

    # reason
    if "reason" not in cfg or not isinstance(cfg["reason"], list) or not cfg["reason"]:
        cfg["reason"] = ["No bullish setup"]

    # last_updated
    if "last_updated" not in cfg:
        cfg["last_updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return cfg


def basic_forecast_update(
    stock_cfg: dict,
    hist_root: str = "historical_data_candles",
    verbose: bool = True,
) -> dict:
    """
    Updated basic forecast update:
    - Fetches recent Redis indicators for all TFs (1min → 1month).
    - Merges with historical CSV for same TFs (not just daily).
    - Computes SR + entry/target/stoploss per TF.
    - Ensures schema completeness with defaults if missing.
    """
    import datetime as dt
    import pandas as pd

    try:
        stock_code = stock_cfg["stock_code"]
        instrument_token = stock_cfg["instrument_token"]
    except KeyError:
        if verbose:
            print("Missing stock_code/instrument_token in stock_cfg")
        return stock_cfg

    # --- 1) Timeframes to monitor ---------------------------------------------

    MONITOR_TFS = [
        "1min",
        "5min",
        "15min",
        "30min",
        "45min",
        "1hour",
        "4hour",
        "1day",  # ✅ Added
        "1month",  # ✅ Added
    ]

    tf_dfs = {}
    for tf in MONITOR_TFS:
        # --- Load Redis recent (newest-first via redis_util helper) ---
        df_recent = _tf_recent_from_redis(stock_code, tf=tf, n=3000)

        if df_recent is None or df_recent.empty:
            if verbose:
                print(f"[ℹ️] {stock_code} {tf}: no usable data in Redis")
            tf_dfs[tf] = pd.DataFrame()
        else:
            # _tf_recent_from_redis already returns chronological DataFrame
            tf_dfs[tf] = df_recent
            if verbose:
                print(
                    f"[DEBUG] {stock_code} {tf}: loaded {len(df_recent)} candles from Redis"
                )

    # # --- 2) Use 1min as primary reference -------------------------------------
    # df_recent = tf_dfs.get("1min", pd.DataFrame())
    # if df_recent.empty:
    #     if verbose:
    #         print(f"No usable data for {stock_code} (1min empty)")
    #     return stock_cfg

    ohlcv = _last_ohlcv_from_df(df_recent)

    # --- 2) Use lowest available TF as reference ---
    df_recent = None
    for candidate_tf in ["5min", "30min", "1hour"]:
        if not tf_dfs.get(candidate_tf, pd.DataFrame()).empty:
            df_recent = tf_dfs[candidate_tf]
            break

    if df_recent is None or df_recent.empty:
        if verbose:
            print(
                f"[⚠️] No usable data for {stock_code} in key timeframes (5/30/60min)."
            )
        return stock_cfg

    # --- 3) Multi-timeframe SR + entry/target/SL ------------------------------
    multi_tf = build_all_timeframe_levels(stock_code, df_dict=tf_dfs, verbose=verbose)
    if not multi_tf:
        if verbose:
            print(f"[⚠️] {stock_code}: build_all_timeframe_levels returned empty")
        return stock_cfg

    # --- 4) Volume threshold (based on 1min recent) ---------------------------
    vol_series = _get(df_recent, "Volume", "volume", default=0)
    volume_threshold = (
        float(vol_series.tail(50).mean() * 1.5) if len(vol_series) else 0.0
    )

    # --- 5) Signal decision (multi-TF) ----------------------------------------
    signals = {}
    reasons_map = {}

    tf_map = {
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

    for idx, tf in tf_map.items():
        df_tf = tf_dfs.get(tf)
        if df_tf is None or df_tf.empty:
            continue

        latest = df_tf.iloc[-1]

        sr = {
            "support": multi_tf.get(f"support{idx}" if idx > 0 else "support"),
            "resistance": multi_tf.get(f"resistance{idx}" if idx > 0 else "resistance"),
        }
        etp = multi_tf.get(f"entry_target_pct{idx}" if idx > 0 else "entry_target_pct")

        is_buy, reasons = is_buy_signal_forecaster(latest, tf, sr, etp)

        sig_key = "signal" if idx == 0 else f"signal{idx}"
        reason_key = "reason" if idx == 0 else f"reason{idx}"

        signals[sig_key] = "BUY" if is_buy else "No Action"
        signals[reason_key] = reasons if reasons else [f"No bullish setup ({tf})"]
        # reasons_map[sig_key] = reasons if reasons else [f"No bullish setup ({tf})"]

    # --- 6) Final assembly ----------------------------------------------------
    new_cfg = {
        "stock_code": stock_code,
        "instrument_token": instrument_token,
        "ohlcv": ohlcv,
        "forecast": "basic_algo",
        "last_updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "volume_threshold": volume_threshold,
    }
    new_cfg.update(multi_tf)
    new_cfg.update(signals)
    # new_cfg.update(reasons_map)

    # --- 7) Validate & enforce schema -----------------------------------------
    new_cfg = _enforce_schema_defaults(new_cfg)

    ok, missing = _all_tf_complete(new_cfg)
    if not ok and verbose:
        print(f"Incomplete forecast for {stock_code}, missing: {missing}")

    # --- 8) Persist BUY signals to Redis (so server + monitor can react instantly)
    try:
        from redis_util import get_redis, save_buy_signal_to_redis

        r = None
        try:
            r = get_redis()
        except Exception:
            r = None

        if r is not None:
            # tf_map here: idx->tf. We'll check the TFs you care about (5min,30min,1hour)
            TF_INDEX_MAP = {
                0: "1min",
                1: "5min",
                2: "15min",
                3: "30min",
                4: "45min",
                5: "1hour",
                6: "4hour",
                7: "1day",  # ✅ Added
                8: "1month",  # ✅ Added
            }

            for idx, tf in TF_INDEX_MAP.items():
                sig_key = "signal" if idx == 0 else f"signal{idx}"
                sig_val = new_cfg.get(sig_key)
                if sig_val == "BUY":
                    # prepare compact payload
                    payload = {
                        "stock_code": stock_code,
                        "timeframe": tf,
                        "tf_index": idx,
                        "signal": "BUY",
                        "entry": new_cfg.get(f"entry{idx}" if idx != 0 else "entry"),
                        "target": new_cfg.get(f"target{idx}" if idx != 0 else "target"),
                        "stoploss": new_cfg.get(
                            f"stoploss{idx}" if idx != 0 else "stoploss"
                        ),
                        "support": new_cfg.get(
                            f"support{idx}" if idx != 0 else "support"
                        ),
                        "resistance": new_cfg.get(
                            f"resistance{idx}" if idx != 0 else "resistance"
                        ),
                        "entry_target_pct": new_cfg.get(
                            f"entry_target_pct{idx}" if idx != 0 else "entry_target_pct"
                        ),
                        "last_updated": new_cfg.get("last_updated"),
                    }
                    try:
                        save_buy_signal_to_redis(r, stock_code, tf, payload)
                    except Exception as e_save:
                        # do not fail forecast due to Redis error
                        if verbose:
                            print(
                                f"[⚠️] Failed save BUY signal to Redis for {stock_code} {tf}: {e_save}"
                            )
    except Exception:
        # be silent on import errors — don't break the forecast
        pass

    return new_cfg
