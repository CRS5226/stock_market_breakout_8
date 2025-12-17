#!/usr/bin/env python3
# pattern_narrator.py

"""
pattern_narrator.py — Advanced per-pattern multi-timeframe narrator with TA-Lib & targeted plots
Each detected pattern per timeframe is saved as its own annotated chart.
"""

import os, json
import pandas as pd
import talib
import mplfinance as mpf
import matplotlib.pyplot as plt
from redis_util import get_redis, load_from_redis
from datetime import datetime
import pytz

INDIA_TZ = pytz.timezone("Asia/Kolkata")
TIMEFRAMES = ["1min", "5min", "15min", "30min", "45min", "1hour", "4hour"]
# TIMEFRAMES = ["30min"]
BASE_DIR = "plot_results"
CANDLE_LIMIT = 10


# ------------------------------------------------------------
# Indicator descriptors
# ------------------------------------------------------------
def describe_rsi(df):
    if "RSI14" not in df.columns or len(df) < 3:
        return None, {}
    rsi_prev, rsi_now = df["RSI14"].iloc[-4], df["RSI14"].iloc[-1]
    trend = "rising" if rsi_now > rsi_prev else "falling"
    return f"RSI {trend} from {rsi_prev:.1f} → {rsi_now:.1f}", {
        "rsi_start": rsi_prev,
        "rsi_end": rsi_now,
    }


def describe_macd(df):
    if not all(x in df.columns for x in ["MACD", "MACD_Signal", "MACD_Hist"]):
        return None, {}
    macd_prev, macd_now = df["MACD"].iloc[-4], df["MACD"].iloc[-1]
    sig_prev, sig_now = df["MACD_Signal"].iloc[-4], df["MACD_Signal"].iloc[-1]
    hist_prev, hist_now = df["MACD_Hist"].iloc[-4], df["MACD_Hist"].iloc[-1]
    trend = "rising" if hist_now > hist_prev else "falling"
    return (
        f"MACD {trend} ({macd_prev:.3f}→{macd_now:.3f}), Signal ({sig_prev:.3f}→{sig_now:.3f}), Hist {hist_prev:.3f}→{hist_now:.3f}",
        {"macd_start": macd_prev, "macd_end": macd_now, "macd_hist_trend": trend},
    )


def describe_adx(df):
    if "ADX" not in df.columns or len(df) < 3:
        return None, {}
    adx_prev, adx_now = df["ADX"].iloc[-4], df["ADX"].iloc[-1]
    trend = "strengthening" if adx_now > adx_prev else "weakening"
    return f"ADX {trend} from {adx_prev:.2f} → {adx_now:.2f}", {
        "adx_start": adx_prev,
        "adx_end": adx_now,
    }


# ------------------------------------------------------------
# TA-Lib pattern auto-discovery — dynamically load all 61 patterns
# ------------------------------------------------------------
def get_all_talib_patterns():
    """Return dict of all available TA-Lib candlestick pattern functions."""
    pattern_dict = {}
    for fn in dir(talib):
        if fn.startswith("CDL"):
            pattern_dict[fn] = fn[3:].replace("_", " ").title()
    return pattern_dict


TA_PATTERNS = get_all_talib_patterns()


def detect_patterns(df):
    """
    Detect all TA-Lib candlestick patterns and return with OHLCV details.
    Each pattern now uses its correct lookback candle count based on TA-Lib definition.
    Adds 'best_tf' and 'strategy' fields for actionable context.
    """
    o, h, l, c = (
        df["Open"].values,
        df["High"].values,
        df["Low"].values,
        df["Close"].values,
    )
    results = []

    # --- Lookback definitions ---
    PATTERN_LOOKBACKS = {
        "CDLDOJI": 1,
        "CDLDRAGONFLYDOJI": 1,
        "CDLGRAVESTONEDOJI": 1,
        "CDLHAMMER": 1,
        "CDLHANGINGMAN": 1,
        "CDLINVERTEDHAMMER": 1,
        "CDLSHOOTINGSTAR": 1,
        "CDLLONGLINE": 1,
        "CDLSHORTLINE": 1,
        "CDLMARUBOZU": 1,
        "CDLSPINNINGTOP": 1,
        "CDLHIGHWAVE": 1,
        "CDLENGULFING": 2,
        "CDLHARAMI": 2,
        "CDLHARAMICROSS": 2,
        "CDLPIERCING": 2,
        "CDLKICKING": 2,
        "CDLKICKINGBYLENGTH": 2,
        "CDLMATCHINGLOW": 2,
        "CDLLADDERBOTTOM": 2,
        "CDLCLOSINGMARUBOZU": 2,
        "CDLBELTHOLD": 2,
        "CDLHOMINGPIGEON": 2,
        "CDLMORNINGDOJISTAR": 3,
        "CDLEVENINGDOJISTAR": 3,
        "CDLMORNINGSTAR": 3,
        "CDLEVENINGSTAR": 3,
        "CDLHIKKAKE": 3,
        "CDLHIKKAKEMOD": 3,
        "CDLTRISTAR": 3,
        "CDL3BLACKCROWS": 3,
        "CDL3INSIDE": 3,
        "CDL3LINESTRIKE": 3,
        "CDL3OUTSIDE": 3,
        "CDL3WHITESOLDIERS": 3,
        "CDLSTICKSANDWICH": 3,
        "CDLXSIDEGAP3METHODS": 3,
        "CDLUPSIDEGAP2CROWS": 3,
        "CDLDOWNSIDEGAP3METHODS": 3,
        "CDL3STARSINSOUTH": 3,
        "CDLUNIQUE3RIVER": 3,
        "CDLRICKSHAWMAN": 1,
        "CDLSEPARATINGLINES": 2,
        "CDLCOUNTERATTACK": 2,
        "CDLCONCEALBABYSWALL": 4,
        "CDLSTALLEDPATTERN": 3,
        "CDLLONGLEGGEDDOJI": 1,
    }

    # --- Strategy mapping for practical trading use ---
    PATTERN_STRATEGIES = {
        "CDLENGULFING": {
            "best_tf": "30min–1H",
            "strategy": "Use with MACD crossover for trend continuation",
        },
        "CDLHAMMER": {
            "best_tf": "30min–45min",
            "strategy": "Use with RSI <30 and ADX >25 for reversal entry",
        },
        "CDLHARAMI": {
            "best_tf": "1H–4H",
            "strategy": "Combine with ADX weakening for early trend reversal",
        },
        "CDLMARUBOZU": {
            "best_tf": "30min",
            "strategy": "Confirm with EMA breakout and volume spike",
        },
        "CDL3LINESTRIKE": {
            "best_tf": "45min",
            "strategy": "Trade continuation post-trend with MACD confirmation",
        },
        "CDL3BLACKCROWS": {
            "best_tf": "1H–1D",
            "strategy": "Bearish continuation confirmation in downtrends",
        },
        "CDL3WHITESOLDIERS": {
            "best_tf": "1H–1D",
            "strategy": "Bullish continuation confirmation in uptrends",
        },
        "CDLMORNINGSTAR": {
            "best_tf": "1D",
            "strategy": "Classic bottom reversal — confirm with RSI divergence",
        },
        "CDLEVENINGSTAR": {
            "best_tf": "1D",
            "strategy": "Classic top reversal — confirm with volume spike",
        },
        "CDLDOJI": {
            "best_tf": "30min–1H",
            "strategy": "Use as indecision marker — wait for next breakout candle",
        },
        "CDLSHOOTINGSTAR": {
            "best_tf": "30min–1H",
            "strategy": "Use at resistance with RSI >70 for short entries",
        },
        "CDLHIKKAKE": {
            "best_tf": "45min",
            "strategy": "Watch for failed breakout reversals",
        },
        "CDLBELTHOLD": {
            "best_tf": "30min–1H",
            "strategy": "Strong single-bar momentum candle — continuation likely",
        },
    }

    # --- Loop through all TA-Lib patterns ---
    for func, pretty_name in TA_PATTERNS.items():
        try:
            arr = getattr(talib, func)(o, h, l, c)
        except Exception as e:
            print(f"⚠️ Error in {func}: {e}")
            continue

        idx = (arr != 0).nonzero()[0]
        if len(idx) == 0:
            continue

        lookback = PATTERN_LOOKBACKS.get(func, 3)
        for i in idx:
            direction = "Bullish" if arr[i] > 0 else "Bearish"
            start_idx = max(0, i - lookback)
            end_idx = i

            slice_df = df.iloc[start_idx : end_idx + 1][
                ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            ].copy()

            ohlcv_list = [
                {
                    "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": float(row.get("Volume", 0.0)),
                }
                for _, row in slice_df.iterrows()
            ]

            start_ts = df["Timestamp"].iloc[start_idx]
            end_ts = df["Timestamp"].iloc[end_idx]

            strat_info = PATTERN_STRATEGIES.get(
                func,
                {
                    "best_tf": "N/A",
                    "strategy": "General observation — confirmation required",
                },
            )

            results.append(
                {
                    "name": f"{direction} {pretty_name}",
                    "start_ts": start_ts.strftime("%Y-%m-%d %H:%M"),
                    "end_ts": end_ts.strftime("%Y-%m-%d %H:%M"),
                    "candles_used": len(slice_df),
                    "best_tf": strat_info["best_tf"],
                    "strategy": strat_info["strategy"],
                    "ohlcv_values": ohlcv_list,
                }
            )

    return results


# ------------------------------------------------------------
# Plot generator for a single pattern
# ------------------------------------------------------------


def analyze_tf(r, stock, tf, limit=CANDLE_LIMIT):
    df = load_from_redis(r, stock, tf, limit=limit)

    # 🧩 Basic diagnostics: how many candles loaded
    total_candles = len(df)
    print(f"📊 {stock} {tf}: loaded {total_candles} candles from Redis")

    if df.empty or total_candles < 5:
        print(f"⚠️ {stock} {tf}: insufficient data for analysis")
        return {
            "timeframe": tf,
            "pattern_text": "No data",
            "total_candles": total_candles,
        }

    patterns = detect_patterns(df)
    print(f"🔎 {stock} {tf}: detected {len(patterns)} raw patterns")

    # --- Build pattern statistics ---
    count_by_length = {}
    for p in patterns:
        n = p["candles_used"]
        count_by_length[n] = count_by_length.get(n, 0) + 1

    # Print pattern count by candle count category
    if count_by_length:
        stats_str = ", ".join(
            [f"{k}-candle: {v}" for k, v in sorted(count_by_length.items())]
        )
        print(f"📈 {stock} {tf} pattern counts → {stats_str}")
    else:
        print(f"❌ {stock} {tf}: no valid pattern signals detected")

    # --- Indicators ---
    rsi_text, rsi_meta = describe_rsi(df)
    macd_text, macd_meta = describe_macd(df)
    adx_text, adx_meta = describe_adx(df)
    indicators = {**rsi_meta, **macd_meta, **adx_meta}

    # --- Summary text ---
    pattern_summaries = [
        f"{p['name']} ({p['candles_used']} candles) {p['start_ts']} → {p['end_ts']}"
        for p in patterns
    ]
    indicator_texts = [x for x in [rsi_text, macd_text, adx_text] if x]
    summary = (
        "; ".join(pattern_summaries + indicator_texts)
        if pattern_summaries or indicator_texts
        else "No major pattern detected"
    )

    # No plotting yet
    plot_paths = []

    return {
        "timeframe": tf,
        "pattern_text": summary,
        "patterns": patterns,
        "pattern_counts": count_by_length,
        "indicators": indicators,
        "timestamp": df["Timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
        "total_candles": total_candles,
    }


# ------------------------------------------------------------
# Multi-timeframe
# ------------------------------------------------------------
def narrate_all_timeframes(stock):
    r = get_redis()
    results = [analyze_tf(r, stock, tf) for tf in TIMEFRAMES]
    return {"stock": stock, "results": results}


# ------------------------------------------------------------
# Exposed real-time function for external use
# ------------------------------------------------------------
def run_pattern_narration_realtime(stock_code, save_json=True):
    """
    Run real-time candlestick + indicator narration for a single stock.
    Saves to pattern_realtime_<stock>.json if save_json=True.
    Returns the narration result as a Python dict.
    """
    r = get_redis()
    print(f"🔍 Running pattern narrator in real-time for {stock_code}...")
    result = narrate_all_timeframes(stock_code)

    if save_json:
        out_path = f"./data/pattern_realtime_{stock_code}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Pattern narration saved to {out_path}")

    return result


# ------------------------------------------------------------
# Test runner
# ------------------------------------------------------------

if __name__ == "__main__":
    test_stocks = ["FORCEMOT"]
    all_results = []
    for s in test_stocks:
        res = narrate_all_timeframes(s)
        all_results.append(res)

    # ✅ Write UTF-8 encoded JSON
    with open("pattern_detailed_results2.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
