#!/usr/bin/env python3
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
# TIMEFRAMES = ["1min", "5min", "15min", "30min", "45min", "1hour", "4hour"]
TIMEFRAMES = ["30min"]
BASE_DIR = "plot_results"
CANDLE_LIMIT = 20


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
# TA-Lib patterns
# ------------------------------------------------------------
# TA_PATTERNS = {
#     "CDLENGULFING": "Engulfing",
#     "CDLHAMMER": "Hammer",
#     "CDLSHOOTINGSTAR": "Shooting Star",
#     "CDLDOJI": "Doji",
#     "CDLEVENINGSTAR": "Evening Star",
#     "CDLMORNINGSTAR": "Morning Star",
#     "CDLHARAMI": "Harami",
#     "CDLPIERCING": "Piercing",
# }


# def detect_patterns(df):
#     o, h, l, c = (
#         df["Open"].values,
#         df["High"].values,
#         df["Low"].values,
#         df["Close"].values,
#     )
#     results = []
#     for func, name in TA_PATTERNS.items():
#         arr = getattr(talib, func)(o, h, l, c)
#         idx = (arr != 0).nonzero()[0]
#         for i in idx:
#             direction = "Bullish" if arr[i] > 0 else "Bearish"
#             start_idx = max(0, i - 2)
#             end_idx = i

#             # ---- Slice the exact candles used ----
#             slice_df = df.iloc[start_idx : end_idx + 1][
#                 ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
#             ].copy()

#             # ---- Convert to readable dict ----
#             ohlcv_list = []
#             for _, row in slice_df.iterrows():
#                 ohlcv_list.append(
#                     {
#                         "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M"),
#                         "Open": float(row["Open"]),
#                         "High": float(row["High"]),
#                         "Low": float(row["Low"]),
#                         "Close": float(row["Close"]),
#                         "Volume": (
#                             float(row["Volume"]) if "Volume" in df.columns else 0.0
#                         ),
#                     }
#                 )

#             # ---- Final pattern entry ----
#             start_ts = df["Timestamp"].iloc[start_idx]
#             end_ts = df["Timestamp"].iloc[end_idx]
#             results.append(
#                 {
#                     "name": f"{direction} {name}",
#                     "start_ts": start_ts.strftime("%Y-%m-%d %H:%M"),
#                     "end_ts": end_ts.strftime("%Y-%m-%d %H:%M"),
#                     "candles_used": len(slice_df),
#                     "ohlcv_values": ohlcv_list,
#                 }
#             )

#     return results


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
    """Detect all TA-Lib candlestick patterns and return with OHLCV details."""
    o, h, l, c = (
        df["Open"].values,
        df["High"].values,
        df["Low"].values,
        df["Close"].values,
    )
    results = []

    for func, pretty_name in TA_PATTERNS.items():
        try:
            arr = getattr(talib, func)(o, h, l, c)
        except Exception as e:
            print(f"⚠️ Error in {func}: {e}")
            continue

        idx = (arr != 0).nonzero()[0]
        for i in idx:
            direction = "Bullish" if arr[i] > 0 else "Bearish"
            start_idx = max(0, i - 2)
            end_idx = i

            # Slice the exact candles used
            slice_df = df.iloc[start_idx : end_idx + 1][
                ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            ].copy()

            # Convert to readable dict
            ohlcv_list = []
            for _, row in slice_df.iterrows():
                ohlcv_list.append(
                    {
                        "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M"),
                        "Open": float(row["Open"]),
                        "High": float(row["High"]),
                        "Low": float(row["Low"]),
                        "Close": float(row["Close"]),
                        "Volume": float(row.get("Volume", 0.0)),
                    }
                )

            # Append final pattern entry
            start_ts = df["Timestamp"].iloc[start_idx]
            end_ts = df["Timestamp"].iloc[end_idx]
            results.append(
                {
                    "name": f"{direction} {pretty_name}",
                    "start_ts": start_ts.strftime("%Y-%m-%d %H:%M"),
                    "end_ts": end_ts.strftime("%Y-%m-%d %H:%M"),
                    "candles_used": len(slice_df),
                    "ohlcv_values": ohlcv_list,
                }
            )

    return results


# ------------------------------------------------------------
# Plot generator for a single pattern
# ------------------------------------------------------------
# def plot_pattern(df, stock, timeframe, pattern, indicators):
#     import matplotlib.dates as mdates
#     import numpy as np

#     pattern_name = pattern["name"].replace(" ", "_")
#     outdir = os.path.join(BASE_DIR, stock, timeframe)
#     os.makedirs(outdir, exist_ok=True)
#     path = os.path.join(outdir, f"{pattern_name}.png")

#     # ---- Prepare data ----
#     df = df.set_index("Timestamp")
#     start = pd.to_datetime(pattern["start_ts"])
#     end = pd.to_datetime(pattern["end_ts"])
#     if df.index.tz is not None:
#         start = (
#             start.tz_localize(df.index.tz)
#             if start.tzinfo is None
#             else start.tz_convert(df.index.tz)
#         )
#         end = (
#             end.tz_localize(df.index.tz)
#             if end.tzinfo is None
#             else end.tz_convert(df.index.tz)
#         )

#     # Focus around pattern ± 10 candles, not just minutes
#     start_i = df.index.get_indexer([start], method="nearest")[0]
#     end_i = df.index.get_indexer([end], method="nearest")[0]
#     window_start = max(0, start_i - 10)
#     window_end = min(len(df) - 1, end_i + 10)
#     df_slice = df.iloc[window_start : window_end + 1].copy()
#     if df_slice.empty:
#         df_slice = df.tail(50)

#     # ---- Dynamic zoom: center around pattern candles ----
#     local_high = df_slice["High"].max()
#     local_low = df_slice["Low"].min()
#     pad = max((local_high - local_low) * 0.05, 0.002 * local_high)
#     if local_high - local_low < 1e-3:  # extremely flat data fallback
#         local_high += 0.01 * local_high
#         local_low -= 0.01 * local_low
#     ylim = (local_low - pad, local_high + pad)

#     # ---- Determine indicators to show ----
#     show_macd = "macd_start" in indicators
#     show_rsi = "rsi_start" in indicators
#     show_adx = "adx_start" in indicators
#     nrows = 1 + sum([show_macd, show_rsi, show_adx])

#     # ---- Build figure ----
#     fig = mpf.figure(figsize=(13, 3.2 * nrows), style="charles", dpi=170)
#     axes = [fig.add_subplot(nrows, 1, 1)]
#     for i in range(2, nrows + 1):
#         axes.append(fig.add_subplot(nrows, 1, i, sharex=axes[0]))

#     # ---- Candlestick panel ----
#     mpf.plot(
#         df_slice,
#         ax=axes[0],
#         type="candle",
#         volume=False,
#         show_nontrading=False,
#         update_width_config={
#             "candle_linewidth": 1.6,
#             "candle_width": 1.0,
#             "ohlc_linewidth": 1.5,
#         },
#         tight_layout=False,
#     )

#     axes[0].set_ylim(*ylim)
#     axes[0].set_title(
#         f"{stock} {timeframe} — {pattern['name']}\n"
#         f"Detected {pattern['start_ts']} → {pattern['end_ts']} "
#         f"({pattern['candles_used']} candles)",
#         fontsize=10,
#         fontweight="bold",
#     )
#     axes[0].axvspan(start, end, color="orange", alpha=0.25, zorder=0)
#     axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

#     # ---- Indicators ----
#     idx = 1
#     if show_macd:
#         ax = axes[idx]
#         ax.plot(df_slice.index, df_slice["MACD"], label="MACD", color="blue", lw=1.4)
#         ax.plot(
#             df_slice.index, df_slice["MACD_Signal"], label="Signal", color="red", lw=1.1
#         )
#         ax.bar(
#             df_slice.index, df_slice["MACD_Hist"], color="gray", alpha=0.5, width=0.02
#         )
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("MACD")
#         idx += 1

#     if show_rsi:
#         ax = axes[idx]
#         ax.plot(
#             df_slice.index, df_slice["RSI14"], color="purple", lw=1.4, label="RSI(14)"
#         )
#         ax.axhline(70, color="red", linestyle="--", lw=0.8)
#         ax.axhline(30, color="green", linestyle="--", lw=0.8)
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("RSI")
#         idx += 1

#     if show_adx:
#         ax = axes[idx]
#         ax.plot(df_slice.index, df_slice["ADX"], color="teal", lw=1.3, label="ADX")
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("ADX")

#     # Ensure candle plot has a visible height
#     axes[0].set_ylim(*ylim)
#     y_range = ylim[1] - ylim[0]
#     if y_range < 0.005 * df_slice["Close"].mean():
#         mid = df_slice["Close"].iloc[-1]
#         axes[0].set_ylim(mid - 0.005 * mid, mid + 0.005 * mid)

#     fig.tight_layout(pad=1.0)
#     fig.savefig(path, bbox_inches="tight")
#     plt.close(fig)
#     return path


# def plot_pattern(df, stock, timeframe, pattern, indicators):
#     import matplotlib.dates as mdates
#     import numpy as np
#     from datetime import timedelta

#     pattern_name = pattern["name"].replace(" ", "_")
#     outdir = os.path.join(BASE_DIR, stock, timeframe)
#     os.makedirs(outdir, exist_ok=True)
#     full_path = os.path.join(outdir, f"{pattern_name}_full.png")
#     zoom_path = os.path.join(outdir, f"{pattern_name}_candles.png")

#     # ---- Convert to timezone-aware IST ----
#     if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
#         df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
#     df = df.dropna(subset=["Timestamp"])
#     if df["Timestamp"].dt.tz is None:
#         df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC").dt.tz_convert(INDIA_TZ)
#     else:
#         df["Timestamp"] = df["Timestamp"].dt.tz_convert(INDIA_TZ)

#     # ---- Filter market hours only ----
#     market_open = datetime.strptime("09:15", "%H:%M").time()
#     market_close = datetime.strptime("15:30", "%H:%M").time()
#     df = df[
#         (df["Timestamp"].dt.time >= market_open)
#         & (df["Timestamp"].dt.time <= market_close)
#         & (df["Timestamp"].dt.dayofweek < 5)
#     ].copy()
#     if df.empty:
#         print(f"⚠️ {stock} {timeframe}: No data within market hours.")
#         return {"full_plot": None, "zoom_plot": None}

#     df = df.sort_values("Timestamp").reset_index(drop=True)

#     # ---- Reindex to continuous candle spacing ----
#     candle_minutes = (
#         30
#         if "30" in timeframe
#         else (
#             15
#             if "15" in timeframe
#             else 5 if "5" in timeframe else 1 if "1min" in timeframe else 60
#         )
#     )
#     base_time = pd.Timestamp(df["Timestamp"].iloc[0]).tz_localize(None)
#     fake_times = [
#         base_time + timedelta(minutes=candle_minutes * i) for i in range(len(df))
#     ]
#     df.index = pd.DatetimeIndex(fake_times, tz=INDIA_TZ)
#     df.index.name = "Datetime"

#     # ---- Locate pattern region ----
#     start_real = pd.to_datetime(pattern["start_ts"]).tz_localize(INDIA_TZ)
#     end_real = pd.to_datetime(pattern["end_ts"]).tz_localize(INDIA_TZ)
#     nearest_start = df["Timestamp"].sub(start_real).abs().idxmin()
#     nearest_end = df["Timestamp"].sub(end_real).abs().idxmin()

#     start_i = df.index.get_loc(nearest_start)
#     end_i = df.index.get_loc(nearest_end)
#     window_start = max(0, start_i - 10)
#     window_end = min(len(df) - 1, end_i + 10)
#     df_slice = df.iloc[window_start : window_end + 1].copy()
#     if df_slice.empty:
#         df_slice = df.tail(50)

#     # ---- Dynamic Y range ----
#     local_high = df_slice["High"].max()
#     local_low = df_slice["Low"].min()
#     pad = max((local_high - local_low) * 0.05, 0.002 * local_high)
#     ylim = (local_low - pad, local_high + pad)

#     # ---- Indicators ----
#     show_macd = "macd_start" in indicators
#     show_rsi = "rsi_start" in indicators
#     show_adx = "adx_start" in indicators
#     nrows = 1 + sum([show_macd, show_rsi, show_adx])

#     # ============================================================
#     # FULL CHART (with yellow highlight)
#     # ============================================================
#     fig = mpf.figure(figsize=(13, 3.2 * nrows), style="charles", dpi=170)
#     axes = [fig.add_subplot(nrows, 1, 1)]
#     for i in range(2, nrows + 1):
#         axes.append(fig.add_subplot(nrows, 1, i, sharex=axes[0]))

#     mpf.plot(
#         df_slice,
#         ax=axes[0],
#         type="candle",
#         volume=False,
#         show_nontrading=False,
#         datetime_format="%H:%M",
#         xrotation=15,
#         update_width_config={"candle_linewidth": 1.5, "candle_width": 0.7},
#     )

#     axes[0].set_ylim(*ylim)
#     axes[0].set_title(
#         f"{stock} {timeframe} — {pattern['name']} (IST Market Hours)\n"
#         f"{pattern['start_ts']} → {pattern['end_ts']}",
#         fontsize=10,
#         fontweight="bold",
#     )

#     # ✅ Keep highlight for full plot only
#     try:
#         axes[0].axvspan(
#             df_slice.index[df_slice.index.get_loc(nearest_start)],
#             df_slice.index[df_slice.index.get_loc(nearest_end)],
#             color="orange",
#             alpha=0.25,
#         )
#     except Exception:
#         pass

#     import matplotlib.dates as mdates

#     axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=INDIA_TZ))
#     axes[0].set_xlim(df_slice.index[0], df_slice.index[-1])

#     # ---- Indicators ----
#     idx = 1
#     if show_macd:
#         ax = axes[idx]
#         ax.plot(df_slice.index, df_slice["MACD"], color="blue", lw=1.4, label="MACD")
#         ax.plot(
#             df_slice.index, df_slice["MACD_Signal"], color="red", lw=1.1, label="Signal"
#         )
#         ax.bar(
#             df_slice.index, df_slice["MACD_Hist"], color="gray", alpha=0.4, width=0.01
#         )
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("MACD")
#         idx += 1

#     if show_rsi:
#         ax = axes[idx]
#         ax.plot(
#             df_slice.index, df_slice["RSI14"], color="purple", lw=1.4, label="RSI(14)"
#         )
#         ax.axhline(70, color="red", linestyle="--", lw=0.8)
#         ax.axhline(30, color="green", linestyle="--", lw=0.8)
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("RSI")
#         idx += 1

#     if show_adx:
#         ax = axes[idx]
#         ax.plot(df_slice.index, df_slice["ADX"], color="teal", lw=1.3, label="ADX")
#         ax.legend(loc="upper left", fontsize=8)
#         ax.set_ylabel("ADX")

#     fig.tight_layout(pad=1.0)
#     fig.savefig(full_path, bbox_inches="tight")
#     plt.close(fig)

#     # ============================================================
#     # ZOOM CHART — without highlight, safe empty handling
#     # ============================================================
#     # ============================================================
#     # ZOOM CHART — without highlight, safe empty handling
#     # ============================================================
#     zoom_df = df_slice.iloc[max(0, start_i - 3) : end_i + 4].copy()
#     if zoom_df.empty:
#         zoom_df = df_slice.tail(3).copy()

#     # Ensure valid OHLC numeric data
#     zoom_df = zoom_df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
#     for col in ["Open", "High", "Low", "Close"]:
#         zoom_df[col] = pd.to_numeric(zoom_df[col], errors="coerce")

#     # If all candles are flat → widen artificially
#     if abs(zoom_df["High"].max() - zoom_df["Low"].min()) < 1e-3:
#         mid = zoom_df["Close"].iloc[-1]
#         zoom_df["High"] = mid * 1.001
#         zoom_df["Low"] = mid * 0.999

#     # Ensure DatetimeIndex
#     if not isinstance(zoom_df.index, pd.DatetimeIndex):
#         zoom_df.index = pd.DatetimeIndex(zoom_df.index)

#     fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=170)
#     mpf.plot(
#         zoom_df,
#         ax=ax2,
#         type="candle",
#         style="charles",
#         volume=False,
#         show_nontrading=False,
#         datetime_format="%H:%M",
#         xrotation=25,
#         update_width_config={"candle_linewidth": 2.0, "candle_width": 0.9},
#     )
#     ax2.set_ylim(*ylim)
#     ax2.set_xlim(zoom_df.index.min(), zoom_df.index.max())
#     ax2.set_title(
#         f"{stock} {timeframe} — {pattern['name']} (Zoomed IST Candles)\n"
#         f"{pattern['start_ts']} → {pattern['end_ts']}",
#         fontsize=11,
#         fontweight="bold",
#     )
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=INDIA_TZ))
#     fig2.tight_layout(pad=1.0)
#     fig2.savefig(zoom_path, bbox_inches="tight")
#     plt.close(fig2)

#     return {"full_plot": full_path, "zoom_plot": zoom_path}


def plot_pattern(df, stock, timeframe, pattern, indicators):
    pass


# ------------------------------------------------------------
# Analyze one timeframe
# ------------------------------------------------------------
# def analyze_tf(r, stock, tf, limit=CANDLE_LIMIT):
#     df = load_from_redis(r, stock, tf, limit=limit)
#     if df.empty or len(df) < 20:
#         return {"timeframe": tf, "pattern_text": "No data"}

#     patterns = detect_patterns(df)
#     rsi_text, rsi_meta = describe_rsi(df)
#     macd_text, macd_meta = describe_macd(df)
#     adx_text, adx_meta = describe_adx(df)

#     indicators = {**rsi_meta, **macd_meta, **adx_meta}
#     indicator_texts = [x for x in [rsi_text, macd_text, adx_text] if x]
#     pattern_texts = [p["name"] for p in patterns]
#     summary = "; ".join(pattern_texts + indicator_texts) or "No major pattern detected"

#     # Plot for each pattern
#     plot_paths = []
#     for p in patterns:
#         p_path = plot_pattern(df.copy(), stock, tf, p, indicators)
#         plot_paths.append({"pattern": p["name"], "path": p_path})

#     return {
#         "timeframe": tf,
#         "pattern_text": summary,
#         "patterns": patterns,
#         "indicators": indicators,
#         "plot_paths": plot_paths,
#         "timestamp": df["Timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
#     }


def analyze_tf(r, stock, tf, limit=CANDLE_LIMIT):
    df = load_from_redis(r, stock, tf, limit=limit)
    if df.empty or len(df) < 20:
        return {"timeframe": tf, "pattern_text": "No data"}

    patterns = detect_patterns(df)
    rsi_text, rsi_meta = describe_rsi(df)
    macd_text, macd_meta = describe_macd(df)
    adx_text, adx_meta = describe_adx(df)
    indicators = {**rsi_meta, **macd_meta, **adx_meta}

    # --- Build readable summary per pattern ---
    pattern_summaries = []
    for p in patterns:
        pattern_summaries.append(
            f"{p['name']} detected using {p['candles_used']} candles from {p['start_ts']} to {p['end_ts']}"
        )

    indicator_texts = [x for x in [rsi_text, macd_text, adx_text] if x]
    summary = (
        "; ".join(pattern_summaries + indicator_texts) or "No major pattern detected"
    )

    # --- Generate individual plots ---
    plot_paths = []
    for p in patterns:
        p_path = plot_pattern(df.copy(), stock, tf, p, indicators)
        plot_paths.append({"pattern": p["name"], "path": p_path})

    return {
        "timeframe": tf,
        "pattern_text": summary,
        "patterns": patterns,
        "indicators": indicators,
        "plot_paths": plot_paths,
        "timestamp": df["Timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
    }


# ------------------------------------------------------------
# Multi-timeframe
# ------------------------------------------------------------
def narrate_all_timeframes(stock):
    r = get_redis()
    results = [analyze_tf(r, stock, tf) for tf in TIMEFRAMES]
    return {"stock": stock, "results": results}


# ------------------------------------------------------------
# Test runner
# ------------------------------------------------------------
# if __name__ == "__main__":
#     test_stocks = ["FORCEMOT"]
#     all_results = []
#     for s in test_stocks:
#         res = narrate_all_timeframes(s)
#         all_results.append(res)
#         # print(json.dumps(res, indent=2))
#     with open("pattern_detailed_results.json", "w") as f:
#         # json.dump(all_results, f, indent=2)
#         json.dump(all_results, f, indent=2, ensure_ascii=False)
#     print("\n✅ Plots saved in plot_results/<stock>/<timeframe>/<pattern>.png")

if __name__ == "__main__":
    test_stocks = ["FORCEMOT"]
    all_results = []
    for s in test_stocks:
        res = narrate_all_timeframes(s)
        all_results.append(res)

    # ✅ Write UTF-8 encoded JSON
    with open("pattern_detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n✅ Plots saved in plot_results/<stock>/<timeframe>/<pattern>.png")
    print("✅ JSON saved as UTF-8 (no more \\u escapes or encoding errors)")
