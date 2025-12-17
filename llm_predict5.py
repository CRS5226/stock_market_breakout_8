#!/usr/bin/env python3
# llm_predict4.py — Simplified GPT Forecasting Engine (with full news sentiment integration)
# --------------------------------------------------------
# Fetches latest candles from Redis, merges all stock news sentiments,
# and builds a context-aware GPT prompt combining price + news data.

import os
import json
import re
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI

# === Local imports ===
from redis_util import get_redis, get_recent_candles_tf

# === ENV ===
load_dotenv()

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4.1-mini")
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "gpt-4.1-nano")
USE_PARALLEL_ROUTING = os.getenv("USE_PARALLEL_ROUTING", "True").lower() in (
    "1",
    "true",
    "yes",
)

TARGET_DIFF_THRESHOLD = float(os.getenv("TARGET_DIFF_THRESHOLD", 1.0))
STOPLOSS_DIFF_THRESHOLD = float(os.getenv("STOPLOSS_DIFF_THRESHOLD", 0.3))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[⚠️] OPENAI_API_KEY not set. GPT calls will fail.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -----------------------------------------------------------
# Model router
# -----------------------------------------------------------
def route_model(stock_code: str):
    """Simple alternating router for load balancing."""
    if not USE_PARALLEL_ROUTING:
        return PRIMARY_MODEL
    return PRIMARY_MODEL if hash(stock_code) % 2 == 0 else SECONDARY_MODEL


# -----------------------------------------------------------
# Safe JSON Parser
# -----------------------------------------------------------
def _safe_json_parse(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                snippet = (
                    match.group(0)
                    .replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null")
                )
                return json.loads(snippet)
        except Exception:
            pass
    print(f"[⚠️ JSON parse failed]: {raw[:200]}...")
    return None


# -----------------------------------------------------------
# Utility enrichments
# -----------------------------------------------------------
def count_respects(df, level, mode="support"):
    if level is None or df is None or df.empty:
        return 0
    try:
        highs = pd.to_numeric(df["High"], errors="coerce")
        lows = pd.to_numeric(df["Low"], errors="coerce")
        closes = pd.to_numeric(df["Close"], errors="coerce")
        level_f = float(level)
        if mode == "support":
            hits = ((lows <= level_f * 1.002) & (closes >= level_f)).sum()
        else:
            hits = ((highs >= level_f * 0.998) & (closes <= level_f)).sum()
        return int(hits)
    except Exception:
        return 0


def compute_consolidation(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None
    try:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 20:
            return None
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        last_idx = std.last_valid_index()
        if last_idx is None:
            return None
        bb_width = (std.loc[last_idx] / ma.loc[last_idx]) * 100.0
        score = 100.0 - bb_width
        return round(max(0.0, min(100.0, score)), 2)
    except Exception:
        return None


def check_all_tf_complete(result_dict):
    issues = []
    for idx in range(1, 8):
        try:
            sl = float(result_dict.get(f"stoploss{idx}") or 0)
            s = float(result_dict.get(f"support{idx}") or 0)
            e = float(result_dict.get(f"entry{idx}") or 0)
            t = float(result_dict.get(f"target{idx}") or 0)
            r = float(result_dict.get(f"resistance{idx}") or 0)
            if not (sl < s < e < t < r):
                issues.append(f"TF{idx}: invalid order")
        except Exception:
            issues.append(f"TF{idx}: missing/invalid values")
    return len(issues) == 0, issues


def get_live_price_from_redis(r, stock_code):
    """Fetch the most recent 1min candle's close price from Redis."""
    try:
        key = f"MARKETDATA:{stock_code}:1min"
        candles = r.hgetall(key)
        if not candles:
            return None, None
        records = [json.loads(v) for v in candles.values()]
        if not records:
            return None, None
        records.sort(key=lambda c: c.get("Timestamp", ""))
        latest = records[-1]
        price = float(latest.get("Close") or 0)
        ts = latest.get("Timestamp")
        return price, ts
    except Exception as e:
        print(f"[⚠️ Live price fetch failed {stock_code}: {e}]")
        return None, None


# -----------------------------------------------------------
# 🔍 Load full news sentiment map
# -----------------------------------------------------------
def get_combined_news_sentiment():
    """
    Loads all stock news sentiments from news/data/news.json
    and combines them into a flat map { "RELIANCE.NS": "BUY", "ONGC.NS": "No Action" }
    Returns (combined_dict, last_updated)
    """
    sentiment_path = os.path.join("news", "data", "news.json")
    if not os.path.exists(sentiment_path):
        print("[⚠️] No news sentiment file found")
        return {}, "N/A"

    try:
        with open(sentiment_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        combined = {}
        for art in articles:
            res = art.get("result", {})
            if isinstance(res, dict):
                combined.update(res)
        last_updated = data.get("last_updated", "N/A")
        return combined, last_updated
    except Exception as e:
        print(f"[⚠️] Failed to load combined news sentiment: {e}")
        return {}, "N/A"


# -----------------------------------------------------------
# 🕯️ Load latest pattern narrator summary (from pattern_narrator.py output)
# -----------------------------------------------------------
def get_latest_pattern_summary(
    stock_code, allowed_tfs=("30min", "45min", "1hour", "4hour")
):
    """
    Loads the latest pattern narrator JSON for a given stock.
    Filters only selected timeframes (default: 30min, 45min).
    Returns compact text summary for GPT context.
    """
    file_path = f"./pattern_narrator_test/data/pattern_realtime_{stock_code}.json"
    if not os.path.exists(file_path):
        print(f"[⚠️] No pattern file found for {stock_code}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        summaries = []

        for r in results:
            tf = r.get("timeframe")
            if tf not in allowed_tfs:
                continue
            ptext = r.get("pattern_text", "")
            # patterns = r.get("patterns", [])
            # for p in patterns:
            #     name = p.get("name", "")
            #     strat = p.get("strategy", "")
            #     best_tf = p.get("best_tf", "")
            #     if strat and best_tf:
            #         summaries.append(f"{name} [{best_tf}] → {strat}")
            #     elif name:
            #         summaries.append(name)
            if ptext:
                summaries.append(ptext)

        if not summaries:
            return None

        compact_summary = "; ".join(summaries[:5])  # limit to 10 items for brevity
        print(f"[🕯️ Pattern Summary Loaded] {stock_code}: {len(summaries)} entries")
        return compact_summary

    except Exception as e:
        print(f"[⚠️] Failed to load pattern summary for {stock_code}: {e}")
        return None


# -----------------------------------------------------------
# Main Forecast Function
# -----------------------------------------------------------
def forecast_config_update(
    stock_cfg: dict, verbose: bool = True, selected_tf: str = None
):
    try:
        stock_code = stock_cfg["stock_code"]
        instrument_token = stock_cfg["instrument_token"]
    except KeyError:
        print("[❌] Missing stock_code/instrument_token in config")
        return stock_cfg

    r = get_redis()
    tz = ZoneInfo("Asia/Kolkata")

    # --- Fetch live price from Redis ---
    live_price, live_ts = get_live_price_from_redis(r, stock_code)
    if live_price:
        print(f"[💹 Live Price] {stock_code}: {live_price} @ {live_ts}")
    else:
        print(f"[⚠️ No live price found] {stock_code}")

    # --- Load all news sentiments ---
    combined_sentiments, sentiment_ts = get_combined_news_sentiment()
    stock_sentiment = combined_sentiments.get(stock_code, "No Action")
    print(
        f"[📰 News Sentiment] {stock_code}: {stock_sentiment} (updated {sentiment_ts})"
    )

    # --- Load pattern narrator insights ---
    pattern_summary = get_latest_pattern_summary(stock_code)
    if pattern_summary:
        print(f"[🕯️ Chart Patterns] {stock_code}: using recent pattern insights.")
    else:
        pattern_summary = "No significant recent patterns detected."

    # --- Timeframes to analyze ---
    MONITOR_TFS = ["5min", "15min", "30min", "45min", "1hour", "4hour"]
    if selected_tf:
        MONITOR_TFS = [selected_tf]

    # --- Fetch recent candles ---
    tf_candles = {}
    for tf in MONITOR_TFS:
        try:
            data = get_recent_candles_tf(r, stock_code, tf, n=2)
            if not data:
                continue
            df = pd.DataFrame(data)
            if df.empty or "Timestamp" not in df.columns:
                continue
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = (
                df.dropna(subset=["Timestamp"])
                .sort_values("Timestamp")
                .reset_index(drop=True)
            )
            tf_candles[tf] = df.tail(2)
            if verbose:
                print(f"[✅ TF fetch] {stock_code} {tf}: {len(df)} rows")
        except Exception as e:
            print(f"[⚠️ Redis fetch failed {stock_code} {tf}: {e}]")

    if not tf_candles:
        print(f"[❌] No candle data for {stock_code}")
        return stock_cfg

    def summarize_df(df):
        """Keep only the most relevant indicators + OHLCV."""
        df = df.copy().round(3)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        drop_cols = [c for c in df.columns if c.lower().startswith("tr_tmp")]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Select important indicators only
        keep_cols = [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "RSI14",
            "MACD_Hist",
            "ADX",
            "ATR14",
            "vol_z",
            "VWAP",
            "ema20_slope_bps",
            "vwap_diff_bps",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df.to_dict(orient="records")

    data_summary = {tf: summarize_df(df) for tf, df in tf_candles.items()}
    tf_text = ", ".join(MONITOR_TFS)

    # --- Build GPT Prompt ---
    # --- Build GPT Prompt ---
    analysis_scope = (
        f"Analyze the last candles for `{stock_code}` on `{selected_tf}` timeframe only."
        if selected_tf
        else f"Analyze the last candles for `{stock_code}` across timeframes: {tf_text}."
    )

    candle_json = json.dumps(data_summary, separators=(",", ":"))

    # Load news sentiment (use as secondary input)
    combined_sentiments, sentiment_ts = get_combined_news_sentiment()
    stock_sentiment = combined_sentiments.get(stock_code, "No Action")

    prompt = f"""
You are a professional quantitative trading assistant specializing in short-term stock forecasts.

Analyze the candle + indicator data for `{stock_code}` below. Focus only on the following signals:
**RSI14, MACD_Hist, ADX, ATR14, Volume, VWAP slope, EMA20 slope, and recent price momentum.**

Use these to evaluate whether a strong BUY setup exists. Combine this with the **news sentiment**
for `{stock_code}` = **{stock_sentiment}** (last updated {sentiment_ts}) to refine confidence.

Use the following **candle chart pattern details** (from recent TA-Lib detections)
to enhance your decision-making and validation of BUY signals:

{pattern_summary}

Your goal:
- Identify a BUY signal *only if* momentum + structure strongly support it.
- Assign a **confidence score (0–100)** that represents the probability the target will be achieved
  within 1–2 trading days.
- If confidence < 70, set `"signal": "No Action"`.

Guidelines:
1. stoploss < support < live price < entry < target < resistance
2. risk/reward ≥ 1.2
3. Intraday (5–30min): target 0.5–1.5%
   Mid-term (45min–1h): 1–3%
   Swing (4h): 2–5%
4. Combine technical + sentiment strength when assigning confidence.
5. Output **valid JSON only**, no explanations outside it.

Candle + Indicator Data (filtered):
{candle_json}

JSON Output Format:
{{
  "stock_code": "{stock_code}",
  "instrument_token": {instrument_token},
  "forecast": "gpt_forecast",
  "last_updated": "{datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}",
  "timeframes": {{
     "{selected_tf if selected_tf else '5min'}": {{
        "support": float,
        "resistance": float,
        "entry": float,
        "target": float,
        "stoploss": float,
        "signal": "BUY" or "No Action",
        "confidence": int,
        "reason": ["Concise reason why BUY setup is valid"]
     }}
  }}
}}
"""

    # --- Call GPT ---
    if client is None:
        print("[⚠️] OpenAI client not initialized")
        return stock_cfg

    # ============================================================
    # 📝 DEBUG LOG: Save full GPT prompt for inspection + token size
    # ============================================================
    try:
        import tiktoken  # lightweight tokenizer for token estimation

        log_dir = "debug_prompts"
        os.makedirs(log_dir, exist_ok=True)

        # File name example: debug_prompts/GMDCLTD_2025-10-30_12-41.txt
        ts_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{stock_code}_{ts_str}.txt")

        # --- Estimate token size ---
        # Select tokenizer based on model name (works for GPT-3.5, GPT-4, GPT-5-mini)
        try:
            encoding = tiktoken.encoding_for_model(model_to_use)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")  # default fallback

        token_count = len(encoding.encode(prompt))

        # --- Save prompt and token info ---
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(prompt)
            f.write(f"\n\n# ================================\n")
            f.write(f"# PROMPT TOKEN SIZE: {token_count}\n")
            f.write(f"# ================================\n")

        print(f"[🧾 Prompt Saved] {log_file} | INPUT TOKENS: {token_count}")

    except Exception as e:
        print(f"[⚠️ Failed to save prompt log for {stock_code}: {e}]")

    # --- GPT Model Selection & Universal Call ---
    model_to_use = route_model(stock_code)
    print(f"[🤖] {stock_code}: sending to {model_to_use}")

    try:
        # ✅ GPT-5 / Responses API (no temperature)
        if model_to_use.startswith("gpt-5"):
            response = client.responses.create(
                model=model_to_use,
                input=prompt,
                max_output_tokens=2500,
            )
            raw_text = getattr(response, "output_text", None) or str(response)

        # ✅ GPT-4 / Chat API (requires temperature)
        else:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2500,
            )
            raw_text = response.choices[0].message.content

        # Parse JSON safely
        gpt_json = _safe_json_parse(raw_text.strip())

    except Exception as e:
        print(f"[❌ GPT error {stock_code}: {e}]")
        traceback.print_exc()
        return stock_cfg

    if not gpt_json or "timeframes" not in gpt_json:
        print(f"[⚠️] Invalid GPT output for {stock_code}")
        return stock_cfg

    # --- Filter out weak confidence scores ---
    if gpt_json and "timeframes" in gpt_json:
        for tf, tfdata in gpt_json["timeframes"].items():
            conf = float(tfdata.get("confidence", 0))
            if conf < 70:
                print(f"[🚫 Rejected] {stock_code} {tf}: Confidence={conf}")
                tfdata["signal"] = "No Action"

    # --- Flatten GPT output ---
    result = {
        "stock_code": stock_code,
        "instrument_token": instrument_token,
        "forecast": "gpt_forecast",
        "last_updated": gpt_json.get("last_updated")
        or datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S"),
    }

    tf_map = {"5min": 1, "15min": 2, "30min": 3, "45min": 4, "1hour": 5, "4hour": 6}
    for tf, tfdata in gpt_json["timeframes"].items():
        if tf not in tf_map:
            continue
        suf = str(tf_map[tf])
        result[f"support{suf}"] = tfdata.get("support")
        result[f"resistance{suf}"] = tfdata.get("resistance")
        result[f"entry{suf}"] = tfdata.get("entry")
        result[f"target{suf}"] = tfdata.get("target")
        result[f"stoploss{suf}"] = tfdata.get("stoploss")
        result[f"signal{suf}"] = tfdata.get("signal")
        result[f"reason{suf}"] = json.dumps(tfdata.get("reason", []))

    # --- Enrich ---
    for tf, idx in tf_map.items():
        if tf not in tf_candles:
            continue
        df = tf_candles[tf]
        suf = str(idx)
        result[f"respected_S{suf}"] = count_respects(
            df, result.get(f"support{suf}"), "support"
        )
        result[f"respected_R{suf}"] = count_respects(
            df, result.get(f"resistance{suf}"), "resistance"
        )
        try:
            e, t = float(result.get(f"entry{suf}") or 0), float(
                result.get(f"target{suf}") or 0
            )
            result[f"entry_target_pct{suf}"] = (
                round(((t - e) / e) * 100, 2) if e else None
            )
        except Exception:
            result[f"entry_target_pct{suf}"] = None
        result[f"consolidation_score{suf}"] = compute_consolidation(df)

    ok, missing = check_all_tf_complete(result)
    if not ok:
        print(f"[⚠️ Incomplete GPT forecast for {stock_code}: {missing}]")

    # --- Filter weak signals ---
    valid_tfs = []
    for tf, idx in tf_map.items():
        suf = str(idx)
        sig = str(result.get(f"signal{suf}", "No Action")).upper()
        if sig != "BUY":
            continue
        try:
            entry = float(result.get(f"entry{suf}") or 0)
            target = float(result.get(f"target{suf}") or 0)
            stop = float(result.get(f"stoploss{suf}") or 0)
            if entry <= 0 or target <= 0 or stop <= 0:
                continue
            tdiff = abs((target - entry) / entry) * 100
            sdiff = abs((entry - stop) / entry) * 100
            if tdiff >= TARGET_DIFF_THRESHOLD and sdiff >= STOPLOSS_DIFF_THRESHOLD:
                valid_tfs.append(tf)
            else:
                result[f"signal{suf}"] = "No Action"
        except Exception:
            result[f"signal{suf}"] = "No Action"

    if not valid_tfs:
        print(f"[🚫 No valid BUY setups] {stock_code}")
        return result

    # --- Save to Redis ---
    try:
        redis_key = f"FORECASTGPT:{stock_code}"
        safe_map = {
            k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v))
            for k, v in result.items()
        }
        r.hset(redis_key, mapping=safe_map)
        if verbose:
            print(f"[💾] Saved GPT forecast → {redis_key}")
    except Exception as e:
        print(f"[⚠️ Redis save failed {stock_code}: {e}]")

    return result
