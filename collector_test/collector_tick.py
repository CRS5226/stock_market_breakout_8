#!/usr/bin/env python3
# tick_collector.py — Strict market-synced tick → 5min & 15min candle builder
import os
import json
import pandas as pd
from datetime import datetime, timedelta, time
import pytz
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker
from indicator import add_indicators, normalize_config

# ===================================================
# CONFIG
# ===================================================
load_dotenv()
CONFIG_PATH = "config.json"
INDIA_TZ = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
MAX_CANDLES = 200
SAVE_RAW_TICKS = True


# ===================================================
# HELPERS
# ===================================================
def make_kite():
    kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))
    return kite


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _stock_dir(s):
    d = os.path.join("candle_data", s.upper())
    ensure_dir(d)
    return d


def _candle_csv(s, tf):
    return os.path.join(_stock_dir(s), f"{tf}.csv")


def _tick_csv(s):
    d = os.path.join("tick_data", s.upper())
    ensure_dir(d)
    return os.path.join(d, "ticks.csv")


def within_market_hours(ts: datetime) -> bool:
    t = ts.astimezone(INDIA_TZ).time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def next_5min_boundary(ts: datetime) -> datetime:
    """Return the next 5-min mark aligned to market (e.g., 11:00, 11:05...)."""
    ts = ts.astimezone(INDIA_TZ)
    minute = ((ts.minute // 5) + 1) * 5
    next_boundary = ts.replace(minute=0, second=0, microsecond=0) + timedelta(
        minutes=minute
    )
    return next_boundary


def market_aligned_bucket(ts: datetime, minutes=5):
    """Return (start, end) of current aligned candle window."""
    ts = ts.astimezone(INDIA_TZ)
    m = (ts.minute // minutes) * minutes
    start = ts.replace(minute=m, second=0, microsecond=0)
    end = start + timedelta(minutes=minutes)
    return start, end


def calculate_indicators(df):
    if df.empty:
        return df
    cfg = {
        "moving_averages": {"ma_fast": 9, "ma_slow": 20},
        "bollinger": {"period": 20, "std_dev": 2},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "adx": {"period": 14},
        "rsi": {"period": 14},
    }
    return add_indicators(df, normalize_config(cfg))


def save_ticks(stock, tick):
    if not SAVE_RAW_TICKS:
        return
    path = _tick_csv(stock)
    df = pd.DataFrame([tick])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_candle(stock, tf, candle):
    csv = _candle_csv(stock, tf)
    df = pd.DataFrame([candle])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if os.path.exists(csv):
        ex = pd.read_csv(csv, parse_dates=["Timestamp"])
        df = (
            pd.concat([ex, df])
            .drop_duplicates("Timestamp")
            .sort_values("Timestamp")
            .tail(MAX_CANDLES)
        )
    df.to_csv(csv, index=False)
    print(f"💾 {stock} {tf}: saved {candle['Timestamp']}")


# ===================================================
# BUILDER
# ===================================================
class LiveCandleBuilder:
    def __init__(self, stock):
        self.stock = stock
        self.cur5 = None
        self.cur15 = None
        self.cur5_end = None
        self.cur15_end = None
        self.ready_to_start = False
        self.start_time = None  # when we actually start collecting valid ticks

    def process_tick(self, tick):
        """Handle each incoming tick and update 5min/15min candles."""
        # Parse tick timestamp safely
        ts = tick.get("last_trade_time") or datetime.now(INDIA_TZ)
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            ts = INDIA_TZ.localize(ts)
        ts = ts.astimezone(INDIA_TZ)

        # Ignore if outside market hours
        if not within_market_hours(ts):
            return

        # Get price/volume
        price = float(tick.get("last_price", 0) or 0)
        vol = float(tick.get("volume_traded", 0) or 0)

        # Save every tick immediately
        save_ticks(
            self.stock,
            {"Timestamp": ts.isoformat(), "last_price": price, "volume": vol},
        )

        # Initialize candle windows on first tick
        if not self.ready_to_start:
            start5, end5 = market_aligned_bucket(ts, 5)
            start15, end15 = market_aligned_bucket(ts, 15)

            self.cur5_end = end5
            self.cur15_end = end15

            self.cur5 = {
                "Timestamp": self.cur5_end,
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": vol,
            }
            self.cur15 = {
                "Timestamp": self.cur15_end,
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": vol,
            }
            self.ready_to_start = True
            self.start_time = ts
            print(
                f"✅ Started collecting {self.stock} (5m candle ends {self.cur5_end.time()})"
            )
            return

        # Defensive check
        if self.cur5_end is None:
            self.cur5_end = market_aligned_bucket(ts, 5)[1]

        # Finalize multiple candles if time jumped
        while ts >= self.cur5_end:
            self.finalize_5m()
            self.cur5_end += timedelta(minutes=5)
            self.cur5 = {
                "Timestamp": self.cur5_end,
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": 0.0,
            }

        # Update current candle
        if self.cur5 is None:
            self.cur5 = {
                "Timestamp": self.cur5_end,
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": vol,
            }
        else:
            c = self.cur5
            c["High"] = max(c["High"], price)
            c["Low"] = min(c["Low"], price)
            c["Close"] = price
            c["Volume"] = c.get("Volume", 0) + vol

    def finalize_5m(self):
        """Finalize current 5m candle and pass to 15m aggregator."""
        if not self.cur5:
            return
        df = calculate_indicators(pd.DataFrame([self.cur5])).round(4)
        candle = df.iloc[0].to_dict()
        candle["Timestamp"] = candle["Timestamp"].replace(second=0, microsecond=0)
        save_candle(self.stock, "5min", candle)
        self.update_15m(candle)

    def update_15m(self, c5):
        ts = c5["Timestamp"]
        start15, end15 = market_aligned_bucket(ts, 15)
        if self.cur15_end is None:
            self.cur15_end = end15
        if self.cur15 is None:
            self.cur15 = {
                "Timestamp": end15,
                "Open": c5["Open"],
                "High": c5["High"],
                "Low": c5["Low"],
                "Close": c5["Close"],
                "Volume": c5["Volume"],
            }
        elif ts >= self.cur15_end:
            self.finalize_15m()
            self.cur15_end += timedelta(minutes=15)
            self.cur15 = {
                "Timestamp": self.cur15_end,
                "Open": c5["Open"],
                "High": c5["High"],
                "Low": c5["Low"],
                "Close": c5["Close"],
                "Volume": c5["Volume"],
            }
        else:
            c = self.cur15
            c["High"] = max(c["High"], c5["High"])
            c["Low"] = min(c["Low"], c5["Low"])
            c["Close"] = c5["Close"]
            c["Volume"] += c5["Volume"]

    def finalize_15m(self):
        if not self.cur15:
            return
        df = calculate_indicators(pd.DataFrame([self.cur15])).round(4)
        candle = df.iloc[0].to_dict()
        candle["Timestamp"] = candle["Timestamp"].replace(second=0, microsecond=0)
        save_candle(self.stock, "15min", candle)


# ===================================================
# MAIN LOOP
# ===================================================
def run_ticker():
    kite = make_kite()
    cfg = json.load(open(CONFIG_PATH))
    tokens, builders = [], {}

    for s in cfg["stocks"]:
        t = int(s["instrument_token"])
        tokens.append(t)
        builders[t] = LiveCandleBuilder(s["stock_code"])

    kws = KiteTicker(os.getenv("KITE_API_KEY"), os.getenv("KITE_ACCESS_TOKEN"))

    def on_connect(ws, resp):
        print("✅ Connected to KiteTicker (Market-aligned mode)")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for t in ticks:
            b = builders.get(t["instrument_token"])
            if b:
                try:
                    b.process_tick(t)
                except Exception as e:
                    print(f"[⚠️] {b.stock}: {e}")

    def on_close(ws, code, reason):
        print(f"❌ Closed {code} {reason}")

    kws.on_connect = on_connect
    kws.on_ticks = on_ticks
    kws.on_close = on_close
    print("🚀 Tick Collector started (immediate candle mode)...")
    kws.connect(threaded=False)


if __name__ == "__main__":
    run_ticker()
