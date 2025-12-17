from fastapi import FastAPI, Request, BackgroundTasks
import requests
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import uvicorn

# ======================================================
# ENV
# ======================================================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_30a")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI()


# ======================================================
# TELEGRAM
# ======================================================
def send_telegram_message(text: str):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=2)
    except:
        pass


# ======================================================
# DATA HELPERS
# ======================================================
def fetch_ohlcv(symbol: str, bars=80):
    df = yf.download(
        symbol + ".NS",
        interval="30m",
        period="5d",
        progress=False,
    )
    return df.tail(bars)


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ======================================================
# STRATEGY 1 — BASE BREAKOUT
# ======================================================
def base_breakout_buy(symbol):
    df = fetch_ohlcv(symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    atr14 = atr(df).iloc[-1]

    base_high = high[:-1].tail(10).max()
    base_low = low[:-1].tail(10).min()

    entry = base_high * 1.0009
    stop = base_low - 0.15 * atr14
    if stop >= entry:
        stop = entry * 0.995

    risk = entry - stop
    atr_pct = (atr14 / close.iloc[-1]) * 100
    rr = 1.5 + min(0.6, atr_pct / 10)
    target = entry + rr * risk

    return entry, target, stop, close.iloc[-1]


# ======================================================
# STRATEGY 2 — BREAKOUT RETEST
# ======================================================
def breakout_retest_buy(symbol):
    df = fetch_ohlcv(symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    atr14 = atr(df).iloc[-1]
    resistance = high[5:-1].tail(13).max()

    entry = resistance * 1.00055
    retest_low = low.tail(7).min()
    stop = retest_low - 0.35 * atr14
    if stop >= entry:
        stop = entry * 0.9935

    risk = entry - stop
    atr_pct = (atr14 / close.iloc[-1]) * 100
    rr = 1.56 + min(0.82, atr_pct / 6.5)
    target = entry + rr * risk

    return entry, target, stop, close.iloc[-1]


# ======================================================
# STRATEGY 3 — HL + BOS
# ======================================================
def hl_bos_buy(symbol):
    df = fetch_ohlcv(symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    atr14 = atr(df).iloc[-1]

    entry = high.tail(16).max() * 1.00015
    stop = min(low.tail(6).min() - 0.10 * atr14, entry * 0.995)

    risk = entry - stop
    sl_pct = (risk / entry) * 100
    rr = min(1.4 + (sl_pct / 100), 2.2)
    target = entry + rr * risk

    return entry, target, stop, close.iloc[-1]


# ======================================================
# STRATEGY 4 — EMA + RSI CONFLUENCE
# ======================================================
def ema_rsi_buy(symbol):
    df = fetch_ohlcv(symbol)
    close, high, low = df["Close"], df["High"], df["Low"]

    atr14 = atr(df).iloc[-1]
    ema20 = ema(close, 20)

    body_mid = (close.iloc[-1] + df["Open"].iloc[-1]) / 2
    entry_candidate = ema20.iloc[-1] + 0.2 * atr14
    entry = min(max(body_mid, entry_candidate), high.iloc[-1] * 1.002)

    stop = low.tail(3).min() - 0.60 * atr14

    risk = entry - stop
    atr_pct = (atr14 / close.iloc[-1]) * 100

    rr = 1.0
    if atr_pct >= 1.0:
        rr = 1.3
    if atr_pct >= 1.3:
        rr = 1.6
    if atr_pct >= 1.5:
        rr = 1.9

    if df["RSI"] if "RSI" in df else False:
        rr += 0.3

    target = entry + rr * risk
    return entry, target, stop, close.iloc[-1]


# ======================================================
# STRATEGY ROUTER
# ======================================================
STRATEGY_MAP = {
    "BASE_BREAKOUT_BUY": base_breakout_buy,
    "BREAKOUT_RETEST_BUY": breakout_retest_buy,
    "HL_BOS_BUY": hl_bos_buy,
    "EMA_RSI_BUY": ema_rsi_buy,
}


# ======================================================
# WEBHOOK
# ======================================================
@app.post("/webhook")
async def webhook(request: Request, bg: BackgroundTasks):

    raw = (await request.body()).decode().strip()
    parts = raw.split("|")

    if len(parts) != 3:
        bg.add_task(send_telegram_message, f"❌ Invalid Alert: {raw}")
        return {"status": "invalid"}

    strategy, symbol, timeframe = parts

    if strategy not in STRATEGY_MAP:
        bg.add_task(send_telegram_message, f"❌ Unknown Strategy: {strategy}")
        return {"status": "unknown"}

    entry, target, stop, price = STRATEGY_MAP[strategy](symbol)

    tgt_pct = ((target - entry) / entry) * 100
    sl_pct = ((entry - stop) / entry) * 100
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = (
        f"🚀 *BUY SIGNAL*\n"
        f"Symbol: `{symbol}`\n"
        f"Timeframe: `{timeframe}`\n"
        f"Timestamp: `{ts}`\n"
        f"Live Price: `{price:.2f}`\n\n"
        f"🎯 Entry: `{entry:.2f}` | Target: `{target:.2f}` | Stoploss: `{stop:.2f}`\n"
        f"📈 Target Diff: `{tgt_pct:.2f}%`\n"
        f"📉 Stoploss Diff: `{sl_pct:.2f}%`"
    )

    bg.add_task(send_telegram_message, msg)
    return {"status": "ok"}


# ======================================================
# START
# ======================================================
if __name__ == "__main__":
    public_ip = requests.get("https://api.ipify.org").text
    print(f"\nWebhook URL: http://{public_ip}:5000/webhook\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)
