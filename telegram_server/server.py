from flask import Flask, request
import requests
import os
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_30a")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Choose what symbol you want Python to compute prices for
SYMBOL = "RELIANCE.NS"   # <- CHANGE THIS TO YOUR SYMBOL

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Server is running", 200

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    requests.post(url, data=payload)

def get_live_price():
    data = yf.Ticker(SYMBOL).history(period="1m")
    return float(data["Close"].iloc[-1])

@app.route("/webhook", methods=["POST"])
def webhook():

    raw = request.get_data(as_text=True).strip()
    signal = raw.upper()  # TEST_BUY or TEST_SELL

    price = get_live_price()      # LIVE MARKET PRICE
    atr = 1.5                     # placeholder ATR until we calculate real ATR
    entry = price

    if signal == "TEST_BUY":
        sl = entry - atr
        tp = entry + 2 * atr
        send_telegram_message(
            f"<b>BUY SIGNAL</b>\nEntry: {entry}\nTarget: {tp}\nStoploss: {sl}"
        )

    elif signal == "TEST_SELL":
        sl = entry + atr
        tp = entry - 2 * atr
        send_telegram_message(
            f"<b>SELL SIGNAL</b>\nEntry: {entry}\nTarget: {tp}\nStoploss: {sl}"
        )

    else:
        send_telegram_message(f"Unknown Payload:\n{raw}")

    return "OK", 200

if __name__ == "__main__":
    public_ip = requests.get("https://api.ipify.org").text
    print("\n==============================")
    print(" Public IP:", public_ip)
    print(" Webhook URL: http://" + public_ip + ":5000/webhook")
    print("==============================\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
