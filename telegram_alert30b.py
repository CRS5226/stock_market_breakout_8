# telegram_alert30a.py

import json
import os
import csv
import logging
import asyncio
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from redis_util import get_redis

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_30b")
CHAT_IDS_FILE = "telegram_chat_ids_30b.json"
CONFIG_FILE = "config.json"
STOCKS_CSV = "stocks_reference.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= File Handling =========================


def load_chat_ids():
    if os.path.exists(CHAT_IDS_FILE):
        with open(CHAT_IDS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_chat_ids(data):
    with open(CHAT_IDS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"stocks": []}


def save_config(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_stock_reference():
    ref = {}
    if os.path.exists(STOCKS_CSV):
        with open(STOCKS_CSV, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ref[row["stock_name"].upper()] = {
                    "stock_code": row["stock_code"].upper(),
                    "stock_name": row["stock_name"].upper(),
                    "instrument_token": int(row["instrument_token"]),
                }
    return ref


# ========================= Notification Senders =========================


async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Telegram token missing.")
        return

    chat_data = load_chat_ids()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    for chat_id in chat_data.keys():
        try:
            await bot.send_message(chat_id=int(chat_id), text=message, parse_mode=None)
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")


def send_trade_alert(
    symbol: str, message: str, price: float, ts, source_tag: str = "ALGO"
):
    """
    Sends formatted trade alert to telegram.
    symbol: stock code
    message: descriptive message (entry/target/sl, breakout, etc.)
    price: current live price
    ts: candle timestamp (already ISO/datetime from df)
    source_tag: strategy/tag name (optional)
    """
    # Convert timestamp if it's datetime-like
    if not isinstance(ts, str):
        try:
            ts = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = str(ts)

    date_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

    telegram_msg = (
        f"🔔 *{source_tag} ALERT*\n"
        f"Symbol: `{symbol}`\n"
        f"Time: `{ts}`\n"
        f"Price: `{price}`\n"
        f"Message: {message}\n"
        f"Sent at: {date_ist}"
    )
    asyncio.run(send_telegram_message(telegram_msg))


def trigger_buy_alert_from_config(
    stock_cfg: dict, tf_index: int, reason: str = "Forecast signaled BUY"
):
    """
    Wrapper: Fetch levels from stock_cfg and send a BUY alert for the given TF.
    Keeps all alert formatting logic inside telegram code.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # Get levels
    entry, target, stop, support, resistance = get_levels_for_tf(stock_cfg, tf_index)
    if not entry or entry == 0:
        return  # skip invalid TFs

    # Timestamp
    ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

    # Send the formatted alert
    send_buy_alert(
        code=stock_cfg["stock_code"],
        tf_index=tf_index,
        ts=ts,
        reason=reason,
        entry=entry,
        target=target,
        stop=stop,
        support=support,
        resistance=resistance,
        live_price=entry,  # optional: replace with live Close if available
        stock_cfg=stock_cfg,  # 👈 pass config for precomputed values
    )


def get_levels_for_tf(stock_cfg: dict, tf_index: int):
    """
    Given stock_cfg and tf_index (0=1min, 1=5min, etc.),
    return entry, target, stoploss, support, resistance.
    """
    if tf_index == 0:  # 1min = base keys
        e, t, s = (
            stock_cfg.get("entry"),
            stock_cfg.get("target"),
            stock_cfg.get("stoploss"),
        )
        sup, res = stock_cfg.get("support"), stock_cfg.get("resistance")
    else:
        e = stock_cfg.get(f"entry{tf_index}")
        t = stock_cfg.get(f"target{tf_index}")
        s = stock_cfg.get(f"stoploss{tf_index}")
        sup = stock_cfg.get(f"support{tf_index}")
        res = stock_cfg.get(f"resistance{tf_index}")

    return e, t, s, sup, res


# def send_buy_alert(
#     code: str,
#     tf_index: int,
#     ts,
#     reason: str,
#     entry: float,
#     target: float,
#     stop: float,
#     support: float,
#     resistance: float,
#     live_price: float = None,
#     stock_cfg: dict = None,
# ):
#     """
#     Send BUY alert with:
#       - entry/target/stoploss diff calculated locally
#       - live price fetched from Redis FORECAST:{stock_code}.ohlcv
#     """

#     # 🕒 Timeframe mapping
#     TF_MAP = {
#         0: "1min",
#         1: "5min",
#         2: "15min",
#         3: "30min",
#         4: "45min",
#         5: "1hour",
#         6: "4hour",
#         7: "1day",
#     }

#     allowed_tfs = {3, 4, 5, 6}  # only major TFs
#     if tf_index not in allowed_tfs:
#         return

#     tf_label = TF_MAP.get(tf_index, f"tf{tf_index}")
#     tz = ZoneInfo("Asia/Kolkata")

#     # Format timestamp
#     if not isinstance(ts, str):
#         try:
#             ts = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
#         except Exception:
#             ts = str(ts)

#     date_ist = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S IST")

#     # ============================================================
#     # 🧮 COMPUTE LOCAL DIFFERENCES
#     # ============================================================
#     entry_target_pct = None
#     stop_pct = None

#     try:
#         if entry and target and entry != 0:
#             entry_target_pct = ((target - entry) / entry) * 100
#     except Exception:
#         pass

#     try:
#         if entry and stop and entry != 0:
#             stop_pct = ((entry - stop) / entry) * 100
#     except Exception:
#         pass

#     # ============================================================
#     # 📡 FETCH LIVE PRICE FROM REDIS
#     # ============================================================
#     try:
#         r = get_redis()
#         # ✅ Fetch the latest live price directly from 1min market data
#         market_key = f"MARKETDATA:{code}:1min"
#         data_map = r.hgetall(market_key)
#         if data_map:
#             latest_ts = max(data_map.keys())
#             latest_candle = json.loads(data_map[latest_ts])
#             if "Close" in latest_candle:
#                 live_price = float(latest_candle["Close"])

#     except Exception as e:
#         print(f"[⚠️ Live price fetch failed for {code}]: {e}")

#     # fallback
#     if live_price is None:
#         live_price = entry or 0.0

#     # ============================================================
#     # 💬 COMPOSE TELEGRAM ALERT MESSAGE
#     # ============================================================
#     message = (
#         f"🚀 *BUY SIGNAL*\n"
#         f"Symbol: `{code}`\n"
#         f"Timeframe: `{tf_label}`\n"
#         f"Timestamp: `{ts}`\n"
#         f"Live Price: `{live_price:.2f}`\n\n"
#         f"🎯 Entry: `{entry}` | Target: `{target}` | Stoploss: `{stop}`\n"
#         f"📊 Support: `{support}` | Resistance: `{resistance}`\n"
#     )

#     if entry_target_pct is not None:
#         message += f"📈 Target Diff: `{entry_target_pct:.2f}%`\n"
#     if stop_pct is not None:
#         message += f"📉 Stoploss Diff: `{stop_pct:.2f}%`\n"

#     # Optionally add risk/reward ratio
#     # if entry_target_pct and stop_pct and stop_pct != 0:
#     #     rr_ratio = entry_target_pct / stop_pct
#     #     message += f"⚖️ Risk:Reward: `{rr_ratio:.2f}`\n"

#     # message += f"\nReason: {reason}\nAlert Sent: {date_ist}"

#     # ============================================================
#     # 🚀 SEND TELEGRAM MESSAGE
#     # ============================================================
#     asyncio.run(send_telegram_message(message))
#     print(f"[🚀 BUY ALERT SENT] {code} {tf_label}")


def send_buy_alert(
    code: str,
    tf_index: int,
    ts,
    reason: str,
    entry: float,
    target: float,
    stop: float,
    support: float,
    resistance: float,
    live_price: float = None,
    stock_cfg: dict = None,
):
    """
    Send BUY alert with ALGO name included.
    """

    # 🆕 Extract ALGO name safely
    algo_name = None
    if isinstance(stock_cfg, dict):
        algo_name = stock_cfg.get("ALGO", None)

    # 🕒 Timeframe mapping
    TF_MAP = {
        0: "1min",
        1: "5min",
        2: "15min",
        3: "30min",
        4: "45min",
        5: "1hour",
        6: "4hour",
        7: "1day",
    }

    allowed_tfs = {0, 1, 2, 3, 4, 5, 6, 7}  # only major TFs
    if tf_index not in allowed_tfs:
        return

    tf_label = TF_MAP.get(tf_index, f"tf{tf_index}")
    tz = ZoneInfo("Asia/Kolkata")

    # Format timestamp
    if not isinstance(ts, str):
        try:
            ts = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = str(ts)

    date_ist = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S IST")

    # ============================================================
    # 🧮 COMPUTE LOCAL DIFFERENCES
    # ============================================================
    entry_target_pct = None
    stop_pct = None

    try:
        if entry and target and entry != 0:
            entry_target_pct = ((target - entry) / entry) * 100
    except Exception:
        pass

    try:
        if entry and stop and entry != 0:
            stop_pct = ((entry - stop) / entry) * 100
    except Exception:
        pass

    # ============================================================
    # 📡 FETCH LIVE PRICE FROM REDIS
    # ============================================================
    try:
        r = get_redis()
        market_key = f"MARKETDATA:{code}:1min"
        data_map = r.hgetall(market_key)
        if data_map:
            latest_ts = max(data_map.keys())
            latest_candle = json.loads(data_map[latest_ts])
            if "Close" in latest_candle:
                live_price = float(latest_candle["Close"])
    except Exception as e:
        print(f"[⚠️ Live price fetch failed for {code}]: {e}")

    if live_price is None:
        live_price = entry or 0.0

    # ============================================================
    # 💬 COMPOSE TELEGRAM ALERT MESSAGE
    # ============================================================

    # 🆕 Include ALGO in header (if available)
    algo_line = f"(ALGO: {algo_name})" if algo_name else ""

    message = (
        f"🚀 *BUY SIGNAL* {algo_line}\n"
        f"Symbol: `{code}`\n"
        f"Timeframe: `{tf_label}`\n"
        f"Timestamp: `{ts}`\n"
        f"Live Price: `{live_price:.2f}`\n\n"
        f"🎯 Entry: `{entry}` | Target: `{target}` | Stoploss: `{stop}`\n"
        f"📊 Support: `{support}` | Resistance: `{resistance}`\n"
    )

    if entry_target_pct is not None:
        message += f"📈 Target Diff: `{entry_target_pct:.2f}%`\n"
    if stop_pct is not None:
        message += f"📉 Stoploss Diff: `{stop_pct:.2f}%`\n"

    # ============================================================
    # 🚀 SEND TELEGRAM MESSAGE
    # ============================================================
    asyncio.run(send_telegram_message(message))
    print(f"[🚀 BUY ALERT SENT] {code} {tf_label} ({algo_name})")


def send_server_feedback():
    """Send a startup message with stock list from config.json."""
    config = load_config()
    stock_list = config.get("stocks", [])

    if not stock_list:
        message = "🚀 Server started, but no stocks found in config.json."
    else:
        stock_lines = "\n".join(
            [f"{i+1}. {s['stock_code']}" for i, s in enumerate(stock_list)]
        )
        message = (
            f"🚀 *Server Started*\n"
            # f"Fetching data for {len(stock_list)} stocks:\n{stock_lines}"
        )

    try:
        asyncio.run(send_telegram_message(message))
    except Exception as e:
        print(f"[⚠️ Telegram Feedback Error] {e}")


def send_config_update(status: str, symbol: str):
    message = f"\n{status}"
    asyncio.run(send_telegram_message(message))


def send_pipeline_status(status: str, symbol: str):
    message = f"\n*Pipeline {status}* for `{symbol}`"
    asyncio.run(send_telegram_message(message))


def send_error_alert(error: str):
    message = f"\n*ERROR Occurred:*\n```{error}```"
    asyncio.run(send_telegram_message(message))


# ========================= Command Handlers =========================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = str(update.message.chat_id)
    chat_data = load_chat_ids()

    if chat_id not in chat_data:
        chat_data[chat_id] = {
            "username": user.username or "",
            "first_name": user.first_name or "",
            "registered_at": datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
                "%Y-%m-%d %H:%M:%S IST"
            ),
        }
        save_chat_ids(chat_data)

    await update.message.reply_text(
        "Welcome! You are now registered to receive alerts.\n\n"
        "Here are the commands:\n"
        "/liststocks - List tracked stocks\n"
        "/addstock - Add stock from CSV menu\n"
        "/removestock - Remove stock from tracking\n"
        "/updatestock CODE KEY VALUE - Update stock property\n"
        "/help - Show this help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    commands = """
/liststocks - List tracked stocks
/addstock - Add stock from CSV menu
/removestock - Remove stock from tracking
/updatestock CODE KEY VALUE - Update stock property
/help - Show this help
"""
    await update.message.reply_text(commands)


async def list_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    stocks = config.get("stocks", [])
    if not stocks:
        await update.message.reply_text("No stocks are currently tracked.")
    else:
        msg = "\n".join(
            [f"{s['stock_code']} ({s['instrument_token']})" for s in stocks]
        )
        await update.message.reply_text(msg)


# ========================= Add Stock =========================


async def add_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(0, 26, 6):
        keyboard.append(
            [
                InlineKeyboardButton(l, callback_data=f"letter_{l}")
                for l in letters[i : i + 6]
            ]
        )
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Select the first letter of the stock name:", reply_markup=reply_markup
    )


async def letter_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    letter = query.data.split("_")[1]
    stock_ref = load_stock_reference()
    stocks = [name for name in stock_ref.keys() if name.startswith(letter)]

    if not stocks:
        await query.edit_message_text(f"No stocks found starting with {letter}.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                name, callback_data=f"stock_{stock_ref[name]['stock_code']}"
            )
        ]
        for name in stocks
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Select a stock starting with {letter}:", reply_markup=reply_markup
    )


async def stock_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    stock_code = query.data.split("_")[1]
    stock_ref = load_stock_reference()
    stock_info = next(
        (v for v in stock_ref.values() if v["stock_code"] == stock_code), None
    )

    if not stock_info:
        await query.edit_message_text(f"Stock {stock_code} not found.")
        return

    config = load_config()
    if any(s["stock_code"] == stock_code for s in config["stocks"]):
        await query.edit_message_text(f"{stock_code} is already in your list.")
        return

    new_stock = {
        "stock_code": stock_info["stock_code"],
        "instrument_token": stock_info["instrument_token"],
        "support": 0.0,
        "resistance": 0.0,
        "volume_threshold": 0,
        "bollinger": {"mid_price": 0, "upper_band": 0, "lower_band": 0},
        "macd": {
            "signal_line": 0,
            "histogram": 0,
            "ma_fast": 0,
            "ma_slow": 0,
            "ma_signal": 0,
        },
        "adx": {"period": 14, "threshold": 20},
        "moving_averages": {"ma_fast": 9, "ma_slow": 20},
        "inside_bar": {"lookback": 1},
        "candle": {"min_body_percent": 0.7},
        "reason": [],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": "Hold",
    }

    config["stocks"].append(new_stock)
    save_config(config)
    await query.edit_message_text(f"✅ {stock_info['stock_name']} added successfully.")


# ========================= Remove Stock =========================


async def remove_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    stocks = config.get("stocks", [])
    if not stocks:
        await update.message.reply_text("No stocks to remove.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                s["stock_code"], callback_data=f"removestock_{s['stock_code']}"
            )
        ]
        for s in stocks
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Select a stock to remove:", reply_markup=reply_markup
    )


async def remove_stock_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    stock_code = query.data.split("_")[1]

    config = load_config()
    stocks = config.get("stocks", [])
    updated_stocks = [s for s in stocks if s["stock_code"] != stock_code]

    if len(updated_stocks) == len(stocks):
        await query.edit_message_text(f"Stock {stock_code} not found in list.")
        return

    config["stocks"] = updated_stocks
    save_config(config)
    await query.edit_message_text(f"🗑 Stock {stock_code} removed successfully.")


# ========================= Main Bot Runner =========================


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("No TELEGRAM_BOT_TOKEN found in environment variables")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Core commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("liststocks", list_stocks))
    app.add_handler(CommandHandler("addstock", add_stock))
    app.add_handler(CommandHandler("removestock", remove_stock))

    # Callbacks for menus
    app.add_handler(CallbackQueryHandler(letter_selected, pattern=r"^letter_"))
    app.add_handler(CallbackQueryHandler(stock_selected, pattern=r"^stock_"))
    app.add_handler(
        CallbackQueryHandler(remove_stock_selected, pattern=r"^removestock_")
    )

    logger.info("Telegram bot polling started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
