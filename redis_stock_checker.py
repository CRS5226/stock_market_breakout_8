# redis_stock_checker.py
import pandas as pd
from redis_util import get_redis, load_from_redis

# 🔽 Change this stock code to check another one
STOCK_CODE = "PARADEEP"

TIMEFRAMES = ["5min", "30min", "1hour"]


def check_stock(stock_code: str):
    r = get_redis()
    print(f"\n📊 Checking Redis data for stock: {stock_code}\n")

    for tf in TIMEFRAMES:
        df = load_from_redis(r, stock_code, tf)
        if df.empty:
            print(f"⏳ {tf}: No data found.")
            continue

        print(f"⏱ {tf} timeframe:")
        print(f"   Columns → {list(df.columns)}")
        print(f"   Rows    → {len(df)}\n")


if __name__ == "__main__":
    check_stock(STOCK_CODE)
