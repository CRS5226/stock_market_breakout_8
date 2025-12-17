# redis_to_csv.py
import pandas as pd
from redis_util import get_redis, load_from_redis

# 🔽 Change these as needed
STOCK_CODE = "GOKEX"
TIMEFRAME = "5min"
OUTPUT_FILE = f"{STOCK_CODE}_{TIMEFRAME}.csv"


def export_to_csv(stock_code: str, timeframe: str, output_file: str):
    r = get_redis()
    df = load_from_redis(r, stock_code, timeframe)

    if df.empty:
        print(f"❌ No data found in Redis for {stock_code} {timeframe}")
        return

    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df)} rows → {output_file}")


if __name__ == "__main__":
    export_to_csv(STOCK_CODE, TIMEFRAME, OUTPUT_FILE)
